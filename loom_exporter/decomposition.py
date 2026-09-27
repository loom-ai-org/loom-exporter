"""`Decomposition` -- how a model becomes exported topologies (BACKLOG.md P4.0.3).

This is the axis `profile` was introduced to be and never became. It answers one question, separately
from *which* family a config belongs to: does this model export as one traced graph, as independently
traced submodules assembled per a `ModularExportSpec`, or as N independently traced phases merged into
one GGUF? Before this module that answer was encoded in the config's CLASS
(`LMMonolithicCausalModelExportConfig` vs `LMModularCausalModelExportConfig`), which is why exporting
the same LFM2 checkpoint two ways needed two classes and two registry entries built from two different
types rather than one type with a field set differently.

**Why a strategy object rather than a mode string on one config.** The three forms need genuinely
different data -- `Modular` needs a `ModularExportSpec` and a dummy sequence length chosen not to
collide with any static dim; `Flattened` needs a trace length and (for quantizable families) a quantize
mode; `MultiPhase` needs the phase list itself. A single config carrying every field with a string
selecting which subset is live makes invalid states representable and pushes the checking into
`export()`. Each decomposition instead carries its own fields, and a config that names one must supply
that decomposition's own hooks.

**The hooks are a protocol, not a base class.** Each `Decomposition` documents what it reads off the
config it is handed; families implement only what their own decomposition asks for. That is what lets a
family template stay family-shaped (how to load THIS checkpoint, what wrapper THIS model needs) while
the decomposition stays decomposition-shaped (trace once vs. trace per submodule vs. trace per phase).

**What is genuinely a choice, and what is not.** Only the causal-LM family currently has two
decompositions available for the same checkpoint -- LFM2 exports either way, which is a caller decision
(`--model lfm2-monolithic` / `--model lfm2-modular`), not a property of the checkpoint. Kokoro cannot be
exported flattened and Qwen3 has no phases: for those families the decomposition is a structural fact
about the model, declared once in the family's own config rather than chosen per run. The field is
universal; the *choice* is not, and a family that only ever has one answer says so by defaulting it.
"""
from dataclasses import dataclass, field
from typing import Optional

from .modular_export import ModularExportSpec
from .spec_protocol import NestedSpec, Unchecked


def _rss_note() -> str:
    """` (rss N.N GiB)`, or `""` where this process cannot read its own resident set.

    Exports of this size are memory-bound before they are anything else -- BACKLOG.md P5.0 exists
    because `MultiPhase.export`'s peak is a sum over phases -- and the one number that would have said
    so was never printed. Reported per phase, beside the node and weight counts, because the phase
    boundary is where the sum grows and where P5.0's remaining changes would show up.

    /proc, not `psutil`: this is a diagnostic line, not a dependency.
    """
    try:
        with open("/proc/self/statm") as f:
            pages = int(f.read().split()[1])
    except (OSError, IndexError, ValueError):
        return ""
    import resource

    return f" (rss {pages * resource.getpagesize() / (1 << 30):.1f} GiB)"


def _check_phase_weight_namespaces(phase_outputs) -> None:
    """No phase's topology reads a weight another phase produced.

    `phase_outputs` is `[(phase name, {topology name: topology}, {weight name: array}), ...]` -- what
    each `convert_phase` returned, in order.

    **This is what makes packing a phase's weights early equal to packing them at the end**
    (BACKLOG.md P5.0). Both quantization gates are questions about ALL topologies -- `name in
    _collect_mul_mat_weight_names()` and the fold's `usage[name] == {(op, 0)}` -- and a phase exporter
    can only see its own. The answers coincide because a multi-phase export leaves `flat_namespace`
    False and every weight is therefore written as `{func_name}.{weight}`, so no other phase's graph
    can name this phase's tensors.

    That is a property of the naming convention rather than of anything a family declares, which is
    exactly why it is checked rather than assumed: a family that ever writes weights some other way
    would get a *plausible* artifact -- a conv kernel folded for a consumer that wanted its declared
    shape, or a weight packed for an op the other phase reads it with -- and the failure would be a
    wrong tensor, not an error.

    **Ownership is by PHASE, not by topology name**, which is the whole reason this takes three
    columns instead of comparing a weight's owner against the name of the topology reading it. One
    phase routinely owns several topologies under names that are not its own: an `extra_streams` alias
    (`decoder` and `decoder_uncond` are one graph run as two streams, family 10's CFG pair), and a
    `RecurrentPhase`'s cells (`text_encoder_lstm` emits `..._fwd` and `..._bwd`, or `..._l0_fwd` for a
    stack). Keyed on the topology's name, every one of those reads as a violation and would have to be
    excused by a rule that also excuses the thing being looked for.
    """
    owner_of_weight, owner_of_topology = {}, {}
    for phase_name, topologies, weights in phase_outputs:
        for name in weights:
            owner_of_weight[name] = phase_name
        for name in topologies:
            owner_of_topology[name] = phase_name

    trespass = {}
    for phase_name, topologies, _ in phase_outputs:
        for topology_name, topo in topologies.items():
            for node in topo.get("nodes", []):
                for input_name in node.get("inputs") or []:
                    producer = owner_of_weight.get(input_name)
                    # A topology reads plenty of names no phase produced -- its own graph inputs,
                    # intermediate values, the driver's tensors. Only a name some phase DID produce
                    # can be read by the wrong one.
                    if producer is not None and producer != phase_name:
                        trespass.setdefault((producer, topology_name), set()).add(input_name)

    if trespass:
        (producer, reader), names = sorted(trespass.items())[0]
        raise ValueError(
            f"topology {reader!r} (phase {owner_of_topology[reader]!r}) reads weight(s) "
            f"{sorted(names)[:4]} that phase {producer!r} produced. Multi-phase weights carry their "
            f"own phase's `{{func_name}}.` prefix precisely so this cannot happen, and each phase "
            f"packs its own weights (quantization eligibility, the conv-kernel fold) from its own "
            f"topology alone -- which is only the same answer as the merged one while that holds. "
            f"See BACKLOG.md P5.0."
        )


class Decomposition:
    """Base for the three real shapes. Subclasses implement `export(config)` and document which hooks
    they read off `config`."""

    def export(self, config) -> str:
        raise NotImplementedError

    def driver_builder(self, config):
        """The `driver_builder.DriverBuilder` that assembles this decomposition's driver for `config`,
        or `None` if this decomposition does not build one.

        **The builder is selected by the decomposition, not owned by the family**
        (`EXPORT-PREPARATION.md` §5 decision 2, BACKLOG.md P4.0.6). The orchestration shape a driver has
        is a property of how the model was decomposed -- one traced graph means prefill-then-argmax, a
        submodule chain means thread the hidden state through it, N phases means whatever that family's
        phases compose into -- so a fourth decomposition (the cross-attention AR decode shape, families
        2 + 6) arrives bringing its own builder rather than every family in it declaring one.

        `config` is passed because the *contents* still are family-specific: `MultiPhase` reads which
        phases and components this family declared. What the decomposition fixes is the shape.

        And `config` is the whole signature, as of P4.0.18. There was a `**context` besides it, in the
        same "hooks are a protocol, not a base class" spirit as `export()` itself, and exactly one thing
        was ever passed through it: `MultiPhase`'s `source=`, the driver text after `render_driver` had
        substituted its generated samplers in. With the substitution gone there is no text to hand over
        -- a builder is built from the config's own component list -- so the parameter went with it
        rather than staying as an extension point nobody had asked for twice.

        Returning `None` is a real answer, not a stub: `Flattened` covers both the synthesized
        prefill path and the bespoke hand-built-Program workflow, and the latter transpiles a MIL `main`
        function op by op rather than assembling components (`LoomGGUFExporter.transpile_to_lua`).
        """
        return None


@dataclass
class Flattened(Decomposition):
    """One traced forward pass -> one `main_topology` topology. Qwen3, LFM2-monolithic, and all three NeMo
    ASR encoders.

    Config hooks: `prepare_environment()` (optional, defaults to a no-op via `LoomExportConfig`),
    `load_model()`, `build_trace(model)` -> `(wrapper, dummy_inputs, mil_inputs)`, `backend_kwargs()`,
    and `export_architecture()`."""

    def export(self, config) -> str:
        import coremltools as ct
        import torch

        from .register import LoomGGUFBackend
        from .spec_protocol import LinkChecker

        config.prepare_environment()
        model = config.load_model()
        # The config's own links (P4.0.5), before the trace. Nothing walks into `config.output`-style
        # NestedSpec fields: those are checked where their context exists -- for the ASR family, inside
        # the traced wrapper's forward, which is the only moment the real outputs exist.
        checker = LinkChecker()
        checker.check(config)
        checker.provide(model=model)
        checker.finish()
        wrapper, dummy_inputs, mil_inputs = config.build_trace(model)

        traced = torch.jit.trace(wrapper, dummy_inputs)
        program = ct.convert(
            traced,
            inputs=mil_inputs,
            convert_to="milinternal",
            # Load-bearing and NOT a per-model choice: ct.convert()'s default FP16-casts every constant
            # weight even for convert_to="milinternal" -- root-caused via Conformer-CTC's CONV_2D
            # subsampling stage, but it applies to every model this exporter has ever produced. See
            # nemo_asr_export.py's module docstring.
            compute_precision=ct.precision.FLOAT32,
        )

        LoomGGUFBackend()(
            program,
            output_path=config.output_path,
            architecture=config.export_architecture(),
            **config.resolved_backend_kwargs(),
        )
        print(f"SUCCESS! Flattened model exported cleanly to: {config.output_path}")
        return config.output_path


@dataclass
class Modular(Decomposition):
    """Independently traced submodules (embedding, rotary table, each decoder layer, final norm, head)
    assembled into one multi-Function Program per `ModularExportSpec` -- LFM2's modular form, and the
    only split mechanism left after the `"atomic"` profile's retirement (EXPORT-ROADMAP.md R7). See
    `modular_export.py`'s own module docstring.

    Config hooks: `prepare_environment()`, `load_model()`, `modular_dummy_inputs()`, `backend_kwargs()`,
    `export_architecture()`, and `max_seq_len`."""

    spec: ModularExportSpec
    # A dummy sequence length deliberately NOT equal to any of the model's own static dims (e.g. LFM2's
    # batch=1, hidden_size=1024, num_attention_heads=16, head_dim=64, vocab_size=65536) -- export_modular
    # marks an axis dynamic when its captured size equals this value, so a collision would wrongly mark a
    # static axis dynamic (or vice versa).
    dummy_seq_len: int = 37

    __links__ = {
        "spec": NestedSpec(
            where="export_modular, which checks every declared attribute path against the real "
                  "nn.Module before it traces anything"
        ),
    }
    __unchecked__ = {
        "dummy_seq_len": Unchecked(
            "a sentinel length, and the one field here where a link would be actively misleading. Its "
            "correctness condition is a NON-collision with any of the model's own static dims -- "
            "checkable in principle, but only against the specific checkpoint, and a wrong value does "
            "not fail: it marks a static axis dynamic (or the reverse) and exports something plausible. "
            "The real guard is the per-model reference test, not a declaration."
        ),
    }

    def export(self, config) -> str:
        from .modular_export import export_modular
        from .register import LoomGGUFBackend
        from .spec_protocol import LinkChecker

        config.prepare_environment()
        model = config.load_model()
        # The config's own links (P4.0.5). `decomposition` is a NestedSpec, so this deliberately does
        # not walk into `self.spec` -- `export_modular` checks that against the real module itself,
        # before it traces anything.
        checker = LinkChecker()
        checker.check(config)
        checker.provide(model=model)
        checker.finish()
        dummy_inputs = config.modular_dummy_inputs(self.dummy_seq_len)

        print("Tracing each submodule standalone...")
        result = export_modular(
            model, self.spec, dummy_inputs,
            seq_len=self.dummy_seq_len, max_seq_len=config.max_seq_len,
        )

        print("Compiling to GGUF (modular blueprint)...")
        # `flat_namespace` is deliberately NOT among backend_kwargs() for this decomposition, and that
        # is load-bearing rather than an omission: a modular Program's per-submodule functions each need
        # their own `{func_name}.{weight}` prefix, which is exactly what leaving this False preserves
        # (topology_ops.py reads it in 8 places). See BACKLOG.md P4.0.3.
        LoomGGUFBackend()(
            result.program,
            output_path=config.output_path,
            architecture=config.export_architecture(),
            modular_layout=result,
            **config.resolved_backend_kwargs(),
        )
        print(f"SUCCESS! Modular-blueprint model exported cleanly to: {config.output_path}")
        return config.output_path


@dataclass
class MultiPhase(Decomposition):
    """N independently traced phases, weights merged, written as one GGUF -- every TTS family
    (Kokoro 2 phases, VITS 3, Matcha 4, Supertonic 4, StyleTTS2 3).

    Not an alternative to `Flattened` for any model that uses it: these phases exist because the model
    genuinely cannot be traced as one graph (separate checkpoints, a Python-side sampler between stages,
    or a submodule that is not the model's own `forward`). Declared, not chosen.

    Config hooks: `phases()`, `samplers()`, `driver_components()`, `driver_script_path`,
    `architecture`."""

    def driver_builder(self, config):
        """A `MultiPhaseDriverBuilder` around this family's component list (P4.0.6/C.4-C.8).

        One shape, as of P4.0.18. There was a second until then -- `RawLuaDriver` around the whole
        hand-written `.lua`, selected when `driver_components()` returned `None` -- and it was the
        adoption step every family passed through on its way to being peeled. All five are peeled, so
        the branch selected nothing and kept `render_driver`'s marker substitution alive behind it.
        `RawLuaDriver` itself stays: it is how the *next* hand-written driver is adopted, in a commit
        whose gate is byte-identity, and its registry entry has said so since D.1.
        """
        from .driver_components import MultiPhaseDriverBuilder

        return MultiPhaseDriverBuilder(peeled=config.driver_components(),
                                      input_aliases=config.driver_input_aliases())

    def export(self, config) -> str:
        from .exporter import WeightPacking
        from .phase_conversion import convert_phase
        from .spec_protocol import LinkChecker

        config.prepare_environment()
        # One checker for the whole export (P4.0.5): every spec this config declares shares a single
        # deferral ledger, so `finish()` below is a real statement about the export rather than about
        # whichever call site happened to have context in hand.
        checker = LinkChecker()
        checker.check(config)
        phase_topologies = {}
        fused_geometry = {}
        # `(phase name, its topologies, its weights)` per phase, in order. Two things read it: the
        # weight merge, which wants the first and third columns, and the namespace invariant that
        # per-phase packing rests on, which wants all three -- see `_check_phase_weight_namespaces`.
        phase_outputs = []
        packing = WeightPacking()
        phases = config.phases()
        # Every phase's axis declarations, checked before the first (slow) trace: an axis name outside
        # axes.py's vocabulary, or a declared_axes entry naming an input this phase does not declare.
        # Both are answerable from the declaration alone, which is why they run here and not after.
        for phase in phases:
            checker.check(phase, f"{type(phase).__name__}({phase.name!r})")
        # Resolved ONCE, here, and for two reasons beyond not doing the work twice. It is where the
        # quantize type each phase packs to comes from (P5.0's second change), so a phase and the
        # artifact cannot disagree about it -- `LoomGGUFExporter` applies the `$LOOM_QUANTIZE` fallback
        # at both ends, and the child inherits the environment. And `contract()`/`hparams()` read what
        # `phases()` left on the config, so asking for them before the model is released is the order
        # that stays correct for a family whose answer ever needs more than an attribute.
        out_kwargs = dict(config.resolved_backend_kwargs())
        quantize = out_kwargs.get("quantize")

        for phase in phases:
            result = convert_phase(phase, quantize=quantize)
            phase_topologies.update(result.topologies)
            phase_outputs.append((phase.name, result.topologies, result.weights))
            packing.absorb(result.packing)
            # The cache geometry of whatever this phase fused, so the output exporter can still write
            # it: that exporter has no program of its own, and until P5.0's first reduction this was
            # `traced_programs.append(mil_prog)` -- the whole converted program, kept for a handful of
            # integers. Extracted for every phase rather than only the fused ones, because
            # `fused_geometry` asks the question and a list that depended on the answer would have to
            # be rebuilt the day a second geometry is added.
            for kind, records in result.geometry.items():
                fused_geometry.setdefault(kind, []).extend(records)

        _check_phase_weight_namespaces(phase_outputs)
        return self._write(config, checker, phases, phase_topologies, phase_outputs,
                           fused_geometry, packing, out_kwargs)

    def _write(self, config, checker, phases, phase_topologies, phase_outputs, fused_geometry,
               packing, out_kwargs) -> str:
        """Merge what the phases produced, build the driver, write the GGUF.

        Split out of `export` because that half is now a loop over `convert_phase` and this half is
        everything that needs all the phases at once -- and because what a phase hands over is a
        `PhaseResult` rather than four accumulating locals, the seam is where it can be drawn.
        """
        from .driver_builder import DriverContext
        from .exporter import LoomGGUFExporter
        from .multi_phase_export import RecurrentPhase, merge_phase_weights

        # A cached phase's capacity reaches the writer from the phase that declared it, not from a
        # second `backend_kwargs()` entry a family would have to keep in step: `_kv_cache_geometry`
        # counts the fused nodes and needs the capacity beside them, and `ExportPhase.kv_cache_size` is
        # already the one place it is stated.
        capacities = {phase.kv_cache_size for phase in phases
                      if getattr(phase, "fuse_attention", False) and phase.kv_cache_size is not None}
        if len(capacities) > 1:
            raise ValueError(
                f"phases declare different kv_cache_size values ({sorted(capacities)}). One KvCache is "
                "allocated for the whole model, so the phases that share it must agree on its capacity."
            )
        # `backend_kwargs()` reaches the OUTPUT exporter, not just the per-phase ones -- which is where
        # anything about the artifact as a whole belongs (the tokenizer vocab, notably). It was dropped
        # entirely before, so a multi-phase family had no way to say anything about its own GGUF.
        # Resolved by `export` while the model was still loaded; see there.
        if capacities:
            out_kwargs["kv_cache_size"] = capacities.pop()
        out_exporter = LoomGGUFExporter(
            None, output_path=config.output_path, architecture=config.architecture, **out_kwargs,
        )
        out_exporter.topologies = phase_topologies
        out_exporter.weights = merge_phase_weights(
            [(name, weights) for name, _, weights in phase_outputs])
        out_exporter.phase_geometry = fused_geometry
        # The phases packed their own weights, so the writer's job is to hash, alias and add -- and
        # this is where it learns which GGML type each name was packed to. `write_gguf` still calls
        # `pack_weights` after it, which now finds only the driver's own `loom.get_weight` tensors
        # unpacked (Kokoro's and Supertonic's default voice styles) and packs those.
        out_exporter.packing = packing

        # Every check this used to route through `render_driver` now runs inside the builder, over the
        # same `checker` (P4.0.18): a `FlowMatchingSpec` reaches it as `FlowMatchingSampler.sub_specs`,
        # and a `run_subgraph` call still written by hand reaches it as the `RunSubgraphCall` its
        # `LuaFragment` parsed out -- with the fragment's own file and line on the label.
        driver_source = self.driver_builder(config).render(
            DriverContext(
                topologies=out_exporter.topologies,
                # A recurrent cell has no root axis -- it is one timestep, with no time
                # dimension for a symbol to range over.
                axes={phase.name: phase.root_axis for phase in phases
                      if not isinstance(phase, RecurrentPhase)},
                weights=out_exporter.weights,
            ),
            checker=checker,
        )
        # Nothing may be written until every declared link has actually run. A link that deferred and
        # never became checkable reads as validated and is not -- see spec_protocol's module docstring.
        checker.finish()
        out_exporter.write_gguf(driver_source)
        print(f"wrote {config.output_path}")
        return config.output_path
