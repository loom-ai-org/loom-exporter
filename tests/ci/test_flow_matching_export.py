"""
Checks `flow_matching_export.py` (EXPORT-IMPROVEMENT.md item 4). Two things matter here:

* the generated Euler sampler must reproduce, exactly, the loop the Matcha/Supertonic drivers hand-wrote
  -- those two are numerically pinned end-to-end by `test_e2e_{matcha,supertonic}_mil_lua_driver.cpp`,
  so what is worth testing at this level is the *shape* of what gets emitted and, above all,
* the export-time validation, which is the actual reason these are specs rather than hand-written Lua.
  A `run_subgraph` call whose argument names don't match the topology's declared inputs is otherwise
  only caught deep inside the engine at run time, with nothing pointing back at the driver line.
"""
import unittest

import sys
from pathlib import Path

from loom_exporter.paths import CONVERTERS, driver_dir
from loom_exporter.flow_matching_export import EstimatorSpec, FlowMatchingSpec, render_sampler
from loom_exporter.spec_protocol import check_links


def _check(spec, **topologies):
    """`spec`'s links against a fake export's topologies, the way `DriverBuilder.build` runs them.

    `strict=False` because `FlowMatchingSpec.func_name` is a `DriverSymbol`: it defers until the built
    driver exists, which is the builder's business and not this module's. What is being exercised here
    is the half answerable from the topologies alone -- and that the deferral is *reported* rather than
    skipped is `test_a_deferred_link_is_reported_rather_than_skipped` below."""
    return check_links(spec, topologies=topologies, strict=False)


def _topology(*input_names, outputs=("v",)):
    """A traced topology as `generate_graph_topology` emits it. The declared output is not decoration:
    the generated Euler loop indexes `run_subgraph`'s single return value, so `FlowMatchingSpec` now
    declares a `TopologyOutputArity` link against it (P4.0.5) -- a two-output estimator would bind `v`
    to the first output's data and integrate the wrong tensor."""
    return {"inputs": [{"name": n} for n in input_names], "outputs": list(outputs)}


MATCHA = FlowMatchingSpec(func_name="sample_decoder", estimator="decoder",
                                 carried_input="z", fixed_inputs=["mu"])
SUPERTONIC = FlowMatchingSpec(func_name="sample_vfe", estimator="vfe", carried_input="z_t",
                                     fixed_inputs=["txt_emb", "stl_emb"])


class TestRenderSampler(unittest.TestCase):
    def test_emits_the_declared_call_with_every_fixed_input(self):
        lua = render_sampler(SUPERTONIC)
        self.assertIn("local function sample_vfe(length, n_elems, n_steps, step_inputs)", lua)
        self.assertIn('loom.run_ode_and_retain("vfe", {n_tokens = length, n_past = 0}, {', lua)
        for line in ("txt_emb = step_inputs.txt_emb,", "stl_emb = step_inputs.stl_emb,"):
            self.assertIn(line, lua)
        # The carried state and the time are the LOOP's, not the caller's: they are named in the opts
        # so the binding can write them per stage, and they are not entries in the argument table.
        self.assertIn('carried = "z_t", time = "t",', lua)
        self.assertNotIn("z_t = z,", lua)

    def test_the_schedule_is_uniform_and_the_default_method_is_euler(self):
        """`t_k = k/n_steps` -- the same points the Lua loop walked, handed over as a schedule.

        Euler by default is load-bearing rather than a preference: a different integrator is a
        different numerical answer, and every flow-matching model in the zoo shipped under this one.
        The engine pins it bit-identically against the loop it replaces (test_lua_bridge_ode.cpp)."""
        lua = render_sampler(MATCHA)
        self.assertIn("for step = 0, n_steps do times[step + 1] = step / n_steps end", lua)
        self.assertIn('method = "euler", times = times, n_elems = n_elems,', lua)
        self.assertNotIn("z[i] = z[i]", lua)

    def test_a_second_order_method_is_opt_in_and_travels_into_the_call(self):
        import dataclasses

        lua = render_sampler(dataclasses.replace(MATCHA, method="midpoint"))
        self.assertIn('method = "midpoint", times = times, n_elems = n_elems,', lua)

    def test_uses_only_double_quotes_so_the_lua_stays_valid(self):
        """The generated code is spliced into a Lua file; a stray Python repr quote would break it."""
        for spec in (MATCHA, SUPERTONIC):
            for line in render_sampler(spec).splitlines():
                if line.lstrip().startswith("--"):
                    continue  # prose comments may legitimately contain apostrophes
                self.assertNotIn("'", line, line)


class TestValidation(unittest.TestCase):
    def test_accepts_a_spec_matching_its_topology(self):
        MATCHA.validate_against_topology(_topology("z", "mu", "t"))
        SUPERTONIC.validate_against_topology(_topology("z_t", "txt_emb", "stl_emb", "t"))

    def test_rejects_an_input_the_topology_never_declared(self):
        with self.assertRaises(ValueError) as cm:
            MATCHA.validate_against_topology(_topology("z", "t"))
        self.assertIn("'mu'", str(cm.exception))
        self.assertIn("sample_decoder", str(cm.exception))

    def test_rejects_leaving_a_declared_input_unsupplied(self):
        with self.assertRaises(ValueError) as cm:
            MATCHA.validate_against_topology(_topology("z", "mu", "t", "cond"))
        self.assertIn("'cond'", str(cm.exception))

    def test_rejects_an_estimator_the_export_never_produced(self):
        with self.assertRaises(ValueError) as cm:
            _check(MATCHA, vocoder=_topology("mel"))
        self.assertIn("decoder", str(cm.exception))


class TestBespokeEstimator(unittest.TestCase):
    """A hand-written sampler generates nothing but is still checked -- the codegen and the validation
    generalize to different extents, which is why EstimatorSpec exists separately.

    StyleTTS2 is the family this was written for, and since P4.0.18 it reaches the same two links
    through `LuaFragment`'s parse of its own `.lua` rather than through a declaration beside it -- see
    `test_driver_components.TestPeeledStyleTTS2`. What is exercised here is the spec itself, which
    is what `FlowMatchingSpec.estimator_spec()` still reduces to."""

    def test_a_bare_estimator_declaration_checks_clean(self):
        spec = EstimatorSpec(topology="diffusion", inputs=["x_in", "time", "embedding"])
        self.assertEqual(_check(spec, diffusion=_topology("x_in", "time", "embedding")), [])

    def test_a_mismatched_bespoke_call_is_still_caught(self):
        spec = EstimatorSpec(topology="diffusion", inputs=["x_in", "time", "attn_mask"])
        with self.assertRaises(ValueError) as cm:
            _check(spec, diffusion=_topology("x_in", "time", "embedding"))
        self.assertIn("attn_mask", str(cm.exception))
        self.assertIn("embedding", str(cm.exception))

    def test_both_spec_kinds_share_one_validation_implementation(self):
        """A refinement spec's own per-step call is an EstimatorSpec, so the two cannot drift apart."""
        self.assertEqual(MATCHA.estimator_spec(),
                         EstimatorSpec(topology="decoder", inputs=["z", "mu", "t"]))


class TestSpecProtocolRetrofit(unittest.TestCase):
    """P4.0.5 stage B.2. Two questions: did the messages survive the move onto `spec_protocol`, and
    does the protocol catch anything the hand-written validator did not."""

    def test_the_mismatch_message_is_preserved_verbatim(self):
        # The whole acceptance criterion of the protocol (`EXPORT-PREPARATION.md` §2): a generic checker
        # that degrades a specific message into "validation failed" is a regression, not a refactor. So
        # this asserts the exact string, not that *a* ValueError was raised.
        with self.assertRaises(ValueError) as cm:
            MATCHA.validate_against_topology(_topology("z", "mu", "t", "cond"))
        self.assertEqual(
            str(cm.exception),
            "FlowMatchingSpec('sample_decoder') does not match topology 'decoder': "
            "leaves declared input(s) unsupplied: ['cond']; "
            "topology declares ['z', 'mu', 't', 'cond'], spec supplies ['z', 'mu', 't'].",
        )

    def test_both_halves_of_the_message_still_appear_together(self):
        with self.assertRaises(ValueError) as cm:
            MATCHA.validate_against_topology(_topology("z", "cond", "t"))
        self.assertEqual(
            str(cm.exception),
            "FlowMatchingSpec('sample_decoder') does not match topology 'decoder': "
            "supplies input(s) it does not declare: ['mu']; "
            "leaves declared input(s) unsupplied: ['cond']; "
            "topology declares ['z', 'cond', 't'], spec supplies ['z', 'mu', 't'].",
        )

    def test_the_unknown_topology_message_is_preserved_verbatim(self):
        with self.assertRaises(ValueError) as cm:
            _check(MATCHA, vocoder=_topology("mel"))
        self.assertEqual(
            str(cm.exception),
            "FlowMatchingSpec('sample_decoder') names topology 'decoder', which is not among the "
            "exported topologies ['vocoder'].",
        )

    def test_a_bespoke_estimator_is_still_named_by_its_topology(self):
        spec = EstimatorSpec(topology="diffusion", inputs=["x_in"])
        with self.assertRaises(ValueError) as cm:
            _check(spec, albert=_topology("tokens"))
        self.assertEqual(
            str(cm.exception),
            "EstimatorSpec('diffusion') names topology 'diffusion', which is not among the exported "
            "topologies ['albert'].",
        )

    def test_a_multi_output_estimator_is_now_rejected(self):
        """The check the hand-written validator never made. `render_sampler` emits
        `local v = loom.run_subgraph(...)` and indexes `v[i]`; against a two-output topology `v` binds
        the first output's DATA and the loop integrates the wrong tensor -- valid Lua, plausible
        shapes, wrong audio, and nothing reports it."""
        with self.assertRaises(ValueError) as cm:
            _check(MATCHA, decoder=_topology("z", "mu", "t", outputs=("v", "logdet")))
        self.assertIn("is built for a topology declaring 1 output(s)", str(cm.exception))
        self.assertIn("'decoder' declares 2: ['v', 'logdet']", str(cm.exception))

    def test_supplied_inputs_is_what_the_generated_lua_actually_passes(self):
        """The link's subject is a derived property, so the declaration cannot drift from the emission:
        every name checked here appears in what `render_sampler` writes.

        The carried state and the time are supplied by the BINDING now rather than by a table entry,
        so they appear as `carried = "z_t"` / `time = "t"` -- still named, still checked against the
        estimator's declared inputs, and no longer values this driver holds."""
        lua = render_sampler(SUPERTONIC)
        self.assertEqual(SUPERTONIC.supplied_inputs, ["z_t", "txt_emb", "stl_emb", "t"])
        for name in SUPERTONIC.fixed_inputs:
            self.assertIn(f"{name} = step_inputs.{name},", lua)
        self.assertIn(f'carried = "{SUPERTONIC.carried_input}"', lua)
        self.assertIn(f'time = "{SUPERTONIC.time_input}"', lua)

    def test_every_field_of_both_specs_is_declared(self):
        """The standing rule, on the first two specs to adopt the protocol: each field is either
        link-checked, covered by another field's link, or documented as uncheckable."""
        from loom_exporter.spec_protocol import dangling_coverage, undeclared_fields

        for cls in (EstimatorSpec, FlowMatchingSpec):
            self.assertEqual(undeclared_fields(cls), [], cls.__name__)
            self.assertEqual(dangling_coverage(cls), [], cls.__name__)

    def test_a_deferred_link_is_reported_rather_than_skipped(self):
        """A spec registered with a checker that never gets the topologies must be REPORTED, not
        silently skipped. What must not happen is a caller believing the spec was validated."""
        from loom_exporter.spec_protocol import LinkChecker, LinkError

        checker = LinkChecker()
        checker.check(MATCHA)
        with self.assertRaises(LinkError) as cm:
            checker.finish()
        self.assertIn("were never checked", str(cm.exception))
        self.assertIn("FlowMatchingSpec('sample_decoder').estimator", str(cm.exception))


if __name__ == "__main__":
    unittest.main()


F5 = FlowMatchingSpec(func_name="sample_estimator", estimator="estimator", carried_input="x",
                      fixed_inputs=["cond", "text_embed"], schedule="caller", guidance=True)
# The shape F5-TTS actually ships: all three declarations at once.
F5_NOISE = FlowMatchingSpec(func_name="sample_estimator", estimator="estimator", carried_input="x",
                            fixed_inputs=["cond", "text_embed"], schedule="caller", guidance=True,
                            caller_noise=True)


class TestCallerSchedule(unittest.TestCase):
    """Family 9's third leaf: F5-TTS integrates a schedule the DRIVER builds.

    `t + coef*(cos(pi/2 * t) - 1 + t)` over a linspace -- "sway sampling", which spends more steps
    near t=0 -- and for low step counts an empirically pruned table instead. Neither is a property of
    the estimator, so the loop is unchanged and only the source of `times` moves.
    """

    def test_the_caller_supplies_times_instead_of_a_step_count(self):
        lua = render_sampler(F5)
        self.assertIn("local function sample_estimator(length, n_elems, times, step_inputs, "
                      "uncond_inputs, cfg_scale)", lua)
        # The uniform branch's own two lines must be GONE, not merely unreached: an emitted
        # `for step = 0, n_steps` would shadow the caller's array with a linspace.
        self.assertNotIn("times[step + 1] = step / n_steps", lua)
        self.assertIn("times = times,", lua)

    def test_the_uniform_default_is_unchanged(self):
        """Every model that already shipped declares no schedule, and its Lua must not move."""
        lua = render_sampler(MATCHA)
        self.assertIn("local function sample_decoder(length, n_elems, n_steps, step_inputs)", lua)
        self.assertIn("for step = 0, n_steps do times[step + 1] = step / n_steps end", lua)
        self.assertNotIn("uncond_inputs", lua)
        # The CALL, not the generated header comment, which now records `guidance=false`.
        self.assertNotIn("guidance = {", lua)

    def test_an_unknown_schedule_is_refused_rather_than_defaulted(self):
        spec = FlowMatchingSpec(func_name="s", estimator="e", carried_input="x", schedule="swayy")
        with self.assertRaises(ValueError) as ctx:
            render_sampler(spec)
        self.assertIn("'swayy'", str(ctx.exception))


class TestGuidance(unittest.TestCase):
    def test_the_unconditional_table_carries_the_same_input_names(self):
        """Guidance drops a conditioning signal's VALUE, not its existence.

        That is why `guidance` is a bool rather than a list of inputs: `supplied_inputs` already
        describes both tables, so the existing `TopologyInput` link checks the unconditional call by
        checking the conditional one.
        """
        lua = render_sampler(F5)
        self.assertIn("scale = cfg_scale,", lua)
        for name in ("cond", "text_embed"):
            self.assertIn(f"{name} = step_inputs.{name},", lua)
            self.assertIn(f"{name} = uncond_inputs.{name},", lua)

    def test_the_spec_still_checks_against_the_real_topology(self):
        # `_check` returns the links it DEFERRED (func_name is a DriverSymbol), so a clean run is
        # "it did not raise" rather than an empty list.
        _check(F5, estimator=_topology("x", "cond", "text_embed", "t"))
        with self.assertRaises(Exception):
            _check(F5, estimator=_topology("x", "cond", "t"))


class TestCallerNoise(unittest.TestCase):
    """The initial state as a driver input, which is what makes a flow-matching export gradeable.

    Flow matching starts from a Gaussian draw. torch's RNG and the engine's are different algorithms,
    so handing both sides the same SEED hands them different NOISE, and a different draw is a different
    valid sample -- F5-TTS's gate measured max |d| 1.25 on audio that was intelligible and correctly
    voiced. Absent a value the engine still draws, so `infer(text)` keeps working.
    """

    def test_the_state_is_the_last_argument_and_reaches_the_opts(self):
        spec = FlowMatchingSpec(func_name="s", estimator="estimator", carried_input="x",
                                fixed_inputs=["cond"], caller_noise=True)
        lua = render_sampler(spec)
        self.assertIn("local function s(length, n_elems, n_steps, step_inputs, state)", lua)
        self.assertIn("state = state,", lua)

    def test_it_composes_with_guidance_and_a_caller_schedule(self):
        lua = render_sampler(F5_NOISE)
        self.assertIn("local function sample_estimator(length, n_elems, times, step_inputs, "
                      "uncond_inputs, cfg_scale, state)", lua)
        self.assertIn("state = state,", lua)
        self.assertIn("scale = cfg_scale,", lua)

    def test_a_spec_without_it_emits_no_state_at_all(self):
        """Every shipped flow-matching model declares none, and its driver text must not move."""
        lua = render_sampler(MATCHA)
        self.assertNotIn("state", lua)


class TestSamplerComponentArguments(unittest.TestCase):
    """`FlowMatchingSampler` refuses a declaration that does not match its spec, in BOTH directions.

    The failure this prevents is silent: a component that supplies `times` to a uniform spec has it
    ignored and integrates a linspace, and one that omits it from a caller-scheduled spec passes `nil`
    where the engine wants N+1 points -- which fails inside the binding, two layers from the line that
    got it wrong.
    """

    def _sampler(self, spec, **kwargs):
        from loom_exporter.driver_components import FlowMatchingSampler
        from loom_exporter.driver_ir import Lit, Var

        defaults = dict(spec=spec, result="_z", length=Var("n"), n_elems=Var("m"),
                        n_steps=Lit(10), step_inputs={})
        defaults.update(kwargs)
        return FlowMatchingSampler(**defaults)

    def _emit(self, component):
        return component.emit(None)

    def test_a_caller_scheduled_spec_without_times_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            self._emit(self._sampler(F5, uncond_inputs={}, guidance_scale=None))
        self.assertIn("times", str(ctx.exception))

    def test_times_against_a_uniform_spec_is_refused(self):
        from loom_exporter.driver_ir import Var

        with self.assertRaises(ValueError) as ctx:
            self._emit(self._sampler(MATCHA, times=Var("times")))
        self.assertIn("builds its own", str(ctx.exception))

    def test_guidance_needs_both_halves(self):
        from loom_exporter.driver_ir import Var

        with self.assertRaises(ValueError) as ctx:
            self._emit(self._sampler(F5, times=Var("times"), uncond_inputs={"cond": Var("z")}))
        self.assertIn("guidance_scale", str(ctx.exception))

    def test_guidance_arguments_against_an_unguided_spec_are_refused(self):
        from loom_exporter.driver_ir import Lit, Var

        with self.assertRaises(ValueError) as ctx:
            self._emit(self._sampler(MATCHA, uncond_inputs={"mu": Var("z")}, guidance_scale=Lit(2)))
        self.assertIn("does not declare guidance", str(ctx.exception))

    def test_a_caller_noise_spec_without_state_is_refused(self):
        """Passing nil would make the engine draw, which is the behaviour the flag exists to replace --
        and it would be SILENT: a gate comparing against a reference draw would simply go red with no
        indication that the declaration, not the model, was the problem."""
        from loom_exporter.driver_ir import Var

        with self.assertRaises(ValueError) as ctx:
            self._emit(self._sampler(F5_NOISE, times=Var("times"),
                                      uncond_inputs={"cond": Var("z"), "text_embed": Var("te")},
                                      guidance_scale=Var("cfg")))
        self.assertIn("caller_noise", str(ctx.exception))

    def test_state_against_a_drawing_spec_is_refused(self):
        from loom_exporter.driver_ir import Var

        with self.assertRaises(ValueError) as ctx:
            self._emit(self._sampler(MATCHA, state=Var("noise")))
        self.assertIn("does not declare caller_noise", str(ctx.exception))
