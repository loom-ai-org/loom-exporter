"""`LoomExportConfig` -- the root of every family's export-config hierarchy (EXPORT-ROADMAP.md R3,
BACKLOG.md P3.1/P4.0.3).

Mirrors `optimum-onnx`'s `OnnxConfig`, but deliberately shallow: it owns the three fields every family
needs regardless of its own mechanics (`architecture`/`output_path`/`decomposition`) and a single
`export()` contract. Everything else -- how a checkpoint is loaded, what its wrapper looks like, how
many phases it traces -- lives on the family-specific subclasses (`causal_lm_export.
LMCausalModelExportConfig`, `nemo_asr_export.ASRNemoEncoderExportConfig`,
`multi_phase_export.BaseMultiPhaseModelExportConfig`, ...). The `{Domain}{Function}ExportConfig` naming
convention every subclass follows is described in BACKLOG.md's P3.1 entry; `LoomExportConfig` itself is
the one name in that hierarchy with no domain prefix, since it sits above every domain.

`export()` is not overridden by any family any more: it delegates to `self.decomposition`, which owns
the trace-and-assemble mechanics, while the config owns the family knowledge the decomposition asks it
for. See `decomposition.py` for why that split, and for which families genuinely have a choice of
decomposition (only causal-LM) versus a structural one (everyone else).
"""
from dataclasses import dataclass

from .decomposition import Decomposition
from .spec_protocol import NestedSpec, Unchecked


@dataclass(kw_only=True)
class LoomExportConfig:
    """Base for every family template's top-level config object -- the thing a registry entry
    constructs and calls `.export()` on."""

    # GGUF `general.architecture` value.
    architecture: str
    # Output .gguf path.
    output_path: str
    # How this model's graph(s) get built: one flattened trace, a ModularExportSpec assembly, or N
    # merged phases. A family with only one possible answer defaults it; only the causal-LM family
    # currently accepts either (LFM2 exports both ways -- a caller decision, not a checkpoint property).
    decomposition: Decomposition

    # The three fields every family inherits, declared once here rather than restated by each of the
    # five (P4.0.5's standing rule; `spec_protocol.declared_raw` merges these along the MRO, which is
    # what makes declaring them once possible at all).
    __links__ = {
        "decomposition": NestedSpec(
            where="Decomposition.export(), which checks the decomposition's own specs -- Modular "
                  "carries a ModularExportSpec whose ModuleAttrPath links are checked against the "
                  "loaded model there, MultiPhase checks each ExportPhase's axes"
        ),
    }
    __unchecked__ = {
        "architecture": Unchecked(
            "the GGUF `general.architecture` string. Free-form by design -- it names the family to a "
            "GGUF reader, and the causal-LM family infers it from the checkpoint's own model_type, so "
            "there is no independent authority to check it against."
        ),
        "output_path": Unchecked(
            "where to write. Nothing real to check it against: the file does not exist yet, and "
            "whether its directory is writable is the filesystem's error to raise, not a spec claim."
        ),
        "task": Unchecked(
            "which task this export was produced under, written into the GGUF as `loom.task` so a host "
            "can dispatch on what the file says rather than on which architecture it recognises. Set by "
            "`main_export` from the recognizer that matched, never by a family -- and unchecked here "
            "because the check that matters is structural rather than a spec claim: "
            "`TaskRegistry.register()` stamps it from the entry, so a recognizer's task IS the task it "
            "was registered under and the two cannot disagree. Empty when a config is built directly "
            "rather than through `main_export`, which `contract()` reads as 'declare nothing'."
        ),
    }

    def export(self) -> str:
        """Runs the whole export -- load, trace, compile, write GGUF -- and returns `output_path`."""
        return self.decomposition.export(self)

    # -- hooks the decompositions read; see decomposition.py for which one needs which ------------------

    def prepare_environment(self) -> None:
        """Import-order workarounds a family needs before its real third-party package is importable at
        all (`patcher.ModelPatcher`). A no-op for families that need none, so every decomposition can
        call it unconditionally."""

    def export_architecture(self) -> str:
        """The GGUF `general.architecture` value, after any per-family resolution. Overridden by the
        causal-LM family, which infers it from the checkpoint's own `model.config.model_type` when the
        caller did not name one."""
        return self.architecture

    def backend_kwargs(self) -> dict:
        """Extra keyword arguments for `LoomGGUFBackend.__call__` beyond `output_path`/`architecture`
        (tokenizer paths, quantization, `root_axis`, `flat_namespace`). Empty by default.

        **An override must carry `hparams=self.hparams()` through.** Every override here builds its own
        dict rather than updating `super()`'s, which is fine for kwargs a family opts into and wrong
        for one every family has -- so the four that exist each pass it explicitly, and
        `test_export_hparams.py` walks the registry to check they still do. A hook honoured by one path
        out of four is worse than no hook: it reads as available and silently does nothing."""
        return {"hparams": self.hparams()}

    # Which task this export was produced under. Set on the INSTANCE by `main_export`, from the
    # recognizer that matched, because that is the last point where the task is known. A class attribute
    # rather than a dataclass field on purpose: every family's config subclasses this one and several
    # declare required fields, so adding a defaulted field here would constrain their field order for
    # nothing.
    task: str = ""

    def resolved_backend_kwargs(self) -> dict:
        """`backend_kwargs()` plus what EVERY export must carry, merged here so a family cannot drop it.

        This is the structural version of the warning above. `hparams` is universal and is nonetheless
        passed by hand in four overrides, held in place only by a test that walks the registry checking
        they still do -- which catches a family that forgets one commit after it ships. The contract is
        universal in the same way and is deliberately NOT routed that way: it is merged over whatever
        the family returns, so forgetting it is not expressible.

        `setdefault` for `hparams` rather than assignment, so an override that does pass it still wins.
        """
        kwargs = dict(self.backend_kwargs())
        kwargs.setdefault("hparams", self.hparams())
        kwargs["contract"] = self.contract()
        return kwargs

    def driver_input_aliases(self) -> dict:
        """`{the name this family's driver body reads: the canonical name a host passes}`.

        A host addresses an input by what it IS -- `tokens` for text or ids, `waveform` for audio,
        `n_steps` for a sampler's step count -- never by which model it is. `caller_input()` makes that
        true for a synthesized driver at every read site; for a driver adopted from hand-written Lua the
        builder normalises the inputs table once at the top of `infer`, and this is what it needs.

        Empty means the body already reads canonical names, which is every synthesized driver.
        """
        return {}

    def contract(self) -> dict:
        """What this export declares about ITSELF -- the task, and the modality pair it maps between.

        Written into the GGUF as `loom.task` / `loom.input.kind` / `loom.output.kind` and read back by
        `loom::ModelContract`. It exists because a host had no way to know what a model was for: the file
        said `loom.architecture`, a per-MODEL name, so any end-to-end door a host offered had to be
        reached through a table of architecture names -- the per-architecture host code all three repos
        forbid. See loom.cpp `docs/HIGH-LEVEL-API.md`.

        The defaults below are per-TASK and cover every family registered today. A family overrides this
        only where its own contract differs from its task's usual one -- Supertonic taking graphemes
        where the other four TTS families take phoneme ids -- or to add a task-specific table, which is
        what Whisper does with the ASR decode ids.

        Empty when `task` is unset, which is what happens when a config is constructed directly rather
        than through `main_export`. An export that cannot say which task it is declares NOTHING rather
        than guessing: a wrong `loom.task` is worse for a host than an absent one, because absence is
        something a host can detect and a wrong name is not.
        """
        if not self.task:
            return {}
        pair = {
            "text-generation": ("text", "text"),
            "automatic-speech-recognition": ("audio", "token_ids"),
            # Four of the five TTS families take phoneme ids a G2P step produces outside the engine;
            # Supertonic encodes graphemes itself and overrides this. The distinction is per-model and is
            # exactly what a host needs in order to know whether it can offer a text door at all.
            "text-to-speech": ("phoneme_ids", "audio"),
        }.get(self.task)
        if pair is None:
            return {"task": self.task}
        return {"task": self.task, "input.kind": pair[0], "output.kind": pair[1]}

    def hparams(self) -> dict:
        """`{key: number}` a **host** needs in order to call this model's driver at all -- written into
        the GGUF as `loom.<key>` KVs, the same namespace `loom::make_kv_cache` already reads its five
        geometry facts from.

        This is one half of a split, and which half a number belongs in is decided by who reads it:

        * a number the **driver** needs is an `ExportConstants` value (`driver_components.py`), bound as
          an IR local, because Lua cannot read GGUF metadata at all;
        * a number the **host** needs -- to size an input it must build, or to interpret an output --
          belongs here, because the driver cannot hand it over before being called.

        Kokoro's `style_dim` is the clearest case of the second: a caller cannot construct `ref_s`
        without knowing how long each of its two halves is, and until this existed the answer lived in
        `tests/tts_driver_inputs.h` -- a C++ test header, which is exactly the "self-contained GGUF"
        claim being false (P4.0.8's first follow-up; KV-CACHE.md 1.1/1.3 made the same argument for
        cache geometry).

        A number both sides need is declared once *here in the config* and rendered twice, which is not
        the "two spellings that can disagree" `kv_cache.h` warns about: both readings come from one
        attribute set in `phases()`, so there is no second authority. Supertonic's fixed text length is
        that case -- the host needs it to build `txt_ids`, the driver needs it to reject a wrong-length
        one.

        Empty by default; `int` values are written as u32 and `float` as f32."""
        return {}
