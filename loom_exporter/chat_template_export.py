"""Reduces a checkpoint's Jinja chat template to a small set of GGUF string KVs the engine can assemble
without a Jinja evaluator (P4.23, ADR-018).

**Why this is here and not in the engine.** The template is per-MODEL, and per-model complexity belongs
in the exporter (ADR-003). The three options were: carry the Jinja source as a KV and let the host
render it -- which leaves `loom_cli` with no renderer at all and makes `text2text.chat` a Python-only
door; ship a Jinja subset in the engine -- which is per-model machinery in the one place this project
keeps removing it from; or reduce the template HERE to the role tags it actually emits. This module is
the third.

**What it produces.** A chat template, for every checkpoint in this set, is:

    prologue + SUM over messages of (prefix[role] + content + suffix[role]) + generation_prefix

with a second prologue for the case where the conversation opens with a system message, because a
template may inject a DEFAULT system turn when it does not (SmolLM2 injects "You are a helpful AI
assistant named SmolLM, trained by Hugging Face"). That is five string KVs and two parallel arrays.

**How the parts are recovered: by differencing real renders, never by reading the Jinja.** Rendering
`[user]`, then `[user, assistant]`, then `[user, assistant, user]` and taking each render's tail over
the previous one isolates one message's block at a time; a sentinel content splits that block into its
prefix and suffix. Nothing here parses a template, so a template written in any style is handled the
same way -- what decides whether a checkpoint is supported is whether its own output DECOMPOSES.

**And every decomposition is verified against `apply_chat_template` before it is written**, on a
four-message conversation, at the STRING level and again at the ID level. A role that fails is dropped
from `roles` and the engine then raises on it by name; if `user` or `assistant` fails, nothing is
written at all and the model simply has no chat door. Gemma 3 is the live example of a partial: its
template folds a system message into the first user turn rather than emitting a system block, so it
exports with `roles = [user, assistant]` and asking it for a system prompt is an error rather than a
silently dropped argument.
"""
import json
from pathlib import Path
from typing import Optional

# Unlikely in real content, and short enough not to perturb a template that measures its input. Two
# distinct ones so a render that emits a message twice cannot look like a clean split.
_SENTINEL = "\x00LOOM_CONTENT\x00"
_SENTINEL2 = "\x00LOOM_SECOND\x00"

# The roles worth probing. `tool` is deliberately absent: no checkpoint in this set is exported with
# tool calling, and a role whose block cannot be verified against a real render must not be guessed.
_ROLES = ("system", "user", "assistant")


class ChatTemplateParts:
    """The decomposition, or the reason there isn't one.

    `roles`/`prefixes`/`suffixes` are parallel, in `_ROLES` order restricted to what verified.
    `unsupported` names each role that was probed and rejected, with why -- printed by the exporter so
    a partial export says out loud what it dropped.
    """

    def __init__(self):
        self.roles: list[str] = []
        self.prefixes: list[str] = []
        self.suffixes: list[str] = []
        self.prologue: str = ""
        self.system_prologue: str = ""
        self.generation_prefix: str = ""
        self.trim_content: bool = False
        self.unsupported: dict[str, str] = {}
        # What `strip_duplicate_bos` removed from the front, or "". Kept because the text the MODEL
        # sees still begins with it -- the vocabulary re-adds it at encode time -- so this is what
        # `verify_chat_template` must put back before comparing against `apply_chat_template`.
        self.stripped_bos: str = ""

    def __bool__(self) -> bool:
        # A template without both halves of a turn is not a chat template: nothing can be asked and
        # nothing can be answered.
        return "user" in self.roles and "assistant" in self.roles

    def block(self, role: str, content: str) -> str:
        i = self.roles.index(role)
        return self.prefixes[i] + content + self.suffixes[i]

    def render(self, messages: list[tuple[str, str]], add_generation_prompt: bool) -> str:
        """The engine's own assembly, in Python -- the reference the verification below compares
        `apply_chat_template` against. Kept identical in shape to `ChatTemplate::apply` in
        `src/core/chat_template.cpp`, because a divergence between them is a divergence between what
        was verified and what ships."""
        out = self.system_prologue if messages and messages[0][0] == "system" else self.prologue
        for role, content in messages:
            out += self.block(role, content.strip() if self.trim_content else content)
        return out + (self.generation_prefix if add_generation_prompt else "")


def _render(tokenizer, messages, add_generation_prompt=False) -> Optional[str]:
    """One `apply_chat_template` call, or None if the template refused it.

    A refusal is ordinary rather than exceptional: Gemma 3's template calls `raise_exception` when the
    roles do not alternate, which is exactly what probing an assistant-first conversation does.
    """
    try:
        return tokenizer.apply_chat_template(
            [{"role": r, "content": c} for r, c in messages],
            tokenize=False, add_generation_prompt=add_generation_prompt,
        )
    except Exception:
        return None


def _split_on_sentinel(block: str, sentinel: str = _SENTINEL) -> Optional[tuple[str, str]]:
    """A message's rendered block into (prefix, suffix). None if the sentinel is absent or repeated --
    either way the block is not `prefix + content + suffix` and must not be treated as if it were."""
    parts = block.split(sentinel)
    return (parts[0], parts[1]) if len(parts) == 2 else None


def derive_chat_template(tokenizer) -> ChatTemplateParts:
    """`tokenizer`'s chat template as assembleable parts, verified. Empty (falsy) when the checkpoint
    has no template, or has one that does not decompose."""
    parts = ChatTemplateParts()
    if not getattr(tokenizer, "chat_template", None):
        return parts

    # The three renders every role but `system` is recovered from. Each adds one message to the last,
    # so each one's TAIL over its predecessor is exactly one message's block -- which is what makes
    # this differencing rather than parsing.
    r_user = _render(tokenizer, [("user", _SENTINEL)])
    r_user_asst = _render(tokenizer, [("user", _SENTINEL2), ("assistant", _SENTINEL)])
    r_gen = _render(tokenizer, [("user", _SENTINEL)], add_generation_prompt=True)
    if r_user is None or r_user_asst is None or r_gen is None:
        parts.unsupported["user"] = "apply_chat_template refused a plain user/assistant conversation"
        return parts

    # `| trim` is real and worth detecting rather than ignoring: Gemma 3's template trims every
    # message, so an engine that did not would disagree with `transformers` by whitespace on any
    # content with a leading newline -- which is exactly what a multi-line prompt has.
    r_padded = _render(tokenizer, [("user", "  " + _SENTINEL + "  ")])
    parts.trim_content = r_padded is not None and ("  " + _SENTINEL + "  ") not in r_padded

    # The assistant block is the second render's tail over the first, once the first's sentinel is
    # rewritten to what the second used for the same message.
    head = r_user.replace(_SENTINEL, _SENTINEL2)
    if not r_user_asst.startswith(head):
        parts.unsupported["assistant"] = (
            "adding an assistant turn rewrote the user turn before it, so a message's rendering is not "
            "independent of what follows it"
        )
        return parts
    asst = _split_on_sentinel(r_user_asst[len(head):])
    if asst is None:
        parts.unsupported["assistant"] = "the rendered block is not prefix+content+suffix"
        return parts

    # The FIRST user block still carries the conversation prologue in front of it -- `<bos>`, or an
    # injected default system turn. Recover the block without it from a user turn that is not first:
    # the third message of a three-message render, whose tail over the two-message render is exactly
    # that block and nothing else.
    r_uu = _render(tokenizer, [("user", _SENTINEL2), ("assistant", _SENTINEL2)])
    r_three = _render(tokenizer, [("user", _SENTINEL2), ("assistant", _SENTINEL2), ("user", _SENTINEL)])
    if r_uu is None or r_three is None or not r_three.startswith(r_uu):
        parts.unsupported["user"] = (
            "a three-message conversation is not the two-message one with a block appended, so a turn's "
            "rendering depends on its position beyond the first"
        )
        return parts
    later_user = _split_on_sentinel(r_three[len(r_uu):])
    if later_user is None:
        parts.unsupported["user"] = "the rendered block is not prefix+content+suffix"
        return parts
    first_block = later_user[0] + _SENTINEL + later_user[1]
    if not r_user.endswith(first_block):
        parts.unsupported["user"] = "the first user turn is not a later one with a prologue in front"
        return parts
    parts.prologue = r_user[: len(r_user) - len(first_block)]

    parts.roles = ["user", "assistant"]
    parts.prefixes = [later_user[0], asst[0]]
    parts.suffixes = [later_user[1], asst[1]]

    if not r_gen.startswith(r_user):
        parts.unsupported["generation_prompt"] = "the generation prompt is not appended to the conversation"
        return ChatTemplateParts()
    parts.generation_prefix = r_gen[len(r_user):]

    # System is probed separately because a template may not have one at all as a distinct block:
    # Gemma 3 folds a system message into the text of the first user turn, which is a different
    # structure rather than a different string, and no amount of differencing recovers a block that
    # was never emitted.
    r_sys = _render(tokenizer, [("system", _SENTINEL), ("user", _SENTINEL2)])
    tail = parts.block("user", _SENTINEL2)
    if r_sys is None:
        parts.unsupported["system"] = "apply_chat_template refused a system message"
    elif not r_sys.endswith(tail):
        parts.unsupported["system"] = (
            "a system message changes how the user turn after it renders (Gemma 3 folds the system "
            "text into the first user turn rather than emitting a block of its own)"
        )
    else:
        sys_block = _split_on_sentinel(r_sys[: len(r_sys) - len(tail)])
        if sys_block is None:
            parts.unsupported["system"] = "the rendered block is not prefix+content+suffix"
        else:
            # Where the system PROLOGUE ends and the system PREFIX begins is undecidable from one
            # render -- a system message is always the first message, so nothing ever renders one with
            # no prologue in front of it the way the second user turn does. The whole head therefore
            # goes in `system_prologue`, leaving the prefix empty: the concatenation is identical
            # either way, and this is the split that puts a leading BOS somewhere
            # `strip_duplicate_bos` can find it (LFM2's system head is `<|startoftext|><|im_start|>
            # system\n`, and only the first of those is the vocabulary's own).
            parts.system_prologue = sys_block[0]
            parts.roles.append("system")
            parts.prefixes.append("")
            parts.suffixes.append(sys_block[1])

    return parts


def verify_chat_template(tokenizer, parts: ChatTemplateParts) -> list[str]:
    """Re-derives real conversations both ways and returns the mismatches, empty when there are none.

    Two levels, because they fail differently. The STRING check catches a decomposition that dropped
    structure. The ID check catches a decomposition that is textually right and tokenizes differently
    anyway -- which is what a doubled BOS is, and it is invisible in the text.
    """
    problems = []
    conversations = [
        ([("user", "Who discovered Brazil?")], True),
        ([("user", "Who discovered Brazil?"), ("assistant", "The Portuguese."),
          ("user", "In what year?")], True),
        ([("user", "Hello"), ("assistant", "Hi.")], False),
    ]
    if "system" in parts.roles:
        conversations.append(([("system", "You are terse."), ("user", "Hello")], True))

    for messages, add_gen in conversations:
        want = _render(tokenizer, messages, add_generation_prompt=add_gen)
        got = parts.render(messages, add_gen)
        if want is None:
            problems.append(f"apply_chat_template refused {messages!r}, which the parts do render")
            continue
        if want != parts.stripped_bos + got:
            problems.append(f"{messages!r} (generation_prompt={add_gen}): rendered "
                             f"{parts.stripped_bos + got!r}, want {want!r}")
            continue
        # The ids are the real contract: the engine encodes the assembled string with the vocabulary's
        # own `add_bos_token` applied, so this is `add_special_tokens=True` against the template's own
        # `tokenize=True` output.
        want_ids = tokenizer.apply_chat_template(
            [{"role": r, "content": c} for r, c in messages], tokenize=True,
            add_generation_prompt=add_gen)
        got_ids = tokenizer(got, add_special_tokens=True)["input_ids"]
        if list(want_ids) != list(got_ids):
            problems.append(f"{messages!r}: ids {got_ids} against apply_chat_template's {want_ids}")
    return problems


def strip_duplicate_bos(parts: ChatTemplateParts, tokenizer_dir) -> None:
    """Removes a leading BOS from either prologue when the VOCABULARY will prepend one anyway.

    `BpeVocab::encode` prepends `bos_token_id` whenever `tokenizer.ggml.add_bos_token` is set, and both
    Gemma 3 and LFM2 set it AND open their template with the same token. Left in, the assembled prompt
    starts with two of them -- textually IDENTICAL to what `apply_chat_template` produces, and two ids
    where the checkpoint has one. That is why `verify_chat_template`'s id check exists beside its string
    check, and why this runs BEFORE it: measured on Gemma 3, the string comparison passes and the ids
    are `[2, 2, 105, ...]` against `[2, 105, ...]`.

    **`tokenizer_config.json`, not the tokenizer OBJECT**, and that is not fussiness: LFM2's
    `add_bos_token` is `true` in its config while the loaded `PreTrainedTokenizerFast` has no such
    attribute at all, so `getattr(tokenizer, "add_bos_token", False)` reads False and the BOS doubles.
    The config is also the right authority on principle -- it is what `write_bpe_vocab` reads to decide
    `tokenizer.ggml.add_bos_token`, which is the KV the engine actually acts on.
    """
    config_path = Path(tokenizer_dir) / "tokenizer_config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    bos = config.get("bos_token")
    bos_piece = bos["content"] if isinstance(bos, dict) else bos
    if not config.get("add_bos_token", False) or not bos_piece:
        return
    for field in ("prologue", "system_prologue"):
        value = getattr(parts, field)
        if value.startswith(bos_piece):
            setattr(parts, field, value[len(bos_piece):])
            parts.stripped_bos = bos_piece


def write_chat_template(writer, parts: ChatTemplateParts) -> None:
    """The KVs `loom::ChatTemplate::load` reads back. Writes nothing for an empty decomposition, which
    is what leaves a model with no chat door rather than a broken one."""
    if not parts:
        return
    writer.add_array("tokenizer.chat_template.roles", parts.roles)
    writer.add_array("tokenizer.chat_template.prefixes", parts.prefixes)
    writer.add_array("tokenizer.chat_template.suffixes", parts.suffixes)
    writer.add_string("tokenizer.chat_template.prologue", parts.prologue)
    writer.add_string("tokenizer.chat_template.system_prologue", parts.system_prologue)
    writer.add_string("tokenizer.chat_template.generation_prefix", parts.generation_prefix)
    writer.add_bool("tokenizer.chat_template.trim_content", parts.trim_content)
