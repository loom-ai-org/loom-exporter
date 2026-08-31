"""P4.23: reducing a checkpoint's Jinja chat template to role tags, and the templates that must be
REFUSED rather than approximated (ADR-018).

Hermetic: the tokenizer here is a stub whose `apply_chat_template` is a Python function, so a template
shape can be written in three lines instead of downloading a checkpoint that happens to have it. The
shapes below are the real ones -- ChatML, Gemma 3's system-folding, SmolLM2's injected default system
turn, Gemma's `| trim`, Qwen3-Base's position-dependent rewriting -- each reduced to the smallest
function that reproduces the property that matters.

What is NOT tested here is that the reduction matches a real checkpoint; that is the exporter's own
`verify_chat_template`, which runs on every export against `apply_chat_template` itself, and the
engine's `tests/gate/test_e2e_chat_generation.cpp`.
"""
import json
import tempfile
import unittest
from pathlib import Path

from loom_exporter.chat_template_export import (derive_chat_template, strip_duplicate_bos,
                                                 verify_chat_template)


class _FakeTokenizer:
    """The two attributes `derive_chat_template` and `verify_chat_template` touch.

    `render` is `(messages, add_generation_prompt) -> str`, or raises the way a real template's
    `raise_exception` does. `encode` stands in for the tokenizer's `__call__` in the id check: one id
    per character is enough, because what the id check is FOR is catching a doubled prefix, and a
    doubled prefix is visible at any granularity.
    """

    def __init__(self, render, add_bos_piece=None):
        self.chat_template = "<not parsed by anything>"
        self._render = render
        self._add_bos_piece = add_bos_piece

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = self._render([(m["role"], m["content"]) for m in messages], add_generation_prompt)
        # `tokenize=True` adds NO specials of its own -- a template that wants a BOS emits one, which
        # is exactly why the doubled-BOS case exists: `__call__` below then adds a second.
        return self._ids("", text) if tokenize else text

    def __call__(self, text, add_special_tokens=True):
        prefix = self._add_bos_piece if (add_special_tokens and self._add_bos_piece) else ""
        return {"input_ids": self._ids(prefix, text)}

    @staticmethod
    def _ids(prefix, text):
        return [ord(c) for c in prefix + text]


def _chatml(messages, add_generation_prompt):
    out = "".join(f"<|im_start|>{r}\n{c}<|im_end|>\n" for r, c in messages)
    return out + ("<|im_start|>assistant\n" if add_generation_prompt else "")


def _write_tokenizer_config(directory, **config):
    Path(directory, "tokenizer_config.json").write_text(json.dumps(config))
    return directory


class TestChatMLDecomposes(unittest.TestCase):
    def setUp(self):
        self.tok = _FakeTokenizer(_chatml)
        self.parts = derive_chat_template(self.tok)

    def test_all_three_roles_are_recovered(self):
        self.assertTrue(self.parts)
        self.assertEqual(sorted(self.parts.roles), ["assistant", "system", "user"])
        self.assertEqual(self.parts.unsupported, {})

    def test_the_parts_reproduce_apply_chat_template(self):
        self.assertEqual(verify_chat_template(self.tok, self.parts), [])

    def test_the_generation_prefix_is_the_only_difference_between_two_renders(self):
        conversation = [("user", "a"), ("assistant", "b")]
        self.assertEqual(self.parts.render(conversation, True),
                          self.parts.render(conversation, False) + "<|im_start|>assistant\n")

    def test_a_template_with_no_prologue_writes_an_empty_one(self):
        """ChatML opens straight onto the first turn. An empty prologue is a real value, not a
        missing one -- the engine reads it with a default of "" for exactly this."""
        self.assertEqual(self.parts.prologue, "")

    def test_the_system_head_lands_in_the_prologue_and_the_prefix_is_empty(self):
        """Where a system PROLOGUE ends and a system PREFIX begins is undecidable from one render --
        a system message is always first, so nothing ever renders one without a prologue in front the
        way the second user turn does. The whole head therefore goes in the prologue, which is also
        the split that puts a leading BOS somewhere `strip_duplicate_bos` can find it."""
        i = self.parts.roles.index("system")
        self.assertEqual(self.parts.system_prologue, "<|im_start|>system\n")
        self.assertEqual(self.parts.prefixes[i], "")
        self.assertEqual(self.parts.block("system", "hi") , "hi<|im_end|>\n")


class TestAnInjectedDefaultSystemTurnIsAPrologue(unittest.TestCase):
    """SmolLM2's shape: the template inserts its own system turn when the conversation does not open
    with one. Which prologue applies is therefore a property of the CONVERSATION, and a scheme with
    only one of them would emit the default system prompt AND the caller's."""

    def setUp(self):
        def render(messages, add_generation_prompt):
            head = "" if messages and messages[0][0] == "system" else "<|im_start|>system\nBe nice<|im_end|>\n"
            return head + _chatml(messages, add_generation_prompt)
        self.tok = _FakeTokenizer(render)
        self.parts = derive_chat_template(self.tok)

    def test_the_default_turn_lands_in_the_plain_prologue(self):
        self.assertEqual(self.parts.prologue, "<|im_start|>system\nBe nice<|im_end|>\n")

    def test_a_caller_supplied_system_turn_replaces_it(self):
        self.assertEqual(self.parts.render([("system", "Be terse"), ("user", "hi")], False),
                          "<|im_start|>system\nBe terse<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n")

    def test_it_verifies_both_ways(self):
        self.assertEqual(verify_chat_template(self.tok, self.parts), [])


class TestGemmaShapedTemplates(unittest.TestCase):
    """Gemma 3's two properties, each of which a naive reduction gets wrong."""

    @staticmethod
    def _render(messages, add_generation_prompt):
        if messages and messages[0][0] == "system":
            # The property that matters: a system message is FOLDED into the first user turn rather
            # than emitted as a block of its own.
            folded = [("user", messages[0][1] + "\n\n" + messages[1][1])] + list(messages[2:])
            messages = folded
        for role, _ in messages:
            if role not in ("user", "assistant"):
                raise ValueError("Conversation roles must alternate user/assistant/...")
        out = "<bos>"
        for role, content in messages:
            tag = "model" if role == "assistant" else role
            out += f"<start_of_turn>{tag}\n{content.strip()}<end_of_turn>\n"
        return out + ("<start_of_turn>model\n" if add_generation_prompt else "")

    def setUp(self):
        self.tok = _FakeTokenizer(self._render, add_bos_piece="<bos>")
        self.parts = derive_chat_template(self.tok)

    def test_user_and_assistant_survive_and_system_is_declined_by_name(self):
        self.assertTrue(self.parts)
        self.assertEqual(self.parts.roles, ["user", "assistant"])
        self.assertIn("system", self.parts.unsupported)
        self.assertIn("folds", self.parts.unsupported["system"])

    def test_the_assistant_role_renders_as_model(self):
        """The template renames the role; the reduction has to carry the RENDERED tag, not the name
        the caller uses. Getting this wrong is invisible until the model answers as `assistant`."""
        self.assertEqual(self.parts.block("assistant", "hi"),
                          "<start_of_turn>model\nhi<end_of_turn>\n")

    def test_trim_is_detected(self):
        self.assertTrue(self.parts.trim_content)
        self.assertEqual(self.parts.render([("user", "  hi  ")], False),
                          "<bos><start_of_turn>user\nhi<end_of_turn>\n")

    def test_the_doubled_bos_is_invisible_in_the_text_and_caught_in_the_ids(self):
        """The reason `verify_chat_template` has an id check at all. Before the strip the STRING
        comparison passes and the ids are one BOS too long; after it, both pass."""
        with tempfile.TemporaryDirectory() as d:
            _write_tokenizer_config(d, add_bos_token=True, bos_token="<bos>")
            problems = verify_chat_template(self.tok, self.parts)
            self.assertTrue(problems, "the string check alone would have approved a doubled BOS")
            self.assertTrue(all("ids" in p for p in problems))

            strip_duplicate_bos(self.parts, d)
            self.assertEqual(self.parts.prologue, "")
            self.assertEqual(self.parts.stripped_bos, "<bos>")
            self.assertEqual(verify_chat_template(self.tok, self.parts), [])

    def test_add_bos_token_is_read_from_the_config_not_the_tokenizer_object(self):
        """LFM2's `add_bos_token` is true in its config and absent as an attribute on the loaded
        `PreTrainedTokenizerFast`, so reading the object strips nothing and the BOS doubles. The
        config is also what `write_bpe_vocab` reads to decide the KV the engine acts on."""
        with tempfile.TemporaryDirectory() as d:
            _write_tokenizer_config(d, bos_token="<bos>")  # no add_bos_token key at all
            strip_duplicate_bos(self.parts, d)
            self.assertEqual(self.parts.prologue, "<bos>", "nothing to strip when the vocab adds none")


class TestTemplatesThatMustBeRefused(unittest.TestCase):
    """A decomposition that cannot be verified is not written, and the model then has no chat door
    rather than a wrong one. Each of these is a real shape."""

    def test_a_checkpoint_with_no_template_declines(self):
        tok = _FakeTokenizer(_chatml)
        tok.chat_template = None
        self.assertFalse(derive_chat_template(tok))

    def test_a_turn_whose_rendering_depends_on_its_position_declines(self):
        """Qwen3-0.6B-Base's shape: the template rewrites EARLIER assistant turns (it strips their
        `<think>` blocks), so a three-message render is not the two-message one with a block appended
        and no amount of differencing recovers a stable block."""
        def render(messages, add_generation_prompt):
            # Wrapped rather than upper-cased: the sentinels are already upper-case, so a case change
            # is a no-op on exactly the strings the differencing looks at -- a fake that mutates
            # invisibly tests nothing, which is how this case first passed for the wrong reason.
            trimmed = [(r, c if i == len(messages) - 1 else f"<think>{c}</think>")
                       for i, (r, c) in enumerate(messages)]
            return _chatml(trimmed, add_generation_prompt)
        parts = derive_chat_template(_FakeTokenizer(render))
        self.assertFalse(parts)
        # Reported against `assistant`, because that is the differencing step that hits it first: the
        # two-message render rewrote the user turn the one-message render had already produced. Which
        # step notices is not the claim -- that a reason is RECORDED, and names the property, is.
        self.assertEqual(list(parts.unsupported), ["assistant"])
        self.assertIn("is not independent of what follows it", parts.unsupported["assistant"])

    def test_a_template_that_refuses_every_conversation_declines(self):
        def render(messages, add_generation_prompt):
            raise ValueError("nope")
        parts = derive_chat_template(_FakeTokenizer(render))
        self.assertFalse(parts)
        self.assertIn("user", parts.unsupported)

    def test_a_generation_prompt_that_is_not_appended_declines(self):
        """`add_generation_prompt` must ADD to the conversation's own render. A template that
        restructures instead has no `generation_prefix` to name."""
        def render(messages, add_generation_prompt):
            if add_generation_prompt:
                return "PREFIX" + _chatml(messages, False) + "<|im_start|>assistant\n"
            return _chatml(messages, False)
        parts = derive_chat_template(_FakeTokenizer(render))
        self.assertFalse(parts)

    def test_falsiness_is_about_a_usable_turn_not_about_emptiness(self):
        """`bool(parts)` means "both halves of a turn are expressible". A decomposition with only a
        user role is not a chat template: nothing can answer."""
        parts = derive_chat_template(_FakeTokenizer(_chatml))
        self.assertTrue(parts)
        parts.roles = ["user"]
        self.assertFalse(parts)


if __name__ == "__main__":
    unittest.main()
