"""Covers `shape_expr.py`: the grammar boundary between the exporter's algebra and the engine.

The properties worth pinning are the ones a future change could silently break -- that `render` never
emits something `src/core/symbol_env.cpp` cannot parse (and raises loudly instead), that `parse` accepts
exactly what `render` produces, and that the assumptions on shape symbols (positive integers) are what
make the simplifications real rather than cosmetic.

Run: ~/.venvs/piper/bin/python3 -m pytest tests/ci/test_shape_expr.py
"""
import re
import sys
from pathlib import Path

import pytest
import sympy

from loom_exporter.paths import CONVERTERS, driver_dir
from loom_exporter.shape_expr import (  # noqa: E402
    N_TOKENS,
    UnsupportedShapeExpression,
    as_expr,
    floor_div,
    has_dynamic_symbol,
    parse,
    render,
    sub_dynamic_symbols,
    symbol,
    to_number,
)

n = N_TOKENS

# Everything symbol_env.cpp's tokenizer can encounter: identifiers, digits, the four operators,
# parentheses, decimal points and spaces. Anything else (notably '**') means an unparseable attribute.
_ALLOWED_CHARS = re.compile(r"^[A-Za-z0-9_+\-*/()., ]*$")   # ',' since Max()/Min() take two arguments

EXPRESSIONS = [
    n,
    as_expr(64),
    as_expr(0),
    as_expr(-1),
    n / 160,
    sympy.floor((n - 512) / 160) + 1,
    (n - 1) * 8 - 8 + 16,
    600 * n + 20,
    4 * n * 64,
    (n + 2) * (n - 3),
    sympy.sqrt(n),
    n ** 2,
    1 / n,
    sympy.Float(1.5) * n,
    -n,
    floor_div(n * 512, 512),
    sympy.floor(n / 2) * 3,
    # A clamped shape -- "padded, but never by a negative amount". VITS's relative-position table is
    # the real one: `2*Max(n_tokens - 5, 0) + 9`, which is what let its 2048-wide static pad (18.9 MB
    # of zeros in the export) become a dynamically-sized one.
    sympy.Max(n - 5, 0),
    2 * sympy.Max(n - 5, 0) + 9,
    sympy.Min(n, 8),
    sympy.Max(sympy.Min(n, 10) * 2, 7),
]


@pytest.mark.parametrize("expr", EXPRESSIONS, ids=lambda e: str(e))
def test_render_round_trips_through_parse(expr):
    text = render(expr)
    assert _ALLOWED_CHARS.match(text), f"{text!r} contains a character symbol_env.cpp cannot tokenize"
    assert "**" not in text
    assert sympy.simplify(parse(text) - expr) == 0


def test_a_clamped_shape_matches_the_engine_at_both_sides_of_the_clamp():
    """`Max` exists in the grammar for exactly one reason -- a pad that must not go negative -- so the
    property worth pinning is that it is still right on the side where the clamp bites. VITS's window
    is 4, so `2*Max(n-5, 0) + 9` is `2n-1` above the window and a flat 9 at or below it."""
    expr = 2 * sympy.Max(n - 5, 0) + 9
    text = render(expr)
    assert _ALLOWED_CHARS.match(text)
    for probe, expected in ((2, 9), (4, 9), (5, 9), (6, 11), (62, 123), (5002, 10003)):
        assert float(parse(text).subs(n, probe)) == expected == float(expr.subs(n, probe))


def test_render_matches_the_engines_arithmetic_at_concrete_lengths():
    """The real contract: whatever `render` prints must evaluate to the same number the expression
    means, for every sequence length the model can be built at."""
    expr = sympy.floor((n - 512) / 160) + 1
    text = render(expr)
    for probe in (1600, 16000, 16001, 31999, 320000):
        assert float(parse(text).subs(n, probe)) == float(expr.subs(n, probe))


# -- what makes the simplification real ---------------------------------------------------------

def test_shape_symbols_are_positive_integers():
    """Not cosmetic: `floor` of a symbol only collapses when the symbol is known to be an integer, and
    that single fact is what turns StyleTTS2's nested-floor monster back into `n_tokens`."""
    assert n.is_integer and n.is_positive
    assert sympy.floor(n) == n
    assert floor_div(512 * n, 512) == n


def test_symbols_are_interned_so_assumptions_cannot_diverge():
    assert symbol("n_tokens") is N_TOKENS
    assert parse("n_tokens") is N_TOKENS


def test_integral_floats_become_integers():
    """A `Float` coefficient blocks the integer assumption, so `floor(1.0*n)` would not collapse."""
    assert as_expr(2.0) == sympy.Integer(2)
    assert sympy.floor(as_expr(1.0) * n) == n
    assert as_expr(1.5) != sympy.Integer(1)


def test_floor_arguments_are_recombined_over_one_denominator():
    """Sympy distributes rational coefficients over sums on construction; the engine evaluates in
    doubles, where one division rounds once and the distributed form rounds three times."""
    assert str(sympy.floor((n - 512) / 160)) == "floor(n_tokens/160 - 16/5)"
    assert render(sympy.floor((n - 512) / 160)) == "floor((n_tokens - 512)/160)"


# -- the grammar's edges ------------------------------------------------------------------------

@pytest.mark.parametrize("expr", [
    sympy.ceiling(n / 3),
    # Max/Min moved OUT of this list when symbol_env.cpp learned them -- a three-argument one is still
    # inexpressible, because sympy's Max is n-ary and the engine's grammar is strictly binary, and
    # folding one into nested pairs here would emit an expression nothing has ever evaluated. Three
    # SYMBOLS, not literals: sympy folds `Max(n, 8, 12)` down to two arguments on construction, so a
    # literal case would silently test the binary path instead.
    sympy.Max(n, symbol("n_past"), symbol("n_kv")),
    n ** sympy.Rational(1, 3),
    sympy.Piecewise((n, n > 2), (1, True)),
    sympy.log(n),
])
def test_inexpressible_constructs_raise_rather_than_emitting_bad_text(expr):
    with pytest.raises(UnsupportedShapeExpression):
        render(expr)


def test_small_integer_powers_expand_into_products():
    assert render(n ** 3) == "n_tokens*n_tokens*n_tokens"


def test_large_integer_powers_raise():
    with pytest.raises(UnsupportedShapeExpression):
        render(n ** 40)


def test_division_by_a_product_keeps_its_parentheses():
    """`a/(b*c)` and `a/b*c` are different numbers under the engine's left-to-right term rule."""
    a, b, c = symbol("a"), symbol("b"), symbol("c")
    text = render(a / (b * c))
    assert float(parse(text).subs({a: 12, b: 3, c: 2})) == 2.0


@pytest.mark.parametrize("text", ["n_tokens **", "floor(n_tokens", "n_tokens 3", "", "@"])
def test_parse_rejects_what_the_engine_would_reject(text):
    with pytest.raises(UnsupportedShapeExpression):
        parse(text)


def test_parse_accepts_the_engines_dollar_sigil():
    assert parse("$n_tokens + 1") == n + 1


# -- MIL symbol substitution --------------------------------------------------------------------

def test_opaque_mil_symbols_become_n_tokens():
    # `isN` is coremltools' own naming for a symbolic dim; MIL hands these over as sympy symbols
    # inside real expressions, e.g. a reshape's `4*is2`.
    assert sub_dynamic_symbols(4 * sympy.Symbol("is2")) == 4 * n
    assert has_dynamic_symbol(4 * sympy.Symbol("is2"))
    assert not has_dynamic_symbol(4 * n)


def test_overrides_replace_a_named_symbol_with_a_whole_expression():
    """Kokoro's decoder_vocoder declares several independent dynamic leaf inputs whose real lengths are
    fixed multiples of the one true quantity -- supplied by raw MIL symbol name."""
    expr = sub_dynamic_symbols(sympy.Symbol("is42") * 4 + 20, {"is42": "600*n_tokens+20"})
    assert expr == 2400 * n + 100
    assert render(expr) == "2400*n_tokens + 100"


def test_to_number_separates_literals_from_expressions():
    assert to_number(as_expr(3)) == 3
    assert isinstance(to_number(as_expr(3)), int)
    assert to_number(sympy.Rational(3, 2)) == 1.5
    assert to_number(n) is None
