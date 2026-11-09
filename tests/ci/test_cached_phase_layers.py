"""Two cached phases in one export share ONE KV cache, so their slots must not collide (loom.cpp
Retro-057).

`fuse_loom_attention` numbers each phase's ATTENTION blocks from 0 because it sees one program, and the
engine gives every cached module that is not a private stream the same cache, sized for all of their
blocks together (`_kv_cache_geometry`). VoxCPM2 was the first export with two different cached stacks,
and its residual LM wrote over the base LM's first eight layers on every call -- with a prefill that
still matched the reference exactly, since neither stack reads its cache until the first decode step.
`offset_cached_attention_layers` moves each later phase past the earlier ones' slots.
"""
from loom_exporter.decomposition import offset_cached_attention_layers


def _topology(n_cached: int, extra=()) -> dict:
    nodes = [{"op": "ATTENTION", "inputs": [], "outputs": [f"a{i}"], "attrs": {"layer": i, "kv_cache": True}}
             for i in range(n_cached)]
    return {"nodes": nodes + list(extra)}


def test_a_later_phase_moves_past_the_earlier_phases_slots():
    base, residual = _topology(28), _topology(8)
    offset_cached_attention_layers({"base_lm": base}, 0)
    offset_cached_attention_layers({"residual_lm": residual}, 28)
    base_slots = [n["attrs"]["layer"] for n in base["nodes"]]
    residual_slots = [n["attrs"]["layer"] for n in residual["nodes"]]
    assert base_slots == list(range(28))
    assert residual_slots == list(range(28, 36))


def test_the_first_cached_phase_is_left_byte_for_byte():
    """Every export before VoxCPM2 has one cached phase, and its bytes must not move."""
    topo = _topology(6)
    before = [dict(n["attrs"]) for n in topo["nodes"]]
    offset_cached_attention_layers({"lm": topo}, 0)
    assert [n["attrs"] for n in topo["nodes"]] == before


def test_an_alias_stream_sharing_the_node_list_is_shifted_once():
    """`extra_streams` makes a shallow copy that shares the phase's node list; shifting both dicts
    would move the shared nodes twice."""
    topo = _topology(3)
    topologies = {"decoder": topo, "decoder_uncond": dict(topo, kv_cache_scope="private")}
    offset_cached_attention_layers(topologies, 10)
    assert [n["attrs"]["layer"] for n in topo["nodes"]] == [10, 11, 12]


def test_an_uncached_attention_and_other_ops_are_untouched():
    extra = ({"op": "ATTENTION", "inputs": [], "outputs": ["x"], "attrs": {"layer": 0, "kv_cache": False}},
             {"op": "MUL_MAT", "inputs": [], "outputs": ["y"]})
    topo = _topology(2, extra)
    offset_cached_attention_layers({"p": topo}, 5)
    assert [n.get("attrs", {}).get("layer") for n in topo["nodes"]] == [5, 6, 0, None]


class _Phase:
    def __init__(self, name, size, own):
        self.name, self.kv_cache_size, self.own_kv_cache = name, size, own


class _Result:
    def __init__(self, topologies, blocks):
        self.topologies = topologies
        self.geometry = {"attention": [(f"op{i}", g) for i, g in enumerate(blocks)]}


def test_a_phase_with_its_own_cache_declares_its_geometry_and_keeps_its_slots():
    """loom.cpp ADR-103: a phase whose cache is its own (MOSS-TTS's codec stages) writes the `kv_cache`
    object the engine allocates privately, from its own fused blocks, with its layers still from 0."""
    from loom_exporter.decomposition import declare_own_kv_cache

    topo = _topology(3)
    declare_own_kv_cache(_Phase("stage_2", 516, {"ring": True, "window": 500}),
                         _Result({"stage_2": topo}, [(12, 64, 64)] * 3))
    assert topo["kv_cache"] == {"n_layer": 3, "n_embd_k": 768, "n_embd_v": 768, "size": 516, "ring": True,
                                "window": 500}
    assert [n["attrs"]["layer"] for n in topo["nodes"]] == [0, 1, 2]


def test_an_own_cache_refuses_what_it_cannot_hold():
    import pytest

    from loom_exporter.decomposition import declare_own_kv_cache

    with pytest.raises(ValueError, match="fused no cached ATTENTION"):
        declare_own_kv_cache(_Phase("p", 8, {"ring": True}), _Result({"p": _topology(0)}, []))
    with pytest.raises(NotImplementedError, match="one width per layer"):
        declare_own_kv_cache(_Phase("p", 8, {"ring": True}), _Result({"p": _topology(2)}, [(2, 4, 4), (1, 4, 4)]))
    with pytest.raises(ValueError, match="window of 9"):
        declare_own_kv_cache(_Phase("p", 8, {"ring": True, "window": 9}), _Result({"p": _topology(1)}, [(1, 4, 4)]))
