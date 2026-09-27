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
