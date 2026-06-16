"""MoRE+ config + unit-split — round-trips and isolation from MoRE (no model)."""

import tempfile

import shadowlm as slm
from shadowlm import more, more_plus as mp


def test_config_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        mp.write_config(d, base_model="Qwen/x", lora_r=4, lora_alpha=8,
                        final_layer_idx=23, num_experts=12, k=3, tau=0.6, group_size=2)
        cfg = mp.read_config(d)
    assert cfg["type"] == "more_plus" and cfg["version"] == 1
    assert (cfg["lora_r"], cfg["lora_alpha"], cfg["final_layer_idx"]) == (4, 8, 23)
    assert (cfg["num_experts"], cfg["k"], cfg["tau"], cfg["group_size"]) == (12, 3, 0.6, 2)


def test_config_isolation_from_more():
    # the two methods use distinct config filenames → neither misfires on the other
    with tempfile.TemporaryDirectory() as d:
        mp.write_config(d, base_model="m", lora_r=4, lora_alpha=4, final_layer_idx=1,
                        num_experts=1, k=1, tau=0.5, group_size=1)
        assert more.read_config(d) is None
    with tempfile.TemporaryDirectory() as d:
        more.write_config(d, base_model="m", rank=8, k=2, num_layers=4)
        assert mp.read_config(d) is None


def test_split_units_cardinality_and_surrogate():
    ds = slm.Dataset.from_list([{"question": f"q{i}", "answer": f"a{i}"} for i in range(6)])
    assert len(mp.split_units(ds, 1)) == 6
    assert len(mp.split_units(ds, 3)) == 2
    surrogate, rows = mp.split_units(ds, 3)[0]
    assert surrogate == "q0" and len(rows) == 3


def test_split_units_chat_surrogate():
    ds = slm.Dataset.from_list([
        {"messages": [{"role": "user", "content": "ask me"},
                      {"role": "assistant", "content": "answer"}]}])
    assert mp.split_units(ds, 1)[0][0] == "ask me"
