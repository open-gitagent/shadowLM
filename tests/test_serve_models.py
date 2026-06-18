"""Studio custom-model store — add / dedup / persist / remove (no HTTP)."""

import json
import tempfile
from pathlib import Path

from shadowlm.serve import Server


def _server(tmp):
    return Server(backend="mlx", accelerator="none", device="cpu", work_root=Path(tmp))


def test_add_lists_first_and_marks_custom():
    with tempfile.TemporaryDirectory() as tmp:
        s = _server(tmp)
        n_curated = len(s.catalog())
        s.add_custom_model("org/cool-model")
        cat = s.catalog()
        assert cat[0]["id"] == "org/cool-model" and cat[0]["custom"] is True
        assert len(cat) == n_curated + 1


def test_dedup_against_catalog_and_self():
    with tempfile.TemporaryDirectory() as tmp:
        s = _server(tmp)
        curated_id = s.catalog()[-1]["id"]
        s.add_custom_model(curated_id)            # already curated → no-op
        s.add_custom_model("org/x")
        s.add_custom_model("org/x")               # duplicate → no-op
        assert [m["id"] for m in s._custom_models] == ["org/x"]


def test_persists_and_survives_reload():
    with tempfile.TemporaryDirectory() as tmp:
        _server(tmp).add_custom_model("org/persisted")
        saved = json.loads((Path(tmp) / "custom_models.json").read_text())
        assert saved[0]["id"] == "org/persisted"
        assert "org/persisted" in [m["id"] for m in _server(tmp).catalog()]  # fresh load


def test_remove():
    with tempfile.TemporaryDirectory() as tmp:
        s = _server(tmp)
        s.add_custom_model("org/temp")
        s.remove_custom_model("org/temp")
        assert "org/temp" not in [m["id"] for m in s.catalog()]
