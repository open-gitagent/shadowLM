"""DatasetStore seeds a fresh studio with samples + a curated catalog, once."""

import tempfile
from pathlib import Path

from shadowlm.serve import DatasetStore


def test_fresh_store_seeds_samples_and_curated():
    with tempfile.TemporaryDirectory() as tmp:
        ds = DatasetStore(Path(tmp) / "datasets")
        metas = ds.list()
        names = {m["name"] for m in metas}
        repos = {m.get("repo") for m in metas if m.get("source") == "hf"}
        assert "ShadowLM Q&A · chat" in names          # a bundled sample
        assert "yahma/alpaca-cleaned" in repos          # a curated HF ref
        assert any(m["source"] == "upload" for m in metas)
        assert any(m["source"] == "hf" for m in metas)


def test_seeding_is_idempotent_and_respects_deletes():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "datasets"
        first = DatasetStore(root).list()
        # delete one, re-open: store is non-empty so it must NOT re-seed
        DatasetStore(root).delete(first[0]["dataset_id"])
        reopened = DatasetStore(root).list()
        assert len(reopened) == len(first) - 1
