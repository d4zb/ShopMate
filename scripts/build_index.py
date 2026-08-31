"""Precompute and cache the offline artifacts the agent loads at startup.

    python scripts/build_index.py --dense

Everything here is derived purely from the frozen catalog, which the organisers
explicitly permit (FAQ section 4: catalog-derived embeddings, derived attributes,
and local sidecar files are all allowed). Nothing depends on session data.

The dense route is optional and off by default: it costs a ~2.5 GB torch
dependency and a 77 MB matrix, so it has to earn its place in the ablation table
before it ships.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.catalog import load  # noqa: E402

CACHE = ROOT / "cache"
DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDINGS = CACHE / "embeddings.npy"
BATCH = 256
#: MiniLM truncates at 256 word pieces anyway; trimming first saves tokenizer time.
MAX_CHARS = 1024


def build_dense(catalog, force: bool = False) -> Path:
    if EMBEDDINGS.exists() and not force:
        print(f"{EMBEDDINGS} already present; pass --force to rebuild")
        return EMBEDDINGS
    from sentence_transformers import SentenceTransformer

    print(f"encoding {len(catalog):,} products with {DENSE_MODEL} (CPU)")
    model = SentenceTransformer(DENSE_MODEL)
    documents = [catalog.document(i)[:MAX_CHARS] for i in range(len(catalog))]
    started = time.perf_counter()
    vectors = model.encode(
        documents, batch_size=BATCH, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=True,
    ).astype(np.float32)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS, vectors)
    print(f"  {vectors.shape} -> {EMBEDDINGS} ({EMBEDDINGS.stat().st_size / 1e6:.0f} MB) "
          f"in {time.perf_counter() - started:.0f}s")
    return EMBEDDINGS


def main() -> int:
    parser = argparse.ArgumentParser(description="Build cached offline artifacts")
    parser.add_argument("--catalog", type=Path, default=ROOT / "data" / "catalog.jsonl")
    parser.add_argument("--dense", action="store_true", help="encode the catalog with MiniLM")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    catalog = load(args.catalog)
    print(f"catalog: {len(catalog):,} products")
    if args.dense:
        build_dense(catalog, force=args.force)
    else:
        print("nothing to do (pass --dense)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
