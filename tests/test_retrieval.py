"""Validate the hand-rolled BM25 against ``rank_bm25``, its reference implementation.

``src/retrieval.py`` does not use ``rank_bm25`` at runtime: its ``get_scores``
walks all 50,000 documents per query term and measures at ~200 ms/query, which
alone would exceed the evaluation budget. Ours precomputes the per-posting weight
and scatter-adds over just the query terms' postings, ~335x faster.

That optimisation is only safe if it computes the same thing, so the library is
kept as the oracle and checked here. The match is exact rather than approximate:
we replicate BM25Okapi's specific IDF, its epsilon-flooring of negative IDFs, and
its lack of query-term deduplication.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.catalog import load  # noqa: E402
from src.retrieval import Bm25Index, Retriever, tokenize  # noqa: E402

CATALOG = ROOT / "data" / "catalog.jsonl"
SAMPLE = 4000
TOLERANCE = 1e-6

QUERIES = [
    "leather ankle boots",
    "cotton crew neck tshirt",
    "sterling silver hoop earrings",
    "waterproof hiking shoes men",
    "black dress women summer",
    "imported",                      # very common term, exercises the IDF floor
    "cotton cotton polyester",       # repeated term: rank_bm25 counts it twice
    "zzzznotarealterm",              # unknown term contributes nothing
    "100 cotton machine wash",
]


@pytest.fixture(scope="module")
def corpus():
    if not CATALOG.exists():
        pytest.skip("data/catalog.jsonl missing; run scripts/fetch_data.py")
    catalog = load(CATALOG)
    return catalog, [tokenize(catalog.document(i)) for i in range(SAMPLE)]


@pytest.fixture(scope="module")
def indexes(corpus):
    from rank_bm25 import BM25Okapi
    _, documents = corpus
    return BM25Okapi(documents, k1=1.5, b=0.75), Bm25Index(documents)


@pytest.mark.parametrize("query", QUERIES)
def test_scores_match_rank_bm25(indexes, query: str) -> None:
    reference, ours = indexes
    tokens = tokenize(query)
    expected = reference.get_scores(tokens)
    actual = ours.score(tokens)
    assert np.abs(expected - actual).max() < TOLERANCE, query


@pytest.mark.parametrize("query", QUERIES)
def test_top_20_ordering_matches_rank_bm25(indexes, query: str) -> None:
    reference, ours = indexes
    tokens = tokenize(query)
    expected = np.argsort(-reference.get_scores(tokens), kind="stable")[:20]
    actual = np.argsort(-ours.score(tokens), kind="stable")[:20]
    assert list(expected) == list(actual), query


def test_empty_query_scores_nothing(indexes) -> None:
    _, ours = indexes
    assert not ours.score([]).any()


def test_cached_index_is_identical_to_a_fresh_build(corpus, tmp_path) -> None:
    """A stale or lossy cache would silently change every ranking."""
    catalog, _ = corpus
    cache = tmp_path / "bm25.pkl"
    built = Retriever.build(catalog, cache)
    assert cache.exists()
    loaded = Retriever.build(catalog, cache)

    query = "leather ankle boots waterproof"
    assert np.array_equal(built.bm25.score(tokenize(query)), loaded.bm25.score(tokenize(query)))
    assert np.array_equal(built.popularity_rank, loaded.popularity_rank)
    assert built.rank(query, top_k=25) == loaded.rank(query, top_k=25)


def test_popularity_prior_orders_by_rating_number(corpus) -> None:
    """Targets sit at the 95.6th percentile by rating_number, so this ordering matters."""
    catalog, _ = corpus
    retriever = Retriever(catalog)
    top = retriever.rank_by_popularity(np.arange(len(catalog), dtype=np.int32), top_k=50)
    counts = [catalog.products[i]["rating_number"] for i in top]
    assert counts == sorted(counts, reverse=True)
