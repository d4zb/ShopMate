"""Retrieval floor: BM25, a popularity prior, and rank fusion.

This is the fallback layer. The inversion path in ``src/inversion.py`` handles
the common case; retrieval carries the load whenever a message fails to parse or
a constraint intersection is unusable.

Two things here are load-bearing and worth stating up front.

**BM25 is a hand-rolled inverted index rather than ``rank_bm25``.** The library's
``BM25Okapi.get_scores`` walks all 50,000 documents per query term and measures
at ~200 ms/query, which is ~400 s for a fallback-only evaluation run and blows
the two-minute budget. Because k1 and b are fixed, the entire per-posting term
weight is precomputable at build time, which reduces scoring to a scatter-add
over just the postings of the query terms. ``rank_bm25`` is retained as the
reference implementation that ``tests/test_retrieval.py`` validates against.

**The popularity prior is a first-class ranking signal, not a tie-break.**
Targets are drawn from real purchase records, so they sit at the 95.6th
percentile of the catalog by ``rating_number`` (median 6,846 against a catalog
median of 12). Ranking a category-filtered pool by popularity alone reaches
Hit@10 = 0.815 on turn 1, so this is the strongest single non-conversational
signal available.
"""

from __future__ import annotations

import pickle
import re
from array import array
from collections import Counter
from pathlib import Path

import numpy as np

from .catalog import Catalog

TOKEN_RE = re.compile(r"[a-z0-9]+")

#: Conversational filler carried by the simulator's templates. These appear in
#: every message and would otherwise dominate the query.
STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "have",
    "i", "in", "is", "it", "its", "just", "looking", "matters", "me", "my", "not",
    "of", "on", "or", "please", "prefer", "preference", "requirement", "s", "so",
    "some", "still", "that", "the", "this", "to", "want", "what", "with", "would",
    "you", "your", "key", "exploring", "actually", "ignore", "earlier", "need",
})

K1 = 1.5
B = 0.75
#: ``rank_bm25`` floors negative IDFs at ``EPSILON * mean(idf)``. Replicated so the
#: equivalence test in tests/test_retrieval.py is strict rather than approximate.
EPSILON = 0.25
#: Reciprocal-rank-fusion damping. 60 is the value from the original RRF paper.
RRF_K = 60


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if len(t) > 1 and t not in STOPWORDS]


class Bm25Index:
    """Okapi BM25 over a precomputed inverted index with baked-in term weights.

    Numerically identical to ``rank_bm25.BM25Okapi(corpus, k1=1.5, b=0.75)``,
    including its negative-IDF flooring and its lack of query-term deduplication.
    """

    def __init__(self, corpus: list[list[str]]) -> None:
        self.n_docs = len(corpus)
        lengths = np.fromiter((len(doc) for doc in corpus), dtype=np.float32, count=self.n_docs)
        avgdl = float(lengths.mean()) or 1.0

        vocab: dict[str, int] = {}
        term_ids, doc_ids, raw_tf = array("i"), array("i"), array("f")
        for doc_id, tokens in enumerate(corpus):
            for term, tf in Counter(tokens).items():
                term_id = vocab.get(term)
                if term_id is None:
                    term_id = vocab[term] = len(vocab)
                term_ids.append(term_id)
                doc_ids.append(doc_id)
                raw_tf.append(tf)

        terms = np.frombuffer(term_ids, dtype=np.int32)
        docs = np.frombuffer(doc_ids, dtype=np.int32)
        tfs = np.frombuffer(raw_tf, dtype=np.float32).astype(np.float64)

        order = np.argsort(terms, kind="stable")
        terms, docs, tfs = terms[order], docs[order], tfs[order]

        # Document frequency per term, and the offsets that slice the postings.
        counts = np.bincount(terms, minlength=len(vocab))
        offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)

        idf = np.log(self.n_docs - counts + 0.5) - np.log(counts + 0.5)
        # A term in more than half the corpus scores negative; rank_bm25 replaces
        # those with a small positive floor derived from the pre-flooring mean.
        idf[idf < 0] = EPSILON * float(idf.mean())

        # k1 and b are fixed, so fold the whole BM25 term weight into the postings.
        norm = K1 * (1.0 - B + B * lengths[docs].astype(np.float64) / avgdl)
        self._weights = idf[terms] * tfs * (K1 + 1.0) / (tfs + norm)
        self._docs = docs
        self._offsets = offsets
        self._vocab = vocab
        # Terms absent from a document still contribute idf * 0 * ... == 0, so
        # only the postings matter; unknown query terms contribute nothing at all.

    def score(self, query: list[str]) -> np.ndarray:
        """Dense score vector over the catalog. Absent terms contribute nothing.

        Query terms are deliberately *not* deduplicated, matching ``rank_bm25``:
        a term repeated in the query is weighted twice.
        """
        out = np.zeros(self.n_docs, dtype=np.float64)
        for term in query:
            term_id = self._vocab.get(term)
            if term_id is None:
                continue
            lo, hi = self._offsets[term_id], self._offsets[term_id + 1]
            # Document ids are unique within a term's postings (one entry per
            # distinct doc/term pair), so plain fancy-index accumulation is safe
            # here and roughly an order of magnitude faster than np.add.at.
            out[self._docs[lo:hi]] += self._weights[lo:hi]
        return out


def _ranks_from_scores(scores: np.ndarray) -> np.ndarray:
    """Dense 1-based rank per catalog position, best score first."""
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(order) + 1)
    return ranks


class Retriever:
    """Fuses BM25, the popularity prior, and optional dense similarity."""

    CACHE_VERSION = 1

    @classmethod
    def build(cls, catalog: Catalog, cache: Path | None = None,
              embeddings: np.ndarray | None = None) -> "Retriever":
        """Load cached BM25 postings if they match this catalog, else build and cache."""
        bm25 = None
        if cache is not None and cache.exists():
            try:
                with cache.open("rb") as handle:
                    payload = pickle.load(handle)
                if (payload.get("version") == cls.CACHE_VERSION
                        and payload.get("n_docs") == len(catalog)):
                    bm25 = Bm25Index.__new__(Bm25Index)
                    bm25.__dict__.update(payload["bm25"])
            except (OSError, pickle.UnpicklingError, KeyError, AttributeError):
                bm25 = None
        retriever = cls(catalog, embeddings=embeddings, _bm25=bm25)
        if cache is not None and bm25 is None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            with cache.open("wb") as handle:
                pickle.dump({"version": cls.CACHE_VERSION, "n_docs": len(catalog),
                             "bm25": retriever.bm25.__dict__},
                            handle, protocol=pickle.HIGHEST_PROTOCOL)
        return retriever

    def __init__(self, catalog: Catalog, embeddings: np.ndarray | None = None,
                 _bm25: "Bm25Index | None" = None) -> None:
        self.catalog = catalog
        self.bm25 = _bm25 or Bm25Index([tokenize(doc) for doc in catalog.documents()])
        self.popularity = np.fromiter(
            (float(p.get("rating_number") or 0) for p in catalog.products),
            dtype=np.float32, count=len(catalog),
        )
        # Precomputed once: the ranking used whenever there is no query signal.
        self.popularity_rank = _ranks_from_scores(self.popularity)
        self.embeddings = embeddings
        self._query_encoder = None

    def set_query_encoder(self, encoder) -> None:
        """Attach a sentence-transformers model for the optional dense route."""
        self._query_encoder = encoder

    def _dense_scores(self, query: str) -> np.ndarray | None:
        if self.embeddings is None or self._query_encoder is None:
            return None
        vector = self._query_encoder.encode([query], normalize_embeddings=True)[0]
        return self.embeddings @ vector.astype(np.float32)

    def fuse(self, query: str, use_bm25: bool = True, use_popularity: bool = True,
             use_dense: bool = False) -> np.ndarray:
        """Fused score per catalog position, higher is better.

        Fusion is reciprocal rank fusion rather than score addition: BM25 scores
        and cosine similarities are on incompatible scales, and the popularity
        prior is on a third scale entirely.
        """
        fused = np.zeros(len(self.catalog), dtype=np.float64)
        if use_popularity:
            fused += 1.0 / (RRF_K + self.popularity_rank)
        tokens = tokenize(query)
        if use_bm25 and tokens:
            fused += 1.0 / (RRF_K + _ranks_from_scores(self.bm25.score(tokens)))
        if use_dense:
            dense = self._dense_scores(query)
            if dense is not None:
                fused += 1.0 / (RRF_K + _ranks_from_scores(dense))
        return fused

    def rank(self, query: str, restrict: np.ndarray | None = None, top_k: int = 10,
             **flags) -> list[int]:
        """Catalog positions best-first, optionally restricted to a candidate pool."""
        fused = self.fuse(query, **flags)
        if restrict is not None:
            if len(restrict) == 0:
                return []
            best = restrict[np.argsort(-fused[restrict], kind="stable")]
            return [int(i) for i in best[:top_k]]
        return [int(i) for i in np.argsort(-fused, kind="stable")[:top_k]]

    def rank_by_popularity(self, pool: np.ndarray, top_k: int = 10) -> list[int]:
        """Pure popularity ordering within a pool, used once inversion has narrowed it."""
        if len(pool) == 0:
            return []
        best = pool[np.argsort(-self.popularity[pool], kind="stable")]
        return [int(i) for i in best[:top_k]]
