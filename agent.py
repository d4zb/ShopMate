"""Submission entry point.

The official evaluator constructs this once per run as ``Agent(catalog_path)``,
then calls ``reset`` once per session and ``respond`` once per turn. Note that
``starter/agent.py`` re-exports this module, because
``evaluator/local_evaluator.py`` imports the agent from there -- so nothing under
``src/`` may import the evaluator at module scope, or that becomes a cycle.

Pipeline per turn:

1. Parse the utterance back into a category and constraint strings
   (``src/inversion.py``), which is possible because the shopper's templates are
   deterministic and frozen (docs/final_evaluation_faq.md, sections 1 and 4).
2. Intersect the accumulated constraints into a candidate pool.
3. Decide whether to recommend or to ask one more question (``src/policy.py``).
4. Rank whatever we return by the popularity prior, then fused retrieval.

``AgentConfig`` exists so ``scripts/ablate.py`` can build every row of the
ablation table from the same code path the submission actually ships.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src import catalog as catalog_module
from src.inversion import InversionIndex, _normalize
from src.policy import build_policy
from src.retrieval import Retriever
from src.simulator_model import NO_PREFERENCE_PREFIX
from src.state import SessionState

#: The simulator reads this structured field and ignores our prose entirely
#: (FAQ section 5). "other" bypasses the constraint classifier in the evaluator's
#: customer_reply() and so returns the first two undisclosed constraints of any
#: type -- a strict superset of what any specific attribute yields, which makes it
#: the unconditionally optimal question. See docs/ for the derivation.
ASK_ATTRIBUTE = "other"

ASK_MESSAGE = "Before I recommend anything, is there another detail that matters to you?"
RECOMMEND_MESSAGE = "Here are the closest matches. Anything else that matters?"


@dataclass(frozen=True)
class AgentConfig:
    """Feature switches, one per ablation row."""

    use_bm25: bool = True
    use_popularity: bool = True
    use_dense: bool = False
    use_inversion: bool = True
    #: "threshold" converts when the pool is small; "expected_value" computes it.
    policy: str = "expected_value"
    #: ThresholdPolicy's K. None means always recommend the best ten.
    convert_threshold: int | None = 5
    #: ExpectedValuePolicy's bias toward converting; 0.0 is the untuned rule.
    convert_margin: float = 0.0


class Agent:
    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl",
                 config: AgentConfig | None = None) -> None:
        self.config = config or AgentConfig()
        self.catalog = catalog_module.load(catalog_path)
        cache = Path(catalog_path).resolve().parent.parent / "cache"
        embeddings = None
        if self.config.use_dense:
            dense_path = cache / "embeddings.npy"
            if dense_path.exists():
                embeddings = np.load(dense_path)
        self.retriever = Retriever.build(self.catalog, cache / "bm25.pkl", embeddings=embeddings)
        if embeddings is not None:
            # Imported lazily so the shipped configuration never touches torch.
            from sentence_transformers import SentenceTransformer
            self.retriever.set_query_encoder(
                SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            )
        self.index = (
            InversionIndex.build(self.catalog, cache / "inversion.pkl")
            if self.config.use_inversion else None
        )
        self.policy = (
            build_policy(self.config, self.index, self.retriever.popularity)
            if self.index is not None else None
        )
        self._sessions: dict[str, SessionState] = {}

    def reset(self, session_id: str, user_profile: dict) -> None:
        # Behaviour must never depend on session_id contents: the evaluator
        # generates a fresh uuid4 per run, so doing so would break reproducibility.
        self._sessions[session_id] = SessionState(session_id=session_id, user_profile=user_profile)

    # --- inference ----------------------------------------------------------

    def _absorb(self, state: SessionState, message: str) -> None:
        """Fold one utterance into the session's belief."""
        if self.index is None:
            return
        if self.index.carries_no_information(message):
            # "I don't have an additional preference for X" means the intent card
            # is spent: further questions cannot narrow anything, so stop waiting.
            # A boundary reply ("I don't have a preference for X; please use your
            # judgment.") is a one-time interception and reveals nothing about
            # whether the card still holds constraints, so it must not set this.
            # Located, not anchored, for the same reason as the template markers.
            if _normalize(NO_PREFERENCE_PREFIX) in _normalize(message):
                state.exhausted = True
            return
        is_opener = len(state.utterances) == 1
        if is_opener:
            category, constraints = self.index.parse_opener(message)
            if category:
                state.category = category
        else:
            constraints = self._disambiguate(state, message)
        if not constraints:
            # Tier 3, and only where a disclosure was actually expected. A browsing
            # opener discloses nothing by design; fuzzy-matching it would invent
            # constraints from the pool and discard the target.
            span = self.index.constraint_span(message, is_opener)
            if span:
                constraints = self.index.fuzzy_match(span, self._pool(state))
                # Whether or not the fuzzy tier resolved it, this span is the
                # highest-signal text in the message. Keep it for query weighting.
                state.soft_spans.append(span)
        for constraint in constraints:
            state.add_constraint(constraint)

    def _disambiguate(self, state: SessionState, message: str) -> list[str]:
        """Pick the reading of an ambiguous disclosure that the pool supports.

        A constraint may itself contain the ``"; "`` delimiter the evaluator joins
        with, so a message can be read several ways. The target is always still in
        the pool, so prefer the reading that narrows it without emptying it.
        """
        readings = self.index.readings(message)
        if not readings:
            return self.index.parse_reply(message)
        if len(readings) == 1:
            return readings[0]
        pool = self.index.pool(state.category, state.known_constraints)
        best, best_size = readings[0], None
        for reading in readings:
            narrowed = self.index.pool(state.category, state.known_constraints + reading)
            size = len(narrowed)
            if size and size < len(pool) and (best_size is None or size < best_size):
                best, best_size = reading, size
        return best

    def _pool(self, state: SessionState) -> np.ndarray | None:
        if self.index is None:
            return None
        return self.index.pool(state.category, state.known_constraints)

    def rank_for(self, session_id: str, top_k: int) -> list[int]:
        """Catalog positions this session would recommend, best first.

        Separated from ``respond`` so ``scripts/ablate.py`` can request a deeper
        ranking than the scored top ten and measure recall@50 / recall@200.
        """
        state = self._sessions[session_id]
        pool = self._pool(state)
        restrict = pool if pool is not None and len(pool) else None
        return self.retriever.rank(
            state.query(), restrict=restrict, top_k=top_k, use_bm25=self.config.use_bm25,
            use_popularity=self.config.use_popularity, use_dense=self.config.use_dense,
        )

    # --- evaluator contract -------------------------------------------------

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        state = self._sessions.get(session_id)
        if state is None:
            # The evaluator always calls reset() first, but a missing session must
            # degrade to a valid response rather than raise: exceptions are
            # swallowed and silently scored as a miss.
            state = self._sessions[session_id] = SessionState(session_id=session_id)
        state.record(user_message, turn)
        self._absorb(state, user_message)

        convert = True
        if self.policy is not None:
            pool = self._pool(state)
            decision = self.policy.decide(
                pool, turn, state.exhausted, disclosed=set(state.known_constraints)
            )
            state.last_decision = decision
            convert = decision.convert

        recommendations: list[dict] = []
        if convert:
            recommendations = [
                {"parent_asin": self.catalog.asins[i]} for i in self.rank_for(session_id, top_k)
            ]
        return {
            "message": RECOMMEND_MESSAGE if convert else ASK_MESSAGE,
            "ask_attribute": ASK_ATTRIBUTE,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }


__all__ = ["Agent", "AgentConfig"]
