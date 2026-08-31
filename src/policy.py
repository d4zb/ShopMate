"""When to recommend, and what to ask.

The brief this project started from assumed the tension was *asking costs turns,
so only ask when the precision gain outweighs the turn cost*. Under this
evaluator that tension does not exist. ``ask_attribute`` and ``recommendations``
are independent fields and a question is free (FAQ section 5), so asking every
turn strictly dominates.

The real tension runs the other way. ``local_evaluator.evaluate`` breaks out of
the turn loop the moment the target appears in the returned list, so a hit at a
poor rank is **permanent** -- there is no second chance to improve it. Converting
at turn 1 with the target at rank 7 scores 0.5 + 0.3/7 + 0.2 = 0.743; waiting one
turn and converting at rank 1 scores 0.5 + 0.3 + 0.18 = 0.98. One turn costs
0.02; lifting a session from rank 2 to rank 1 is worth 0.15. So the agent should
decline to recommend while it is uncertain, and commit once it is not.

Two policies implement that, and ``scripts/sweep.py`` compares them:

``ThresholdPolicy`` converts when the pool is at most K -- one tunable number.
``ExpectedValuePolicy`` computes both options under the known disclosure model
and converts when converting wins. It has no tuned parameter, so if it lands on
the swept optimum that is evidence the rule is derived rather than fitted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MAX_TURNS = 10
TOP_K = 10
#: Weights lifted from evaluator/local_evaluator.py:evaluate.
W_HIT, W_MRR, W_EFF = 0.50, 0.30, 0.20
#: Above this many candidates, one-step lookahead is both too costly and a
#: foregone conclusion: waiting always wins, so do not spend the time computing it.
LOOKAHEAD_CAP = 400


def turn_score(rank: int | None, turn: int) -> float:
    """Score a single session that converts at ``rank`` on ``turn``."""
    if rank is None or rank > TOP_K:
        return W_EFF * 0.0  # a miss is scored as turn 11, i.e. zero efficiency
    return W_HIT + W_MRR / rank + W_EFF * (11.0 - turn) / 10.0


@dataclass(frozen=True)
class Decision:
    convert: bool
    pool_size: int
    value_now: float
    value_wait: float

    @property
    def margin(self) -> float:
        return self.value_now - self.value_wait


class ThresholdPolicy:
    """Convert once the pool is small enough. K is the single tunable parameter."""

    def __init__(self, threshold: int | None) -> None:
        self.threshold = threshold

    def decide(self, pool: np.ndarray, turn: int, exhausted: bool, **_) -> Decision:
        size = len(pool)
        if self.threshold is None:
            return Decision(True, size, 0.0, 0.0)
        convert = exhausted or size <= self.threshold or turn >= MAX_TURNS - 2
        return Decision(convert, size, 0.0, 0.0)


class ExpectedValuePolicy:
    """One-step lookahead through the simulator's own disclosure policy.

    Believing each pool member equally likely to be the target, compare the score
    of converting now against the score of asking once more and converting next
    turn. The lookahead is exact rather than estimated: for each candidate we know
    precisely which constraints the shopper would disclose next if that candidate
    were the target, so we can compute the pool it would collapse to.
    """

    def __init__(self, index, popularity: np.ndarray, margin: float = 0.0) -> None:
        self.index = index
        self.popularity = popularity
        #: Bias toward converting. 0.0 is the neutral, untuned rule; sweeping it
        #: traces the whole policy family from always-wait to always-convert.
        self.margin = margin

    def value_of_converting(self, pool: np.ndarray, turn: int) -> float:
        """Expected score if we return the top ten now, under a uniform belief."""
        size = len(pool)
        if size == 0:
            return 0.0
        order = pool[np.argsort(-self.popularity[pool], kind="stable")]
        hit = min(TOP_K, size) / size
        mrr = sum(1.0 / rank for rank in range(1, min(TOP_K, size) + 1)) / size
        return W_HIT * hit + W_MRR * mrr + W_EFF * (11.0 - turn) / 10.0

    def value_of_waiting(self, pool: np.ndarray, turn: int, disclosed: set[str]) -> float:
        """Expected score if we ask once more, then convert on the next turn.

        Set intersection rather than ``np.intersect1d``: postings lists have a
        median length of 1, so the numpy call is dominated by its own overhead.
        """
        if turn >= MAX_TURNS:
            return 0.0
        popularity = self.popularity
        members = [int(x) for x in pool]
        pool_set = set(members)
        # Popularity ranking of the untouched pool, computed once rather than per candidate.
        ordered = sorted(members, key=lambda i: (-popularity[i], i))
        rank_in_pool = {position: rank for rank, position in enumerate(ordered, start=1)}

        total = 0.0
        for candidate in members:
            remaining = [c for c in self.index.constraints_of[candidate] if c not in disclosed]
            narrowed: set[int] | None = None
            for constraint in remaining[:2]:
                postings = self.index.constraint_sets.get(constraint)
                if postings is None:
                    continue
                base = pool_set if narrowed is None else narrowed
                # Set intersection walks the smaller operand; the largest postings
                # list has 13,633 entries while the pool is capped at LOOKAHEAD_CAP.
                step = base & postings
                if step:
                    narrowed = step
            if narrowed is None:
                rank = rank_in_pool[candidate]
            else:
                key = (-popularity[candidate], candidate)
                rank = 1 + sum(1 for other in narrowed if (-popularity[other], other) < key)
            total += turn_score(rank if rank <= TOP_K else None, turn + 1)
        return total / len(members)

    def decide(self, pool: np.ndarray, turn: int, exhausted: bool,
               disclosed: set[str] | None = None) -> Decision:
        size = len(pool)
        if size == 0:
            return Decision(True, size, 0.0, 0.0)
        now = self.value_of_converting(pool, turn)
        # Nothing left to learn, or too wide to reason about cheaply. When the
        # card is spent, waiting cannot narrow anything and only burns turns.
        if exhausted or turn >= MAX_TURNS - 2:
            return Decision(True, size, now, 0.0)
        if size > LOOKAHEAD_CAP:
            return Decision(False, size, now, now + 1.0)
        wait = self.value_of_waiting(pool, turn, disclosed or set())
        return Decision(now + self.margin >= wait, size, now, wait)


def build_policy(config, index, popularity: np.ndarray):
    """Select the policy an ``AgentConfig`` asks for."""
    if config.policy == "expected_value":
        return ExpectedValuePolicy(index, popularity, margin=config.convert_margin)
    return ThresholdPolicy(config.convert_threshold)
