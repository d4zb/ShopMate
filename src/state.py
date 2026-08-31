"""Per-session conversation state.

One instance per ``session_id``. Indexes are shared and immutable; everything
mutable lives here, so sessions cannot leak into each other — the evaluator
requires isolation and reuses no state between sessions.

The retrieval query is built from the whole accumulated conversation rather than
the last utterance alone. The organiser's baseline uses only the last message,
which is a large part of why it scores 0.0251 on browsing sessions: the opener
carries the category and every later turn drops it.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SessionState:
    session_id: str
    user_profile: dict = field(default_factory=dict)
    #: Every customer utterance, oldest first.
    utterances: list[str] = field(default_factory=list)
    #: Constraint strings recovered from those utterances, in disclosure order.
    known_constraints: list[str] = field(default_factory=list)
    #: Spans that *should* have held a constraint but resolved to nothing. They
    #: cannot filter the pool, but they are still the most informative words the
    #: shopper said, so they get the same query weighting a resolved constraint does.
    soft_spans: list[str] = field(default_factory=list)
    #: The recovered ``coarse_category``, once the opener has been parsed.
    category: str | None = None
    #: Set once the customer signals there is nothing left to disclose.
    exhausted: bool = False
    turn: int = 0
    #: The most recent recommend-or-ask Decision, kept so scripts/demo.py and
    #: scripts/error_analysis.py can show the reasoning rather than just the outcome.
    last_decision: object = None

    def record(self, message: str, turn: int) -> None:
        self.utterances.append(message)
        self.turn = turn

    def add_constraint(self, constraint: str) -> None:
        """Accumulate. Constraints never contradict — every disclosed value is a
        true attribute of the target, including the intent-override 'new_value'."""
        if constraint and constraint not in self.known_constraints:
            self.known_constraints.append(constraint)

    def query(self) -> str:
        """Retrieval query over the full conversation, weighting what we parsed.

        Recovered constraints and the category are repeated so they outweigh the
        template boilerplate that surrounds them in the raw utterances.
        """
        parts = list(self.utterances)
        if self.category:
            parts.append(self.category)
        parts.extend(self.known_constraints)
        parts.extend(self.soft_spans)
        return " ".join(parts)
