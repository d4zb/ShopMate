"""Posterior inference over the catalog from the shopper's utterances.

``simulator_model`` gives us the shopper's generative model: which strings a
given target product would cause them to say. This module runs that backwards.
Every utterance is a verbatim substring of the target's own catalog metadata, so
recovering it identifies a set of products that could have produced it, and the
target is guaranteed to be in that set.

The index is sharp. Across the 50,000-product catalog there are 60,670 distinct
constraint strings with a **median postings list of one** -- most constraints
identify a single product outright. ``coarse_category`` has 1,115 values with a
median bucket of 8.

Two design rules, both learned from the data:

*Recover constraints by longest-match against the index, never by splitting on
the delimiter.* The evaluator joins disclosed constraints with ``"; "``, but
5.76% of constraint strings contain ``"; "`` themselves and 12% contain a period,
so naive splitting silently corrupts those sessions. Where a message is
ambiguous, the current candidate pool disambiguates it: the true constraints must
be satisfied by the target, and the target is always still in the pool.

*A constraint may never empty the pool.* Constraints are always true attributes
of the target, so an empty intersection means we mis-parsed. Skipping the
offending constraint degrades to a wider pool; applying it would discard the
answer permanently.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np

from .catalog import Catalog
from .simulator_model import (
    BOUNDARY_PREFIX,
    BROWSING_SUFFIX,
    BUYING_MARKER,
    DISCLOSE_PREFIX,
    NO_PREFERENCE_PREFIX,
    NUDGE,
    OPENER_PREFIX,
    OVERRIDE_PREFIX,
    card_constraints,
    coarse_category,
)


_TOKEN_RE = re.compile(r"[a-z0-9]+")

#: Tier-3 acceptance threshold: the fraction of a known constraint's tokens that
#: must appear in the observed text. 0.7 was swept on the dev split; below ~0.5 it
#: starts accepting unrelated constraints, above ~0.85 it stops rescuing anything.
FUZZY_THRESHOLD = 0.7


def _normalize(text: str) -> str:
    """Case- and whitespace-insensitive key for tier-2 lookup."""
    return re.sub(r"\s+", " ", text).strip().casefold()


#: Template markers, raw and normalized. The raw pass runs first so an untouched
#: transcript takes exactly the path it always did.
_MARKERS = {
    "opener": OPENER_PREFIX, "browsing": BROWSING_SUFFIX, "buying": BUYING_MARKER,
    "disclose": DISCLOSE_PREFIX, "override": OVERRIDE_PREFIX,
}
_NORM_MARKERS = {key: _normalize(value) for key, value in _MARKERS.items()}


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.casefold())


def _after(message: str, marker: str) -> str | None:
    """Text following the first occurrence of ``marker``, or None if absent.

    Located rather than anchored, so leading or trailing chatter around the
    template cannot prevent recognition. Falls back to a normalized scan so a
    case or whitespace change does not lose the marker either.
    """
    cut = message.find(marker)
    if cut != -1:
        return message[cut + len(marker):]
    normalized = _normalize(message)
    cut = normalized.find(_normalize(marker))
    if cut == -1:
        return None
    # Map the normalized offset back by consuming the same number of tokens.
    consumed = len(_normalize(marker).split())
    return " ".join(re.sub(r"\s+", " ", message).strip().split()[
        len(normalized[:cut].split()) + consumed:
    ])


def _strip_terminator(text: str) -> str:
    """Remove the single period the template appends -- and only that one.

    ``_clean_constraint`` strips trailing punctuation *before* truncating to 180
    characters, so a truncated constraint can legitimately end in a period: 96 of
    them do. ``rstrip(".")`` would eat the constraint's own period along with the
    template's and the result would match nothing.
    """
    return text[:-1] if text.endswith(".") else text


class InversionIndex:
    """Inverted indexes from simulator-visible strings back to catalog positions."""

    #: Bumped whenever the derived tables change shape, so a stale cache from an
    #: older build is rejected rather than silently loaded.
    CACHE_VERSION = 1

    @classmethod
    def build(cls, catalog: Catalog, cache: Path | None = None) -> "InversionIndex":
        """Load the cached tables if they match this catalog, else build and cache them.

        Building costs ~33 s because ``intent_card`` runs over all 50,000
        products. FAQ section 4 explicitly permits precomputed local sidecar
        files, and the tables are a pure function of the frozen catalog.
        """
        if cache is not None and cache.exists():
            try:
                with cache.open("rb") as handle:
                    payload = pickle.load(handle)
                if (payload.get("version") == cls.CACHE_VERSION
                        and payload.get("n_products") == len(catalog)
                        and payload.get("first_asin") == catalog.asins[0]
                        and payload.get("last_asin") == catalog.asins[-1]):
                    return cls(catalog, _payload=payload)
            except (OSError, pickle.UnpicklingError, KeyError, AttributeError):
                pass  # a corrupt or stale cache must never be fatal
        index = cls(catalog)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            with cache.open("wb") as handle:
                pickle.dump(index._as_payload(), handle, protocol=pickle.HIGHEST_PROTOCOL)
        return index

    def _as_payload(self) -> dict:
        return {
            "version": self.CACHE_VERSION,
            "n_products": len(self.catalog),
            "first_asin": self.catalog.asins[0],
            "last_asin": self.catalog.asins[-1],
            "by_category": self.by_category,
            "by_constraint": self.by_constraint,
            "constraints_of": self.constraints_of,
        }

    def __init__(self, catalog: Catalog, _payload: dict | None = None) -> None:
        self.catalog = catalog
        if _payload is not None:
            self.by_category = _payload["by_category"]
            self.by_constraint = _payload["by_constraint"]
            self.constraints_of = _payload["constraints_of"]
            self.all_positions = np.arange(len(catalog), dtype=np.int32)
            self._build_posting_sets()
            return
        by_category: dict[str, list[int]] = {}
        by_constraint: dict[str, list[int]] = {}
        self.constraints_of: list[list[str]] = []

        for position, product in enumerate(catalog.products):
            category = coarse_category([str(v) for v in product.get("categories") or []])
            by_category.setdefault(category, []).append(position)
            constraints = card_constraints(product)
            self.constraints_of.append(constraints)
            for constraint in constraints:
                by_constraint.setdefault(constraint, []).append(position)

        self.by_category = {k: np.array(v, dtype=np.int32) for k, v in by_category.items()}
        self.by_constraint = {k: np.array(v, dtype=np.int32) for k, v in by_constraint.items()}
        self.all_positions = np.arange(len(catalog), dtype=np.int32)
        self._build_posting_sets()

    def _build_posting_sets(self) -> None:
        """Set mirrors of the postings, for the policy's one-step lookahead.

        Postings are usually tiny (median 1) but the tail is long: the most common
        constraint string appears on 13,633 products. Python-side intersection has
        to iterate the smaller operand, which set-vs-set does automatically and
        array scanning does not. Total membership across all postings is only
        ~197k, so this costs a fraction of a second and a few MB.
        """
        self.constraint_sets = {
            constraint: frozenset(int(p) for p in postings)
            for constraint, postings in self.by_constraint.items()
        }
        # Tier 2 of resolution: a case- and whitespace-insensitive view of the same
        # keys. First insertion wins on collision so the mapping is deterministic.
        self.by_constraint_norm: dict[str, str] = {}
        for constraint in self.by_constraint:
            self.by_constraint_norm.setdefault(_normalize(constraint), constraint)
        self.by_category_norm: dict[str, str] = {}
        for category in self.by_category:
            self.by_category_norm.setdefault(_normalize(category), category)
        # Tier 3: token sets for fuzzy containment scoring.
        self.constraint_tokens = {
            constraint: frozenset(_tokens(constraint)) for constraint in self.by_constraint
        }
        # (key, canonical) pairs for the raw and normalized opener passes. Longest
        # first so a category that is a prefix of another cannot shadow it.
        self._categories_by_length = [
            (c, c) for c in sorted(self.by_category, key=len, reverse=True)
        ]
        self._norm_categories_by_length = sorted(
            ((_normalize(c), c) for c in self.by_category),
            key=lambda pair: len(pair[0]), reverse=True,
        )

    # --- parsing ------------------------------------------------------------

    def parse_opener(self, message: str) -> tuple[str | None, list[str]]:
        """Recover ``(category, constraints)`` from the shopper's first message.

        Two passes. The first works on the raw text with the literal markers, which
        is what the untouched evaluator emits and keeps the exact path bit-for-bit.
        If that fails to identify a category, the second retries the whole parse in
        normalized space (casefolded, whitespace collapsed) against normalized
        markers, which is what rescues a lowercased or re-spaced transcript.

        Markers are located positionally rather than anchored to the start of the
        string: anchoring with ``startswith`` meant one prepended word ("Hi!")
        broke recognition outright and cost 0.63 TechnicalScore.
        """
        for text, markers, categories in (
            (message, _MARKERS, self._categories_by_length),
            (_normalize(message), _NORM_MARKERS, self._norm_categories_by_length),
        ):
            category, constraints = self._parse_opener_pass(text, markers, categories)
            if category is not None:
                return category, constraints
        return None, []

    def _parse_opener_pass(self, text: str, markers: dict[str, str],
                           categories: list[tuple[str, str]]) -> tuple[str | None, list[str]]:
        cut = text.find(markers["opener"])
        if cut == -1:
            return None, []
        body = text[cut + len(markers["opener"]):]

        browsing = body.find(markers["browsing"])
        if browsing != -1:
            return self._resolve_category(body[:browsing]), []

        buying = body.find(markers["buying"])
        if buying != -1:
            constraint = body[buying + len(markers["buying"]):]
            return (self._resolve_category(body[:buying]),
                    self._resolve_constraints(_strip_terminator(constraint)))

        # Intent-override opener: "I'm looking for {category}. {old_value}".
        # The boundary between the two is only recoverable by longest match.
        for key, canonical in categories:
            marker = key + ". "
            cut = body.find(marker)
            if cut != -1:
                return canonical, self._resolve_constraints(
                    _strip_terminator(body[cut + len(marker):])
                )
        # Nothing recognisable after the category: fall back to prefix matching.
        return self._resolve_category(body, allow_prefix=True), []

    def constraint_span(self, message: str, is_opener: bool) -> str | None:
        """The slice of a message that is supposed to contain constraints.

        ``None`` means this message discloses nothing by design -- a browsing
        opener, or a reply with no recognisable disclosure marker. That distinction
        matters: tier-3 fuzzy matching must never run on a message that genuinely
        carries no constraints, or it invents them out of the pool and poisons it.
        """
        for text, markers, categories in (
            (message, _MARKERS, self._categories_by_length),
            (_normalize(message), _NORM_MARKERS, self._norm_categories_by_length),
        ):
            if is_opener:
                cut = text.find(markers["opener"])
                if cut == -1:
                    continue
                body = text[cut + len(markers["opener"]):]
                if markers["browsing"] in body:
                    return None          # browsing openers disclose nothing by design
                buying = body.find(markers["buying"])
                if buying != -1:
                    return _strip_terminator(body[buying + len(markers["buying"]):])
                for key, _canonical in categories:
                    marker = key + ". "
                    at = body.find(marker)
                    if at != -1:
                        return _strip_terminator(body[at + len(marker):])
                continue
            for name in ("disclose", "override"):
                at = text.find(markers[name])
                if at != -1:
                    return _strip_terminator(text[at + len(markers[name]):])
        return None

    def parse_reply(self, message: str) -> list[str]:
        """Recover constraint strings from a follow-up message. Raw pass, then normalized."""
        for text, markers in ((message, _MARKERS), (_normalize(message), _NORM_MARKERS)):
            for name in ("disclose", "override"):
                # The override's new_value is hard_constraints[0]: a true attribute
                # of the target, not a contradiction of anything said earlier.
                at = text.find(markers[name])
                if at != -1:
                    recovered = self._resolve_constraints(
                        _strip_terminator(text[at + len(markers[name]):])
                    )
                    if recovered:
                        return recovered
        return []

    def carries_no_information(self, message: str) -> bool:
        normalized = _normalize(message)
        return any(
            _normalize(marker) in normalized
            for marker in (NO_PREFERENCE_PREFIX, BOUNDARY_PREFIX, NUDGE)
        )

    def _resolve_category(self, text: str, allow_prefix: bool = False) -> str | None:
        """Tier 1 exact, tier 2 normalized, and optionally tier 3 longest prefix.

        The opener template is ``"I'm looking for {category}..."``, so whatever
        follows the category is trailing text rather than part of it. When the
        surrounding template is damaged -- truncated, or re-worded so the suffix is
        unrecognisable -- the category is still the *start* of the remainder, and
        matching the longest known category prefix recovers it.
        """
        text = text.strip()
        if text in self.by_category:
            return text
        found = self.by_category_norm.get(_normalize(text))
        if found is not None:
            return found
        if allow_prefix:
            normalized = _normalize(text)
            # Sorted longest-first, so the most specific category wins.
            for key, canonical in self._norm_categories_by_length:
                if normalized.startswith(key):
                    return canonical
        return None

    def _resolve_constraints(self, body: str) -> list[str]:
        """Split a disclosure body into known constraints without trusting the delimiter.

        The evaluator emits at most two constraints joined by ``"; "``, but a
        constraint may itself contain ``"; "``. Every split point that yields two
        known constraints is a valid reading; so is the whole body as one. The
        caller disambiguates against the live pool.
        """
        body = body.strip()
        if not body:
            return []
        readings = self._readings(body)
        # Deliberately empty rather than [body] when nothing resolves. Returning the
        # unmatched text as if it were a constraint is inert (it matches no postings)
        # but it masks the failure from the caller and so blocks the tier-3 fuzzy
        # fallback. The raw text is already in the retrieval query via
        # SessionState.utterances, so nothing is lost by admitting the miss.
        return readings[0] if readings else []

    def _lookup_constraint(self, text: str) -> str | None:
        """Tier 1 exact, then tier 2 normalized. Returns the canonical key."""
        text = text.strip()
        if text in self.by_constraint:
            return text
        return self.by_constraint_norm.get(_normalize(text))

    def fuzzy_match(self, text: str, pool: np.ndarray,
                    threshold: float = FUZZY_THRESHOLD) -> list[str]:
        """Tier 3: recover constraints from paraphrased or truncated text.

        Scored by token containment -- what fraction of a known constraint's tokens
        appear in the observed text -- and searched only over the cards of products
        **in the current pool**. That restriction is what makes this both cheap and
        safe: the only constraints it can propose are ones some surviving candidate
        genuinely has, so a wrong guess widens the pool rather than inventing an
        attribute out of the whole catalog.

        Returns **one** match, not the two the evaluator may have disclosed. Swept
        on dev: at a cap of 1 the "last word dropped" row scores 0.695 and
        "trailing chatter" 0.799; at a cap of 2 they fall to 0.679 and 0.791, and
        as low as 0.641/0.662 at a loose threshold. A second fuzzy guess is usually
        a constraint the shopper never said, and unlike an unmatched string it
        *does* intersect the pool -- so it narrows it wrongly and can evict the
        target. One confident guess is worth more than two hopeful ones.

        The threshold itself barely matters: 0.5 through 0.9 land within 0.002 of
        each other at a cap of 1, so it is not a tuned parameter.
        """
        observed = set(_tokens(text))
        if not observed:
            return []
        scored: list[tuple[float, int, str]] = []
        seen: set[str] = set()
        for position in pool:
            for constraint in self.constraints_of[int(position)]:
                if constraint in seen:
                    continue
                seen.add(constraint)
                tokens = self.constraint_tokens.get(constraint)
                if not tokens:
                    continue
                score = len(observed & tokens) / len(tokens)
                if score >= threshold:
                    # Longer constraints break ties: they explain more of the text.
                    scored.append((score, len(tokens), constraint))
        if not scored:
            return []
        scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
        return [scored[0][2]]

    def _readings(self, body: str) -> list[list[str]]:
        readings: list[list[str]] = []
        whole = self._lookup_constraint(body)
        if whole is not None:
            readings.append([whole])
        # The evaluator joins with "; ", but a constraint may contain that too, and
        # a perturbed transcript may use a different separator entirely.
        for separator in ("; ", ", ", " | ", "  "):
            start = 0
            while True:
                cut = body.find(separator, start)
                if cut == -1:
                    break
                left = self._lookup_constraint(body[:cut])
                right = self._lookup_constraint(body[cut + len(separator):])
                if left is not None and right is not None:
                    readings.append([left, right])
                start = cut + 1
            start = cut + 1
        # Two known constraints explain more of the message than one long guess.
        readings.sort(key=lambda r: -len(r))
        return readings

    def readings(self, message: str) -> list[list[str]]:
        """All valid interpretations of a disclosure message, most specific first."""
        for text, markers in ((message, _MARKERS), (_normalize(message), _NORM_MARKERS)):
            for name in ("disclose", "override"):
                at = text.find(markers[name])
                if at != -1:
                    found = self._readings(
                        _strip_terminator(text[at + len(markers[name]):])
                    )
                    if found:
                        return found
        return []

    # --- posterior ----------------------------------------------------------

    def pool(self, category: str | None, constraints: list[str]) -> np.ndarray:
        """Candidate positions consistent with everything disclosed so far.

        Constraints are applied in disclosure order and any that would empty the
        pool is skipped: an empty intersection means a parse error, and discarding
        the target is unrecoverable while an over-wide pool is not.
        """
        pool = self.by_category.get(category) if category else None
        if pool is None:
            pool = self.all_positions
        for constraint in constraints:
            postings = self.by_constraint.get(constraint)
            if postings is None:
                continue
            narrowed = np.intersect1d(pool, postings, assume_unique=False)
            if len(narrowed):
                pool = narrowed
        return pool
