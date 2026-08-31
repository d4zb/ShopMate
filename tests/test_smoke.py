"""End-to-end smoke test: 5 real sessions through the unmodified evaluator.

The evaluator wraps ``respond`` in ``try/except Exception`` and silently
substitutes an empty response, so a crash shows up only as a slightly worse
score. This test therefore drives the agent through a strict validating proxy
that re-raises, and asserts the response schema on every single turn.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402

CATALOG = ROOT / "data" / "catalog.jsonl"
DATASET = ROOT / "data" / "public_set.jsonl"
SESSION_COUNT = 5

ALLOWED_ATTRIBUTES = {
    "category", "material", "color", "size", "style", "brand",
    "budget", "feature", "use_case", "other",
}


@pytest.fixture(scope="module")
def catalog():
    if not CATALOG.exists():
        pytest.skip("data/catalog.jsonl missing; run scripts/fetch_data.py")
    return catalog_index(CATALOG)


class StrictProxy:
    """Forwards to the real agent, validating the contract and re-raising failures."""

    def __init__(self, inner, catalog_ids: set[str]) -> None:
        self.inner = inner
        self.catalog_ids = catalog_ids
        self.turns = 0

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.inner.reset(session_id, user_profile)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        response = self.inner.respond(session_id, user_message, turn, top_k)
        self.turns += 1

        assert isinstance(response, dict), f"turn {turn}: response is {type(response)!r}, not dict"
        assert isinstance(response.get("message"), str), f"turn {turn}: 'message' must be a str"

        attribute = response.get("ask_attribute", "__missing__")
        assert attribute is None or attribute in ALLOWED_ATTRIBUTES, (
            f"turn {turn}: ask_attribute {attribute!r} is not an allowed value or None"
        )

        recommendations = response.get("recommendations")
        assert isinstance(recommendations, list), f"turn {turn}: 'recommendations' must be a list"
        seen: set[str] = set()
        for item in recommendations:
            assert isinstance(item, dict), f"turn {turn}: recommendation {item!r} is not a dict"
            asin = item.get("parent_asin")
            assert isinstance(asin, str) and asin, f"turn {turn}: bad parent_asin {asin!r}"
            assert asin in self.catalog_ids, f"turn {turn}: {asin} is not in the catalog"
            assert asin not in seen, f"turn {turn}: duplicate parent_asin {asin}"
            seen.add(asin)

        usage = response.get("usage")
        if usage is not None:
            assert isinstance(usage, dict)
            for key in ("prompt_tokens", "completion_tokens"):
                assert isinstance(usage[key], int) and usage[key] >= 0, f"turn {turn}: bad usage[{key}]"

        return response


def test_five_sessions_end_to_end(catalog) -> None:
    from agent import Agent

    catalog_ids, categories, products = catalog
    samples = load_jsonl(DATASET)[:SESSION_COUNT]

    proxy = StrictProxy(Agent(str(CATALOG)), catalog_ids)
    result = evaluate(proxy, samples, catalog_ids, categories, products)

    assert proxy.turns > 0, "agent was never called"
    assert result["sample_count"] == SESSION_COUNT
    assert 0.0 <= result["hit_rate_at_10"] <= 1.0
    assert 0.0 <= result["mrr"] <= 1.0
    assert 1.0 <= result["mttc"] <= 11.0
    assert len(result["sessions"]) == SESSION_COUNT
    for session in result["sessions"]:
        if session["hit"]:
            assert 1 <= session["best_rank"] <= 10
            assert 1 <= session["first_hit_turn"] <= 10
        else:
            assert session["best_rank"] is None and session["reciprocal_rank"] == 0.0


def test_recommendations_are_capped_at_ten(catalog) -> None:
    """Only the first 10 valid unique ASINs are scored; returning more is harmless but pointless."""
    from evaluator.local_evaluator import normalize_recommendations

    catalog_ids, _, _ = catalog
    sample_asins = list(catalog_ids)[:25]
    payload = [{"parent_asin": asin} for asin in sample_asins]
    assert normalize_recommendations(payload, catalog_ids) == sample_asins[:10]
