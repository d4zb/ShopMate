"""Equivalence test for the copied simulator model.

``src/simulator_model.py`` duplicates six functions from the official evaluator
because importing them would create a cycle. That duplication is only safe if it
is continuously verified, so this compares both implementations across all 50,000
catalog products. If the organisers' released final package changes any of these
definitions, this test fails rather than the agent silently mis-parsing every
message and scoring zero.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluator import local_evaluator as official  # noqa: E402
from src import simulator_model as ours  # noqa: E402

CATALOG = ROOT / "data" / "catalog.jsonl"


@pytest.fixture(scope="module")
def products() -> list[dict]:
    if not CATALOG.exists():
        pytest.skip("data/catalog.jsonl missing; run scripts/fetch_data.py")
    return official.load_jsonl(CATALOG)


def test_constants_match() -> None:
    assert ours.ALLOWED_ATTRIBUTES == official.ALLOWED_ATTRIBUTES
    assert ours.MATERIALS == official.MATERIALS
    assert ours.SEARCH_FIELDS == official.SEARCH_FIELDS
    assert ours.MATERIAL_RE.pattern == official.MATERIAL_RE.pattern
    assert ours.COLOR_RE.pattern == official.COLOR_RE.pattern


def test_intent_card_matches_across_whole_catalog(products) -> None:
    assert len(products) == 50_000
    for product in products:
        assert ours.intent_card(product) == official.intent_card(product), product["parent_asin"]


def test_coarse_category_and_searchable_text_match(products) -> None:
    for product in products:
        values = [str(v) for v in product.get("categories") or []]
        assert ours.coarse_category(values) == official.coarse_category(values)
        assert ours.searchable_text(product) == official.searchable_text(product)


def test_classify_constraint_matches_on_every_disclosable_constraint(products) -> None:
    seen = 0
    for product in products:
        for constraint in ours.card_constraints(product):
            assert ours.classify_constraint(constraint) == official.classify_constraint(constraint)
            seen += 1
    assert seen > 100_000, "expected to cover most of the catalog's constraints"


def test_templates_reproduce_the_evaluators_own_messages(products) -> None:
    """The recogniser prefixes must actually match strings initial_message emits."""
    product = products[0]
    card = ours.intent_card(product)
    category = ours.coarse_category([str(v) for v in product.get("categories") or []])
    sample = {"scenario_type": "buying", "intent_card": card}
    buying = official.initial_message(sample, category, set())
    assert buying.startswith(ours.OPENER_PREFIX)
    assert ours.BUYING_MARKER in buying

    browsing = official.initial_message({"scenario_type": "browsing", "intent_card": card}, category, set())
    assert browsing.endswith(ours.BROWSING_SUFFIX)

    disclosure, _ = official.customer_reply(
        {"scenario_type": "buying", "intent_card": card}, "other", set(), False
    )
    assert disclosure.startswith(ours.DISCLOSE_PREFIX)

    spent, _ = official.customer_reply(
        {"scenario_type": "buying", "intent_card": {"hard_constraints": [], "soft_preferences": []}},
        "other", set(), False,
    )
    assert ours.carries_no_information(spent)

    boundary, used = official.customer_reply(
        {"scenario_type": "boundary", "intent_card": card}, "other", set(), False
    )
    assert used is True and ours.carries_no_information(boundary)

    nudge, _ = official.customer_reply({"scenario_type": "buying", "intent_card": card}, None, set(), False)
    assert nudge == ours.NUDGE
