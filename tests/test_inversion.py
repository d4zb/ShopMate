"""Inversion and policy safety tests.

Two failure modes here are catastrophic rather than merely costly, and both get
a dedicated test:

* **An emptied candidate pool.** Constraints are always true attributes of the
  target, so an empty intersection can only mean a parse error. Applying it
  anyway discards the answer permanently.
* **Suppression that never lifts.** The agent withholds recommendations while
  uncertain. If it fails to commit before the turn limit it scores zero for that
  session -- strictly worse than any recommendation it could have made.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import Agent, AgentConfig  # noqa: E402
from evaluator import local_evaluator as official  # noqa: E402
from src import simulator_model as sim  # noqa: E402

CATALOG = ROOT / "data" / "catalog.jsonl"


@pytest.fixture(scope="module")
def agent() -> Agent:
    if not CATALOG.exists():
        pytest.skip("data/catalog.jsonl missing; run scripts/fetch_data.py")
    return Agent(str(CATALOG), config=AgentConfig())


def _opener(agent: Agent, product: dict, scenario: str) -> str:
    card = sim.intent_card(product)
    category = sim.coarse_category([str(v) for v in product.get("categories") or []])
    sample = {"scenario_type": scenario, "intent_card": card,
              "behavior": {"override": {"old_value": card["soft_preferences"][-1]}}}
    return official.initial_message(sample, category, set())


def test_opener_parsing_round_trips_for_every_scenario(agent: Agent) -> None:
    for product in agent.catalog.products[:300]:
        expected = sim.coarse_category([str(v) for v in product.get("categories") or []])
        for scenario in ("buying", "browsing", "intent_override"):
            category, _ = agent.index.parse_opener(_opener(agent, product, scenario))
            assert category == expected, (product["parent_asin"], scenario)


def test_target_always_survives_its_own_disclosures(agent: Agent) -> None:
    """The whole approach rests on this: disclosing the truth never evicts the truth."""
    for position, product in enumerate(agent.catalog.products[:500]):
        category = sim.coarse_category([str(v) for v in product.get("categories") or []])
        constraints = sim.card_constraints(product)
        pool = agent.index.pool(category, constraints)
        assert position in set(int(x) for x in pool), product["parent_asin"]


def test_contradictory_constraints_never_empty_the_pool(agent: Agent) -> None:
    """A synthetic session that contradicts itself must degrade, not collapse."""
    a = sim.card_constraints(agent.catalog.products[0])
    b = sim.card_constraints(agent.catalog.products[1])
    category = sim.coarse_category([str(v) for v in agent.catalog.products[0].get("categories") or []])

    pool = agent.index.pool(category, a + b)
    assert len(pool) > 0, "conflicting constraints emptied the pool"

    # Also with a category that cannot co-occur with those constraints at all.
    other = sim.coarse_category([str(v) for v in agent.catalog.products[1].get("categories") or []])
    assert len(agent.index.pool(other, a + b)) > 0
    # And with pure nonsense, which must fall through to the unrestricted catalog.
    assert len(agent.index.pool("no such category", ["no such constraint"])) == len(agent.catalog)


def test_delimiter_inside_a_constraint_is_not_split(agent: Agent) -> None:
    """5.76% of constraint strings contain the '; ' the evaluator joins with."""
    tricky = next((c for c in agent.index.by_constraint if "; " in c), None)
    assert tricky is not None, "expected at least one constraint containing '; '"
    message = sim.DISCLOSE_PREFIX + tricky + "."
    assert agent.index.parse_reply(message) == [tricky]


def test_constraint_ending_in_a_period_survives_the_template_terminator(agent: Agent) -> None:
    """Regression: only the template's own trailing period may be stripped.

    ``_clean_constraint`` strips punctuation before truncating to 180 chars, so 96
    catalog constraints end in a period. ``rstrip(".")`` removed both theirs and
    the template's, producing a string that matched nothing.
    """
    tricky = next((c for c in agent.index.by_constraint if c.endswith(".")), None)
    assert tricky is not None, "expected at least one constraint ending in '.'"
    assert agent.index.parse_reply(sim.DISCLOSE_PREFIX + tricky + ".") == [tricky]
    assert agent.index.parse_reply(sim.OVERRIDE_PREFIX + tricky + ".") == [tricky]


def test_every_disclosure_message_round_trips(agent: Agent) -> None:
    """Generate real messages via the evaluator and parse each one back exactly."""
    checked = 0
    for product in agent.catalog.products[:400]:
        card = sim.intent_card(product)
        sample = {"scenario_type": "buying", "intent_card": card}
        disclosed: set[str] = set()
        for _ in range(3):
            reply, _ = official.customer_reply(sample, "other", disclosed, False)
            if sim.carries_no_information(reply):
                break
            expected = [c for c in sim.card_constraints(product) if c in disclosed]
            recovered = agent.index.parse_reply(reply)
            assert set(recovered) <= set(expected), (product["parent_asin"], reply[:80])
            assert recovered, f"failed to recover anything from {reply[:80]!r}"
            checked += 1
    assert checked > 300


PERTURBATIONS = {
    "polite prefix": lambda m: "Hi! " + m,
    "trailing chatter": lambda m: m + " Thanks so much!",
    "lowercase": lambda m: m.lower(),
    "double spaces": lambda m: m.replace(" ", "  "),
    "no trailing period": lambda m: m[:-1] if m.endswith(".") else m,
}


@pytest.mark.parametrize("name", sorted(PERTURBATIONS))
def test_perturbed_openers_still_recover_the_category(agent: Agent, name: str) -> None:
    """Anchored matching meant one prepended word cost 0.63 TechnicalScore."""
    perturb = PERTURBATIONS[name]
    for product in agent.catalog.products[:120]:
        expected = sim.coarse_category([str(v) for v in product.get("categories") or []])
        for scenario in ("buying", "browsing", "intent_override"):
            clean = _opener(agent, product, scenario)
            recovered, _ = agent.index.parse_opener(perturb(clean))
            assert recovered == expected, (name, scenario, product["parent_asin"])


def test_browsing_opener_reports_no_constraint_span(agent: Agent) -> None:
    """A browsing opener discloses nothing; fuzzy matching it invents constraints.

    Regression: letting tier 3 run on browsing openers dropped that scenario from
    0.9597 to 0.8100 because it fabricated constraints and evicted the target.
    """
    for product in agent.catalog.products[:60]:
        opener = _opener(agent, product, "browsing")
        assert agent.index.constraint_span(opener, is_opener=True) is None
        assert agent.index.constraint_span("Hi! " + opener.lower(), is_opener=True) is None


def test_buying_opener_exposes_its_constraint_span(agent: Agent) -> None:
    for product in agent.catalog.products[:60]:
        card = sim.intent_card(product)
        if not card["hard_constraints"]:
            continue
        span = agent.index.constraint_span(_opener(agent, product, "buying"), is_opener=True)
        assert span == card["hard_constraints"][0], product["parent_asin"]


def test_fuzzy_match_returns_at_most_one_constraint(agent: Agent) -> None:
    """Two fuzzy guesses measured worse than one: the second is usually invented,
    and unlike an unmatched string it does intersect the pool and can evict the target."""
    import numpy as np

    product = agent.catalog.products[0]
    category = sim.coarse_category([str(v) for v in product.get("categories") or []])
    pool = agent.index.pool(category, [])
    constraints = sim.card_constraints(product)
    matched = agent.index.fuzzy_match("; ".join(constraints) + " and some extra words", pool)
    assert len(matched) <= 1
    if matched:
        assert matched[0] in agent.index.by_constraint


def test_fuzzy_match_declines_when_nothing_is_close(agent: Agent) -> None:
    import numpy as np

    pool = np.arange(200, dtype=np.int32)
    assert agent.index.fuzzy_match("zzz qqq xyzzy plugh", pool) == []
    assert agent.index.fuzzy_match("", pool) == []


def test_suppression_always_lifts_before_the_turn_limit(agent: Agent) -> None:
    """Withholding forever scores zero, so the policy must commit while it can."""
    session = "safety-check"
    agent.reset(session, {})
    product = agent.catalog.products[0]
    message = _opener(agent, product, "browsing")
    recommended_by = None
    for turn in range(1, official.MAX_TURNS + 1):
        response = agent.respond(session, message, turn, 10)
        if response["recommendations"]:
            recommended_by = turn
            break
        # The shopper stops disclosing; the agent must notice and commit.
        message = sim.NO_PREFERENCE_PREFIX + "other."
    assert recommended_by is not None, "agent never recommended anything"
    assert recommended_by <= official.MAX_TURNS - 2


def test_response_is_contract_valid_even_without_reset(agent: Agent) -> None:
    """Exceptions are swallowed by the evaluator and scored as a miss, so degrade instead."""
    response = agent.respond("never-reset", "I'm looking for Shoes, but I'm still exploring.", 1, 10)
    assert isinstance(response["message"], str)
    assert response["ask_attribute"] in sim.ALLOWED_ATTRIBUTES or response["ask_attribute"] is None
    for item in response["recommendations"]:
        assert item["parent_asin"] in agent.catalog.index_of


def test_sessions_do_not_leak_into_each_other(agent: Agent) -> None:
    first, second = "session-a", "session-b"
    agent.reset(first, {})
    agent.reset(second, {})
    agent.respond(first, _opener(agent, agent.catalog.products[10], "buying"), 1, 10)
    agent.respond(second, _opener(agent, agent.catalog.products[20], "buying"), 1, 10)
    assert agent._sessions[first].category != agent._sessions[second].category or \
        agent._sessions[first].known_constraints != agent._sessions[second].known_constraints
    assert len(agent._sessions[second].utterances) == 1
