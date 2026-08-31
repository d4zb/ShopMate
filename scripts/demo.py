"""Walk through one complete multi-turn session, showing the agent's reasoning.

    python scripts/demo.py                     # an intent-override session
    python scripts/demo.py --scenario browsing
    python scripts/demo.py --sample-id public_0001

Required deliverable: ``docs/competition_specification.md`` asks for "One
demonstrated multi-turn session" and the FAQ repeats it.

The session is driven by the **unmodified** official evaluator, so what is printed
is a real scored session, not a mock-up. For each turn it shows the shopper's
message, what the agent recovered from it, how far the candidate pool collapsed,
and -- the part that matters -- the expected-value comparison behind the decision
to recommend or to ask one more question.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import evaluator.local_evaluator as ev  # noqa: E402
from agent import Agent, AgentConfig  # noqa: E402

RULE = "-" * 78


def wrap(text: str, indent: str = "    ", width: int = 74) -> str:
    words, lines, current = text.split(), [], ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return "\n".join(indent + line for line in lines)


class Narrator:
    """Prints the agent's internal state as the evaluator drives the session."""

    def __init__(self, inner: Agent, target: str) -> None:
        self.inner = inner
        self.target = target
        self.session_id = ""

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.session_id = session_id
        self.inner.reset(session_id, user_profile)
        print(f"\n{RULE}\nprofile: {user_profile.get('summary', '(none)')}\n{RULE}")

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        print(f"\nTURN {turn}")
        print("  shopper:")
        print(wrap(user_message, indent="    | "))

        before = self.inner._sessions[session_id]
        previous = len(self.inner._pool(before)) if self.inner.index else None

        response = self.inner.respond(session_id, user_message, turn, top_k)
        state = self.inner._sessions[session_id]
        pool = self.inner._pool(state)

        print(f"  parsed  : category={state.category!r}")
        for constraint in state.known_constraints:
            print(f"            + {constraint[:66]}")
        if pool is not None:
            print(f"  pool    : {previous:,} -> {len(pool):,} candidates")

        decision = state.last_decision
        if decision is not None and decision.value_wait:
            verdict = "RECOMMEND" if decision.convert else "ASK, stay silent"
            print(f"  decision: E[now]={decision.value_now:.4f}  "
                  f"E[wait]={decision.value_wait:.4f}  -> {verdict}")
        elif decision is not None:
            print(f"  decision: E[now]={decision.value_now:.4f}  "
                  f"(card spent, must commit) -> RECOMMEND")

        print(f"  agent   :")
        print(wrap(response["message"], indent="    | "))
        recommendations = response["recommendations"]
        if not recommendations:
            print("            (no recommendations yet -- not confident enough to show a list)")
        else:
            for rank, item in enumerate(recommendations[:5], start=1):
                asin = item["parent_asin"]
                title = self.inner.catalog.products[self.inner.catalog.index_of[asin]]["title"]
                marker = "  <== TARGET" if asin == self.target else ""
                print(f"            {rank}. {asin}  {title[:44]}{marker}")
            if len(recommendations) > 5:
                print(f"            ... {len(recommendations) - 5} more")
        return response


def main() -> int:
    parser = argparse.ArgumentParser(description="Narrate one full multi-turn session")
    parser.add_argument("--scenario", choices=("buying", "browsing", "intent_override", "boundary"),
                        default="intent_override")
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--catalog", type=Path, default=ROOT / "data" / "catalog.jsonl")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "public_set.jsonl")
    args = parser.parse_args()

    samples = ev.load_jsonl(args.dataset)
    if args.sample_id:
        chosen = [s for s in samples if s["sample_id"] == args.sample_id]
    else:
        chosen = [s for s in samples if s["scenario_type"] == args.scenario][:1]
    if not chosen:
        print("no matching session")
        return 1
    sample = chosen[0]

    catalog_ids, categories, products = ev.catalog_index(args.catalog)
    target = str(sample["ground_truth"]["parent_asin"])

    print(f"session  : {sample['sample_id']}  ({sample['scenario_type']}, "
          f"{sample.get('difficulty_bucket')})")
    print(f"target   : {target}  {products[target]['title'][:52]}")
    if sample["scenario_type"] == "intent_override":
        print("note     : the evaluator discards any hit before the override is revealed,")
        print("           so this scenario cannot convert earlier than turn 3.")

    narrator = Narrator(Agent(str(args.catalog), config=AgentConfig()), target)
    result = ev.evaluate(narrator, [sample], catalog_ids, categories, products)
    session = result["sessions"][0]

    print(f"\n{RULE}")
    if session["hit"]:
        print(f"RESULT   : found at rank {session['best_rank']} on turn {session['first_hit_turn']}"
              f"   RR={session['reciprocal_rank']:.4f}")
    else:
        print("RESULT   : not found within 10 turns")
    print(f"score    : {result['recommended_technical_score']:.4f} (this session alone)")
    print(RULE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
