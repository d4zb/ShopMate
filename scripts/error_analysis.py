"""Where the agent still loses, and why.

    python scripts/error_analysis.py --split dev

Runs the shipped configuration behind a probe that records, per session, what was
parsed out of each utterance and how the candidate pool evolved. Then reports the
sessions the agent missed outright and the ones it found but ranked poorly.

Writes ``docs/error_analysis.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from agent import Agent, AgentConfig  # noqa: E402

WORST_N = 5


class Trace:
    """Records the agent's internal state per turn, keyed by session order."""

    def __init__(self, inner: Agent) -> None:
        self.inner = inner
        self.rows: list[dict] = []
        self._current: dict | None = None

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._current = {"turns": [], "session_id": session_id}
        self.rows.append(self._current)
        self.inner.reset(session_id, user_profile)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        response = self.inner.respond(session_id, user_message, turn, top_k)
        state = self.inner._sessions[session_id]
        pool = self.inner._pool(state)
        self._current["turns"].append({
            "turn": turn,
            "message": user_message[:110],
            "category": state.category,
            "n_constraints": len(state.known_constraints),
            "pool": int(len(pool)) if pool is not None else None,
            "recommended": len(response["recommendations"]),
            "exhausted": state.exhausted,
        })
        return response


def main() -> int:
    parser = argparse.ArgumentParser(description="Error analysis for the shipped configuration")
    parser.add_argument("--split", choices=("dev", "holdout", "all"), default="dev")
    parser.add_argument("--catalog", type=Path, default=ROOT / "data" / "catalog.jsonl")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "public_set.jsonl")
    parser.add_argument("--splits", type=Path, default=ROOT / "data" / "splits.json")
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    if args.split != "all":
        wanted = set(json.loads(args.splits.read_text(encoding="utf-8"))[args.split])
        samples = [s for s in samples if s["sample_id"] in wanted]
    catalog_ids, categories, products = catalog_index(args.catalog)

    probe = Trace(Agent(str(args.catalog), config=AgentConfig()))
    result = evaluate(probe, samples, catalog_ids, categories, products)

    merged = []
    for sample, session, trace in zip(samples, result["sessions"], probe.rows):
        merged.append({**session, "difficulty": sample.get("difficulty_bucket"),
                       "target": sample["ground_truth"]["parent_asin"], "trace": trace["turns"]})

    missed = [m for m in merged if not m["hit"]]
    ranked_poorly = sorted([m for m in merged if m["hit"] and m["best_rank"] > 1],
                           key=lambda m: -m["best_rank"])[:WORST_N]

    # Did category parsing ever fail? That is the failure mode with real blast radius.
    parse_failures = [m for m in merged if m["trace"] and m["trace"][0]["category"] is None]
    unparsed_turns = sum(
        1 for m in merged for t in m["trace"] if t["turn"] > 1 and t["n_constraints"] == 0
    )

    out = ["# Error analysis", "",
           f"Configuration: shipped default. Split: `{args.split}` ({len(samples)} sessions).",
           f"Hit@10 {result['hit_rate_at_10']:.4f} | MRR {result['mrr']:.4f} | "
           f"MTTC {result['mttc']:.3f} | TechnicalScore {result['recommended_technical_score']:.4f}",
           "", "## Parsing health", "",
           f"- Openers whose `coarse_category` failed to resolve: **{len(parse_failures)}/{len(samples)}**",
           f"- Follow-up turns that yielded no constraint: **{unparsed_turns}**",
           "", f"## Missed entirely ({len(missed)})", ""]

    if not missed:
        out.append("None.")
    for m in missed:
        pools = " -> ".join(str(t["pool"]) for t in m["trace"])
        out += [f"### `{m['sample_id']}` ({m['scenario_type']}, {m['difficulty']})",
                f"- target `{m['target']}`, pool trajectory: `{pools}`",
                f"- turns: {len(m['trace'])}, category `{m['trace'][0]['category']}`", ""]

    out += [f"## Found but ranked below 1 ({len(ranked_poorly)} worst)", ""]
    if not ranked_poorly:
        out.append("None.")
    for m in ranked_poorly:
        pools = " -> ".join(str(t["pool"]) for t in m["trace"])
        out += [f"### `{m['sample_id']}` ({m['scenario_type']}, {m['difficulty']})",
                f"- rank {m['best_rank']} at turn {m['first_hit_turn']}, "
                f"pool trajectory: `{pools}`",
                f"- constraints recovered: {m['trace'][-1]['n_constraints']}, "
                f"exhausted: {m['trace'][-1]['exhausted']}", ""]

    path = ROOT / "docs" / "error_analysis.md"
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
