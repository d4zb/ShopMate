"""One-command evaluation against the unmodified official evaluator.

    python scripts/evaluate.py --agent ours --split dev

Imports ``evaluator.local_evaluator.evaluate`` directly rather than
reimplementing any scoring, so the numbers printed here are the organiser's
numbers. ``--split`` selects the frozen dev/holdout partition from
``data/splits.json``; the holdout is reserved for final reporting only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402

AGENTS = {
    # The organiser's weak BM25 starter, preserved for the ablation table.
    "starter": ("baselines.starter_bm25", "Agent"),
    # Our submission. `starter.agent` re-exports this, which is what the
    # official `python -m evaluator.local_evaluator` command picks up.
    "ours": ("agent", "Agent"),
}


def build_agent(name: str, catalog_path: Path):
    module_name, attribute = AGENTS[name]
    module = __import__(module_name, fromlist=[attribute])
    return getattr(module, attribute)(str(catalog_path))


def select(samples: list[dict], split: str, splits_path: Path) -> list[dict]:
    if split == "all":
        return samples
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    wanted = set(splits[split])
    return [sample for sample in samples if sample["sample_id"] in wanted]


def technical_score(summary: dict) -> float:
    """Mirror of the evaluator's composite, for per-scenario rows it does not compute."""
    efficiency = max(0.0, min(1.0, (11.0 - float(summary["mttc"])) / 10.0))
    return 0.50 * summary["hit_rate_at_10"] + 0.30 * summary["mrr"] + 0.20 * efficiency


def print_table(result: dict) -> None:
    header = f"{'segment':<18}{'n':>5}{'Hit@10':>9}{'MRR':>9}{'MTTC':>8}{'Eff':>8}{'Score':>9}"
    print(header)
    print("-" * len(header))

    def row(name: str, summary: dict, efficiency: float, score: float) -> None:
        print(
            f"{name:<18}{summary['sample_count']:>5}{summary['hit_rate_at_10']:>9.4f}"
            f"{summary['mrr']:>9.4f}{summary['mttc']:>8.3f}{efficiency:>8.4f}{score:>9.4f}"
        )

    row("OVERALL", result, result["efficiency"], result["recommended_technical_score"])
    print("-" * len(header))
    for name, summary in result["scenario_metrics"].items():
        efficiency = max(0.0, min(1.0, (11.0 - float(summary["mttc"])) / 10.0))
        row(name, summary, efficiency, technical_score(summary))


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate an agent on the public sessions")
    parser.add_argument("--agent", choices=sorted(AGENTS), default="ours")
    parser.add_argument("--split", choices=("dev", "holdout", "all"), default="dev")
    parser.add_argument("--catalog", type=Path, default=ROOT / "data" / "catalog.jsonl")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "public_set.jsonl")
    parser.add_argument("--splits", type=Path, default=ROOT / "data" / "splits.json")
    parser.add_argument("--output", type=Path, default=None, help="write full results.json here")
    args = parser.parse_args()

    if args.split == "holdout":
        print("NOTE: evaluating on the HOLDOUT split. This is reserved for final reporting.\n")

    started = time.perf_counter()
    samples = select(load_jsonl(args.dataset), args.split, args.splits)
    catalog_ids, categories, products = catalog_index(args.catalog)
    loaded = time.perf_counter()

    agent = build_agent(args.agent, args.catalog)
    built = time.perf_counter()

    result = evaluate(agent, samples, catalog_ids, categories, products)
    finished = time.perf_counter()

    print(f"agent={args.agent}  split={args.split}  sessions={len(samples)}\n")
    print_table(result)
    print(
        f"\nload {loaded - started:.1f}s | agent init {built - loaded:.1f}s | "
        f"eval {finished - built:.1f}s | total {finished - started:.1f}s"
    )
    usage = result["reported_token_usage"]
    print(f"reported tokens: {usage['total_tokens']:,}")

    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
