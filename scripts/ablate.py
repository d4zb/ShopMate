"""Regenerate the ablation table, plus recall instrumentation.

    python scripts/ablate.py --split dev

Every row is a real run of the unmodified official evaluator through the same
``Agent`` the submission ships, differing only in ``AgentConfig`` flags.

Recall is measured separately from Hit@10 because they answer different
questions. Hit@10 is capped by retrieval: if the target is not in the candidate
pool, no amount of ranking recovers it. The recall@50 / recall@200 gap says
whether the next unit of effort belongs in retrieval or in ranking. It is
computed by a proxy that counts ``reset`` calls to recover which sample is in
flight -- the evaluator processes sessions sequentially and never tells the agent
which one it is looking at.
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
from agent import Agent, AgentConfig  # noqa: E402

DEEP_K = 200
RECALL_AT = (10, 50, 200)

#: (label, config). Ordered so each row adds exactly one component.
ROWS: list[tuple[str, AgentConfig | None]] = [
    ("provided BM25 starter", None),
    ("BM25 only", AgentConfig(use_bm25=True, use_popularity=False, use_inversion=False,
                              policy="threshold", convert_threshold=None)),
    ("popularity prior only", AgentConfig(use_bm25=False, use_popularity=True, use_inversion=False,
                                          policy="threshold", convert_threshold=None)),
    ("+ RRF fusion of both", AgentConfig(use_inversion=False,
                                         policy="threshold", convert_threshold=None)),
    ("+ dense MiniLM and RRF", AgentConfig(use_dense=True, use_inversion=False,
                                           policy="threshold", convert_threshold=None)),
    ("+ simulator inversion", AgentConfig(use_inversion=True,
                                          policy="threshold", convert_threshold=None)),
    ("+ conversion timing (full)", AgentConfig(use_inversion=True, policy="expected_value")),
    ("full + dense", AgentConfig(use_dense=True, use_inversion=True, policy="expected_value")),
]


class RecallProbe:
    """Wraps an agent, recording how deep the target sits in its full ranking."""

    def __init__(self, inner: Agent, targets: list[str]) -> None:
        self.inner = inner
        self.targets = targets
        self.session_no = -1
        self.best_depth: list[int | None] = []
        self._session_id = ""

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.session_no += 1
        self._session_id = session_id
        self.best_depth.append(None)
        self.inner.reset(session_id, user_profile)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        response = self.inner.respond(session_id, user_message, turn, top_k)
        deep = self.inner.rank_for(session_id, DEEP_K)
        target = self.targets[self.session_no]
        asins = self.inner.catalog.asins
        for depth, position in enumerate(deep, start=1):
            if asins[position] == target:
                current = self.best_depth[self.session_no]
                if current is None or depth < current:
                    self.best_depth[self.session_no] = depth
                break
        return response


def build(label: str, config: AgentConfig | None, catalog_path: Path):
    if config is None:
        from baselines.starter_bm25 import Agent as Starter
        return Starter(str(catalog_path))
    return Agent(str(catalog_path), config=config)


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate the ablation table")
    parser.add_argument("--split", choices=("dev", "holdout", "all"), default="dev")
    parser.add_argument("--catalog", type=Path, default=ROOT / "data" / "catalog.jsonl")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "public_set.jsonl")
    parser.add_argument("--splits", type=Path, default=ROOT / "data" / "splits.json")
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "ablation.json")
    parser.add_argument("--recall", action="store_true", help="also measure recall@k (slower)")
    args = parser.parse_args()

    if args.split == "holdout":
        print("NOTE: HOLDOUT split. Reserved for final reporting.\n")

    samples = load_jsonl(args.dataset)
    if args.split != "all":
        wanted = set(json.loads(args.splits.read_text(encoding="utf-8"))[args.split])
        samples = [s for s in samples if s["sample_id"] in wanted]
    targets = [str(s["ground_truth"]["parent_asin"]) for s in samples]
    catalog_ids, categories, products = catalog_index(args.catalog)

    header = f"{'configuration':<28}{'Hit@10':>9}{'MRR':>9}{'MTTC':>8}{'Score':>9}{'secs':>7}"
    if args.recall:
        header += "".join(f"{f'R@{k}':>8}" for k in RECALL_AT)
    print(f"split={args.split}  sessions={len(samples)}\n")
    print(header)
    print("-" * len(header))

    table = []
    for label, config in ROWS:
        started = time.perf_counter()
        agent = build(label, config, args.catalog)
        probe = RecallProbe(agent, targets) if (args.recall and config is not None) else None
        result = evaluate(probe or agent, samples, catalog_ids, categories, products)
        elapsed = time.perf_counter() - started

        row = f"{label:<28}{result['hit_rate_at_10']:>9.4f}{result['mrr']:>9.4f}" \
              f"{result['mttc']:>8.3f}{result['recommended_technical_score']:>9.4f}{elapsed:>7.1f}"
        entry = {"configuration": label, "hit_rate_at_10": result["hit_rate_at_10"],
                 "mrr": result["mrr"], "mttc": result["mttc"],
                 "technical_score": result["recommended_technical_score"], "seconds": round(elapsed, 1)}
        if args.recall:
            if probe is None:
                row += "".join(f"{'-':>8}" for _ in RECALL_AT)
            else:
                for k in RECALL_AT:
                    value = sum(1 for d in probe.best_depth if d is not None and d <= k) / len(samples)
                    entry[f"recall_at_{k}"] = round(value, 6)
                    row += f"{value:>8.4f}"
        print(row)
        table.append(entry)

    args.output.write_text(json.dumps({"split": args.split, "rows": table}, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
