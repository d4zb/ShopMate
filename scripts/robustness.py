"""How far does the agent degrade when the shopper stops speaking in templates?

    python scripts/robustness.py --split dev

The submission's primary path models the organiser's deterministic customer
templates (FAQ sections 1 and 4). The fair criticism of that is "it only works
because the templates are frozen". This quantifies the criticism instead of
conceding it.

Each perturbation rewrites what the *agent* sees while leaving the evaluator's
own bookkeeping untouched: ``initial_message`` and ``customer_reply`` are wrapped
in memory, so the disclosed-constraint set, the scenario policy and the scoring
are all exactly as the organisers wrote them. Only the surface string changes.

Writes ``docs/robustness.md``.
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

#: Each entry rewrites one customer utterance. Ordered roughly by severity.
PERTURBATIONS: dict[str, callable] = {
    "none (control)": lambda m: m,
    "trailing period dropped": lambda m: m[:-1] if m.endswith(".") else m,
    "opener reworded": lambda m: m.replace("I'm looking for ", "I need "),
    "'; ' delimiter -> ', '": lambda m: m.replace("; ", ", "),
    "polite prefix added": lambda m: "Hi! " + m,
    "trailing chatter added": lambda m: m + " Thanks so much!",
    "all lowercase": lambda m: m.lower(),
    "double spaces": lambda m: m.replace(" ", "  "),
    "last word dropped": lambda m: " ".join(m.split()[:-1]),
}


def install(perturb) -> None:
    """Wrap the evaluator's two message emitters, in memory only."""
    def initial(sample, category, disclosed):
        return perturb(install.original_initial(sample, category, disclosed))

    def reply(sample, attribute, disclosed, boundary_used):
        message, used = install.original_reply(sample, attribute, disclosed, boundary_used)
        return perturb(message), used

    ev.initial_message, ev.customer_reply = initial, reply


install.original_initial = ev.initial_message
install.original_reply = ev.customer_reply


def restore() -> None:
    ev.initial_message = install.original_initial
    ev.customer_reply = install.original_reply


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure robustness to non-template phrasing")
    parser.add_argument("--split", choices=("dev", "holdout", "all"), default="dev")
    parser.add_argument("--catalog", type=Path, default=ROOT / "data" / "catalog.jsonl")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "public_set.jsonl")
    parser.add_argument("--splits", type=Path, default=ROOT / "data" / "splits.json")
    args = parser.parse_args()

    samples = ev.load_jsonl(args.dataset)
    if args.split != "all":
        wanted = set(json.loads(args.splits.read_text(encoding="utf-8"))[args.split])
        samples = [s for s in samples if s["sample_id"] in wanted]
    catalog_ids, categories, products = ev.catalog_index(args.catalog)

    header = f"{'perturbation':<28}{'Hit@10':>9}{'MRR':>9}{'MTTC':>8}{'Score':>9}{'vs ctrl':>9}"
    print(f"split={args.split}  sessions={len(samples)}\n")
    print(header)
    print("-" * len(header))

    rows, control = [], None
    for name, perturb in PERTURBATIONS.items():
        install(perturb)
        try:
            result = ev.evaluate(Agent(str(args.catalog), config=AgentConfig()),
                                 samples, catalog_ids, categories, products)
        finally:
            restore()
        score = result["recommended_technical_score"]
        if control is None:
            control = score
        print(f"{name:<28}{result['hit_rate_at_10']:>9.4f}{result['mrr']:>9.4f}"
              f"{result['mttc']:>8.3f}{score:>9.4f}{score - control:>+9.4f}")
        rows.append({"perturbation": name, "hit_rate_at_10": result["hit_rate_at_10"],
                     "mrr": result["mrr"], "mttc": result["mttc"],
                     "technical_score": score, "delta_vs_control": round(score - control, 6)})

    worst = min(rows, key=lambda r: r["technical_score"])
    print(f"\ncontrol {control:.4f} | worst {worst['perturbation']} {worst['technical_score']:.4f} "
          f"({worst['delta_vs_control']:+.4f})")

    lines = ["# Robustness to non-template phrasing", "",
             f"Split: `{args.split}` ({len(samples)} sessions). Each row rewrites every",
             "customer utterance before the agent sees it; the evaluator's own scoring and",
             "disclosure bookkeeping are untouched.", "",
             "| perturbation | Hit@10 | MRR | MTTC | TechnicalScore | vs control |",
             "|---|---|---|---|---|---|"]
    lines += [f"| {r['perturbation']} | {r['hit_rate_at_10']:.4f} | {r['mrr']:.4f} | "
              f"{r['mttc']:.3f} | {r['technical_score']:.4f} | {r['delta_vs_control']:+.4f} |"
              for r in rows]
    (ROOT / "docs" / "robustness.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (ROOT / "docs" / "robustness.json").write_text(
        json.dumps({"split": args.split, "sessions": len(samples), "rows": rows}, indent=2) + "\n",
        encoding="utf-8")
    print("wrote docs/robustness.md, docs/robustness.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
