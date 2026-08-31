"""Sweep the conversion threshold and compare it against the untuned EV rule.

    python scripts/sweep.py --split dev

This is the evidence for the project's central claim. The original brief assumed
the tunable quantity was *how many questions to ask*, and predicted a flat curve
because asking is free under this evaluator. It is flat, and that is a null
result reported in the README.

The quantity that actually matters is *when to allow conversion*. Because
``evaluate`` breaks on the first hit, a premature hit at a poor rank is
unrecoverable. That curve is not flat: it has an interior optimum.

Outputs ``docs/sweep.json``, ``docs/sweep.md`` and ``docs/sweep.svg``. The chart
is emitted as hand-built SVG rather than via matplotlib, which is outside this
project's dependency budget.
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

THRESHOLDS: list[int | None] = [1, 2, 3, 5, 7, 10, 25, 50, None]


def run(config: AgentConfig, samples, catalog_ids, categories, products, catalog_path) -> dict:
    agent = Agent(str(catalog_path), config=config)
    return evaluate(agent, samples, catalog_ids, categories, products)


def svg(points: list[tuple[str, float]], best: str, path: Path) -> None:
    """Minimal dependency-free line chart of score against threshold."""
    width, height, pad = 720, 360, 56
    values = [v for _, v in points]
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    lo, hi = lo - span * 0.15, hi + span * 0.15
    step = (width - 2 * pad) / max(len(points) - 1, 1)

    def xy(i: int, v: float) -> tuple[float, float]:
        return pad + i * step, height - pad - (v - lo) / (hi - lo) * (height - 2 * pad)

    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'font-family="system-ui,sans-serif" font-size="12">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{width/2}" y="24" text-anchor="middle" font-size="15" font-weight="600">'
        f'TechnicalScore vs conversion threshold K</text>',
        f'<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#94a3b8"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#94a3b8"/>',
    ]
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        value = lo + frac * (hi - lo)
        y = height - pad - frac * (height - 2 * pad)
        body.append(f'<line x1="{pad}" y1="{y:.1f}" x2="{width-pad}" y2="{y:.1f}" stroke="#e2e8f0"/>')
        body.append(f'<text x="{pad-8}" y="{y+4:.1f}" text-anchor="end" fill="#475569">{value:.3f}</text>')

    path_d = " ".join(
        ("M" if i == 0 else "L") + f"{xy(i,v)[0]:.1f},{xy(i,v)[1]:.1f}"
        for i, (_, v) in enumerate(points)
    )
    body.append(f'<path d="{path_d}" fill="none" stroke="#2563eb" stroke-width="2.5"/>')
    for i, (label, value) in enumerate(points):
        x, y = xy(i, value)
        highlight = label == best
        body.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{5 if highlight else 3.5}" '
            f'fill="{"#dc2626" if highlight else "#2563eb"}"/>'
        )
        body.append(f'<text x="{x:.1f}" y="{height-pad+18}" text-anchor="middle" fill="#475569">{label}</text>')
        if highlight:
            body.append(f'<text x="{x:.1f}" y="{y-12:.1f}" text-anchor="middle" fill="#dc2626" '
                        f'font-weight="600">{value:.4f}</text>')
    body.append(f'<text x="{width/2}" y="{height-12}" text-anchor="middle" fill="#475569">'
                f'convert only when the candidate pool has at most K members '
                f'(rightmost = always recommend)</text>')
    body.append("</svg>")
    path.write_text("\n".join(body), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep the conversion threshold")
    parser.add_argument("--split", choices=("dev", "holdout", "all"), default="dev")
    parser.add_argument("--catalog", type=Path, default=ROOT / "data" / "catalog.jsonl")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "public_set.jsonl")
    parser.add_argument("--splits", type=Path, default=ROOT / "data" / "splits.json")
    args = parser.parse_args()

    if args.split == "holdout":
        print("NOTE: HOLDOUT split. Reserved for final reporting.\n")

    samples = load_jsonl(args.dataset)
    if args.split != "all":
        wanted = set(json.loads(args.splits.read_text(encoding="utf-8"))[args.split])
        samples = [s for s in samples if s["sample_id"] in wanted]
    catalog_ids, categories, products = catalog_index(args.catalog)
    shared = (samples, catalog_ids, categories, products, args.catalog)

    print(f"split={args.split}  sessions={len(samples)}\n")
    header = f"{'policy':<26}{'Hit@10':>9}{'MRR':>9}{'MTTC':>8}{'Score':>9}{'secs':>7}"
    print(header)
    print("-" * len(header))

    rows, points = [], []
    for threshold in THRESHOLDS:
        label = "always recommend" if threshold is None else f"K={threshold}"
        started = time.perf_counter()
        result = run(AgentConfig(policy="threshold", convert_threshold=threshold), *shared)
        elapsed = time.perf_counter() - started
        score = result["recommended_technical_score"]
        print(f"{label:<26}{result['hit_rate_at_10']:>9.4f}{result['mrr']:>9.4f}"
              f"{result['mttc']:>8.3f}{score:>9.4f}{elapsed:>7.1f}")
        rows.append({"policy": "threshold", "threshold": threshold, "label": label,
                     "hit_rate_at_10": result["hit_rate_at_10"], "mrr": result["mrr"],
                     "mttc": result["mttc"], "technical_score": score})
        points.append(("inf" if threshold is None else str(threshold), score))

    started = time.perf_counter()
    result = run(AgentConfig(policy="expected_value"), *shared)
    elapsed = time.perf_counter() - started
    ev_score = result["recommended_technical_score"]
    print("-" * len(header))
    print(f"{'expected-value (untuned)':<26}{result['hit_rate_at_10']:>9.4f}{result['mrr']:>9.4f}"
          f"{result['mttc']:>8.3f}{ev_score:>9.4f}{elapsed:>7.1f}")
    rows.append({"policy": "expected_value", "threshold": None, "label": "expected-value (untuned)",
                 "hit_rate_at_10": result["hit_rate_at_10"], "mrr": result["mrr"],
                 "mttc": result["mttc"], "technical_score": ev_score})

    best = max((r for r in rows if r["policy"] == "threshold"), key=lambda r: r["technical_score"])
    best_label = "inf" if best["threshold"] is None else str(best["threshold"])
    print(f"\nbest swept threshold: {best['label']} at {best['technical_score']:.4f}")
    print(f"untuned expected-value rule: {ev_score:.4f} "
          f"({ev_score - best['technical_score']:+.4f} vs best swept)")

    (ROOT / "docs" / "sweep.json").write_text(
        json.dumps({"split": args.split, "sessions": len(samples), "rows": rows}, indent=2) + "\n",
        encoding="utf-8")
    lines = ["| policy | Hit@10 | MRR | MTTC | TechnicalScore |", "|---|---|---|---|---|"]
    lines += [f"| {r['label']} | {r['hit_rate_at_10']:.4f} | {r['mrr']:.4f} | "
              f"{r['mttc']:.3f} | {r['technical_score']:.4f} |" for r in rows]
    (ROOT / "docs" / "sweep.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    svg(points, best_label, ROOT / "docs" / "sweep.svg")
    print("wrote docs/sweep.json, docs/sweep.md, docs/sweep.svg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
