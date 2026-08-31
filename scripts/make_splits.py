"""Freeze a 150/50 dev/holdout split of the 200 public sessions.

Stratified by ``scenario_type`` so both halves keep the official 40/40/15/5
Buying/Browsing/Intent-Override/Boundary mix. The result is committed as
``data/splits.json`` and must never be regenerated with a different seed:
the holdout is only touched once, in the final reporting step.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

SEED = 1337
HOLDOUT_FRACTION = 0.25

ROOT = Path(__file__).resolve().parent.parent
PUBLIC_SET = ROOT / "data" / "public_set.jsonl"
SPLITS = ROOT / "data" / "splits.json"


def main() -> None:
    samples = [json.loads(line) for line in PUBLIC_SET.open(encoding="utf-8") if line.strip()]

    by_scenario: dict[str, list[str]] = defaultdict(list)
    for sample in samples:
        by_scenario[sample["scenario_type"]].append(sample["sample_id"])

    rng = random.Random(SEED)
    dev: list[str] = []
    holdout: list[str] = []
    for scenario in sorted(by_scenario):
        ids = sorted(by_scenario[scenario])
        rng.shuffle(ids)
        cut = round(len(ids) * HOLDOUT_FRACTION)
        holdout.extend(ids[:cut])
        dev.extend(ids[cut:])

    payload = {
        "seed": SEED,
        "holdout_fraction": HOLDOUT_FRACTION,
        "dev": sorted(dev),
        "holdout": sorted(holdout),
    }
    SPLITS.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    scenario_of = {sample["sample_id"]: sample["scenario_type"] for sample in samples}
    print(f"seed={SEED}  dev={len(dev)}  holdout={len(holdout)}  -> {SPLITS}")
    for name, ids in (("dev", dev), ("holdout", holdout)):
        counts: dict[str, int] = defaultdict(int)
        for sample_id in ids:
            counts[scenario_of[sample_id]] += 1
        breakdown = "  ".join(f"{key}={counts[key]}" for key in sorted(counts))
        print(f"  {name:8s} {breakdown}")


if __name__ == "__main__":
    main()
