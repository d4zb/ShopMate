# Robustness to non-template phrasing

Split: `dev` (150 sessions). Each row rewrites every
customer utterance before the agent sees it; the evaluator's own scoring and
disclosure bookkeeping are untouched.

| perturbation | Hit@10 | MRR | MTTC | TechnicalScore | vs control |
|---|---|---|---|---|---|
| none (control) | 0.9933 | 0.9665 | 2.653 | 0.9536 | +0.0000 |
| trailing period dropped | 0.9933 | 0.9665 | 2.653 | 0.9536 | +0.0000 |
| opener reworded | 0.9867 | 0.9510 | 2.960 | 0.9394 | -0.0141 |
| '; ' delimiter -> ', ' | 0.9933 | 0.9660 | 2.667 | 0.9531 | -0.0004 |
| polite prefix added | 0.9933 | 0.9665 | 2.627 | 0.9541 | +0.0005 |
| trailing chatter added | 0.8400 | 0.8012 | 4.073 | 0.7989 | -0.1547 |
| all lowercase | 0.9533 | 0.8887 | 3.240 | 0.8985 | -0.0551 |
| double spaces | 0.9533 | 0.8887 | 3.240 | 0.8985 | -0.0551 |
| last word dropped | 0.7467 | 0.6771 | 5.087 | 0.6947 | -0.2588 |
