"""Official evaluator entry point.

``evaluator/local_evaluator.py`` hardcodes ``from starter.agent import Agent``,
so this module is the hook the organiser's harness pulls on. The submission
rules permit replacing the starter Agent provided the interface is preserved,
so this re-exports our implementation from the repository root.

The organiser's original weak BM25 starter is preserved unmodified at
``baselines/starter_bm25.py`` and is still used for the ablation table.
"""

from __future__ import annotations

from agent import Agent

__all__ = ["Agent"]
