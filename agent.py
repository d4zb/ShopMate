"""Submission entry point.

The official evaluator constructs this once per run as ``Agent(catalog_path)``,
then calls ``reset`` once per session and ``respond`` once per turn.

Scaffolding stage: this re-exports the organiser's weak BM25 starter unchanged,
so the harness is proven end to end before any behaviour is added. It is
replaced by the real implementation in the retrieval step.

Note that ``starter/agent.py`` re-exports *this* module, because the evaluator
imports the agent from there. Nothing under ``src/`` may import the evaluator at
module scope or that becomes an import cycle.
"""

from __future__ import annotations

from baselines.starter_bm25 import Agent

__all__ = ["Agent"]
