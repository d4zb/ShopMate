"""A generative model of the simulated shopper, copied from the official evaluator.

The organisers guarantee that the final 800-session package uses "the same input
schema, Agent interface, metric formula, stopping rule, invalid-output handling,
deterministic customer-message templates, and ``ask_attribute`` response policy
as the released official evaluator. No undisclosed natural-language paraphrases
are introduced." (docs/final_evaluation_faq.md, section 1.) FAQ section 4 adds
that intent cards there derive from the same frozen catalog metadata we hold.

That makes the shopper's utterance distribution a known, deterministic function
of the target product, so we can model it explicitly and invert it. Every
function below is a verbatim copy of its counterpart in
``evaluator/local_evaluator.py``.

It has to be a copy rather than an import. ``evaluator/local_evaluator.py:12``
does ``from starter.agent import Agent``, and ``starter/agent.py`` re-exports our
agent, so importing the evaluator from here would be an import cycle.
``tests/test_simulator_model.py`` asserts equivalence against the evaluator's own
functions across all 50,000 catalog products, which is what makes the copy safe:
if the released package ever changes these definitions, that test fails loudly
instead of the agent silently mis-parsing.
"""

from __future__ import annotations

import re

# --- verbatim from evaluator/local_evaluator.py -----------------------------

ALLOWED_ATTRIBUTES = {
    "category", "material", "color", "size", "style", "brand",
    "budget", "feature", "use_case", "other",
}
MATERIALS = ("cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon", "fabric")
SEARCH_FIELDS = ("title", "features", "details", "description", "categories", "store")
MATERIAL_RE = re.compile(r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric)\b", re.I)
COLOR_RE = re.compile(r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange)\b", re.I)


def searchable_text(product: dict) -> str:
    parts: list[str] = []
    for field in SEARCH_FIELDS:
        value = product.get(field)
        if isinstance(value, dict):
            parts.extend(f"{key} {item}" for key, item in value.items())
        elif isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif value is not None:
            parts.append(str(value))
    return " ".join(parts).strip()


def _flatten_values(value: object) -> list[str]:
    if isinstance(value, dict):
        return [f"{key}: {item}" for key, item in value.items() if item not in (None, "", [])]
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)] if value not in (None, "") else []


def _clean_constraint(value: str, limit: int) -> str:
    return re.sub(r"\s+", " ", value).strip(" -;,.\t\n")[:limit].rstrip()


def intent_card(product: dict, limit: int = 180) -> dict:
    title = _clean_constraint(str(product.get("title") or "product"), limit)
    candidates = [*_flatten_values(product.get("features")), *_flatten_values(product.get("details"))]
    corpus = searchable_text(product)
    material = MATERIAL_RE.search(corpus)
    color = COLOR_RE.search(corpus)
    if material:
        candidates.insert(0, material.group(1).lower())
    if color:
        candidates.insert(1, f"color: {color.group(1).lower()}")
    if product.get("price") not in (None, ""):
        candidates.append(f"budget around ${product['price']}")
    cleaned = list(dict.fromkeys(_clean_constraint(item, limit) for item in candidates if _clean_constraint(item, limit)))
    if not cleaned:
        cleaned = [title]
    return {
        "target_category": title,
        "hard_constraints": cleaned[:2],
        "soft_preferences": cleaned[2:4] or cleaned[:1],
    }


def coarse_category(values: list[str]) -> str:
    excluded = {"clothing", "clothing shoes & jewelry", "clothing, shoes & jewelry"}
    cleaned: list[str] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part and part.lower() not in excluded:
                cleaned.append(part)
    return " ".join(cleaned[-2:]) if cleaned else "clothing item"


def classify_constraint(value: str) -> str:
    lowered = value.lower()
    if "budget" in lowered or re.search(r"(?:\$|<=|under)\s*\d", lowered):
        return "budget"
    if any(material in lowered for material in MATERIALS):
        return "material"
    if any(word in lowered for word in ("color", "black", "white", "blue", "red", "pink", "green")):
        return "color"
    if any(word in lowered for word in ("size", "sizing", "width", "wide", "narrow")):
        return "size"
    if any(word in lowered for word in ("department", "style", "fit", "sleeve", "neck")):
        return "style"
    if any(word in lowered for word in ("hiking", "running", "gym", "winter", "outdoor", "work")):
        return "use_case"
    return "feature"


# --- the message templates, as recognisers -----------------------------------
# Derived from initial_message() and customer_reply(). The evaluator builds these
# by f-string interpolation; we take them apart again.

OPENER_PREFIX = "I'm looking for "
BROWSING_SUFFIX = ", but I'm still exploring."
BUYING_MARKER = ". A key requirement is: "
DISCLOSE_PREFIX = "For that, what matters is: "
OVERRIDE_PREFIX = "Actually, ignore my earlier preference. What I need is: "

#: Replies that carry no new information. The first two mean the intent card is
#: spent, which is the signal to stop withholding and commit to a recommendation.
NO_PREFERENCE_PREFIX = "I don't have an additional preference for "
BOUNDARY_PREFIX = "I don't have a preference for "
NUDGE = "Those options are not quite right yet. Ask me about one specific attribute."


def carries_no_information(message: str) -> bool:
    return (
        message.startswith(NO_PREFERENCE_PREFIX)
        or message.startswith(BOUNDARY_PREFIX)
        or message == NUDGE
    )


def card_constraints(product: dict) -> list[str]:
    """The constraint strings this product would ever disclose, in disclosure order.

    ``customer_reply`` iterates hard_constraints then soft_preferences, so this
    ordering is exactly the sequence a shopper reveals them in.
    """
    card = intent_card(product)
    return list(dict.fromkeys([*card["hard_constraints"], *card["soft_preferences"]]))
