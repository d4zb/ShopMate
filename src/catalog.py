"""Loading and normalising the frozen 50,000-product catalog.

Catalog order is the canonical index order used by every downstream array:
row ``i`` of the embedding matrix and of any score vector is ``asins[i]``.
The file is read once per process and shared immutably across sessions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

#: Fields the organiser exposes to participants, verified against catalog.jsonl.
FIELDS = (
    "parent_asin", "title", "features", "description",
    "price", "categories", "details", "average_rating", "rating_number", "store",
)

#: Field order used to build the retrieval document for each product.
TEXT_FIELDS = ("title", "categories", "features", "details", "description", "store")


def _flatten(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


@dataclass(frozen=True)
class Catalog:
    """An immutable, index-addressable view over the catalog."""

    products: list[dict]
    asins: list[str]
    index_of: dict[str, int]

    def __len__(self) -> int:
        return len(self.products)

    def document(self, position: int) -> str:
        """Concatenated searchable text for one product, used by BM25 and dense encoding."""
        product = self.products[position]
        return " ".join(part for part in (_flatten(product.get(field)) for field in TEXT_FIELDS) if part)

    def documents(self) -> list[str]:
        return [self.document(position) for position in range(len(self.products))]


def load(catalog_path: str | Path = "data/catalog.jsonl") -> Catalog:
    products: list[dict] = []
    asins: list[str] = []
    with Path(catalog_path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            product = json.loads(line)
            products.append(product)
            asins.append(str(product["parent_asin"]))
    return Catalog(products=products, asins=asins, index_of={asin: i for i, asin in enumerate(asins)})
