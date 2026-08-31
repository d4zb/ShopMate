"""Download and verify the frozen 50,000-product catalog from the participant kit.

The catalog is a 19 MB release asset rather than a repository file, so it is
fetched once and verified against the release SHA256SUMS before use. Everything
downstream assumes exactly 50,000 rows keyed by ``parent_asin``.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

RELEASE = "https://github.com/TechJam2026/techjam-conversational-search/releases/download/participant-kit"
ARCHIVE_NAME = "catalog.jsonl.gz"
EXPECTED_ROWS = 50_000

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


def _download(url: str, destination: Path) -> None:
    print(f"downloading {url}")
    with urllib.request.urlopen(url) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    print(f"  -> {destination} ({destination.stat().st_size:,} bytes)")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_digest() -> str | None:
    """Read the archive's digest out of the release SHA256SUMS manifest."""
    try:
        with urllib.request.urlopen(f"{RELEASE}/SHA256SUMS") as response:
            manifest = response.read().decode("utf-8")
    except OSError as error:
        print(f"warning: could not fetch SHA256SUMS ({error}); skipping verification")
        return None
    for line in manifest.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].lstrip("*") == ARCHIVE_NAME:
            return parts[0]
    print(f"warning: {ARCHIVE_NAME} not listed in SHA256SUMS; skipping verification")
    return None


def _decompress(archive: Path, destination: Path) -> int:
    rows = 0
    with gzip.open(archive, "rt", encoding="utf-8") as source, destination.open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for line in source:
            if line.strip():
                handle.write(line if line.endswith("\n") else line + "\n")
                rows += 1
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch and verify the frozen catalog")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    catalog = DATA_DIR / "catalog.jsonl"
    archive = DATA_DIR / ARCHIVE_NAME

    if catalog.exists() and not args.force:
        rows = sum(1 for line in catalog.open(encoding="utf-8") if line.strip())
        print(f"{catalog} already present with {rows:,} rows")
        if rows != EXPECTED_ROWS:
            print(f"ERROR: expected {EXPECTED_ROWS:,} rows, found {rows:,}. Re-run with --force.")
            return 1
        return 0

    if not archive.exists() or args.force:
        _download(f"{RELEASE}/{ARCHIVE_NAME}", archive)

    expected = _expected_digest()
    if expected is not None:
        actual = _sha256(archive)
        if actual != expected:
            print(f"ERROR: SHA256 mismatch\n  expected {expected}\n  actual   {actual}")
            return 1
        print(f"sha256 verified: {actual}")

    rows = _decompress(archive, catalog)
    print(f"decompressed {rows:,} rows -> {catalog}")
    if rows != EXPECTED_ROWS:
        print(f"ERROR: expected {EXPECTED_ROWS:,} rows, found {rows:,}")
        return 1

    archive.unlink()
    return 0


if __name__ == "__main__":
    sys.exit(main())
