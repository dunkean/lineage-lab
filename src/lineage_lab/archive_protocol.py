"""Common explorer contract and read-only archive format dispatch."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Protocol


class GenealogyArchive(Protocol):
    path: Path

    def overview(self) -> dict: ...
    def person(self, identity: int | str) -> dict: ...
    def lineage(self, identity: int | str, depth=4, direction="ancestors", limit=1000) -> dict: ...
    def residence(self, identity: int | str, year: int) -> int | None: ...
    def residents(self, settlement: int, limit=100, after=-1) -> list: ...
    def resident_atlas(self, settlement: int, **filters) -> dict: ...
    def map_at(self, year: int, race=None) -> list: ...
    def settlement_history(self, settlement: int) -> list: ...
    def distributions(self, year: int) -> dict: ...


def open_archive(path):
    from .hybrid_archive import HybridArchive
    from .store import Archive

    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
        row = db.execute("SELECT value FROM metadata WHERE key='population_model'").fetchone()
    return HybridArchive.open(path) if row and row[0] == '"hybrid"' else Archive(path)
