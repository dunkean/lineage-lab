"""Golden histories captured before changes on base 41a573c11ee05e7b6e5e1bfc304369480a134360."""

import hashlib
import json

import pytest

from lineage_lab.config import Scenario
from lineage_lab.engine import generate


def standalone_history_digest(mode, seed):
    archive = generate(
        Scenario(
            population_model=mode,
            seed=seed,
            initial_population=240,
            years=25,
            virtual_settlements=3,
            snapshot_interval=1,
            compute_threads=1,
            events=[
                {
                    "name": "crisis",
                    "start_year": 1010,
                    "end_year": 1012,
                    "extra_mortality": 0.04,
                    "fertility_factor": 0.7,
                }
            ],
        ),
        None,
    )
    if mode == "hybrid":
        keys = (
            "records",
            "factors",
            "final_counts",
            "core_data",
            "core",
            "history",
            "maps",
            "racial_maps",
            "groups",
            "plans",
            "flows",
        )
        facts = {key: archive._payload[key] for key in keys}
    else:
        facts = {
            "people": {k: v.tolist() for k, v in archive.data.items()},
            "unions": archive.union_table.data.tolist(),
            "moves": archive.move_table.data.tolist(),
            "history": archive.overview()["history"],
            "maps": [archive.map_at(y) for y in range(1000, 1026)],
        }
    raw = json.dumps(facts, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


@pytest.mark.parametrize(
    "mode,seed,expected",
    [
        ("full", 42, "9f0f7335956066ca46a2cba0e35afa5f3a28ae9d061f37e6ff94a432d3cce67f"),
        ("full", 123, "6947ed5d58ac1fc0a47248f55cd0a03d33884737b25cd81bd93ba25cb06271cf"),
        ("hybrid", 42, "a3692e21293f19f35d691e918993904f393b3760fdb4ec6c48238a59f5980fac"),
        ("hybrid", 123, "08c4b5f8b0d8cc9c0205cc064c0aa93914e2469e1e62cb65769a9045d92e1351"),
    ],
)
def test_seeded_standalone_histories_are_unchanged(mode, seed, expected):
    assert standalone_history_digest(mode, seed) == expected
