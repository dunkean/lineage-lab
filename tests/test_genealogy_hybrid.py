"""Hybrid facts survive query order, concurrency, eviction and archive reopening."""

import copy
import json
import math
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer
from itertools import pairwise
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest
from pydantic import ValidationError

from lineage_lab.archive_protocol import open_archive
from lineage_lab.config import Scenario, load_scenario
from lineage_lab.engine import generate
from lineage_lab.hybrid_archive import HybridArchive
from lineage_lab.server import make_handler
from lineage_lab.studio import Studio


@pytest.fixture(scope="module")
def world():
    return generate(
        Scenario(
            population_model="hybrid",
            initial_population=240,
            virtual_settlements=3,
            seed=123,
            years=45,
            snapshot_interval=5,
            hybrid={"projection_cache_mebibytes": 1},
        ),
        None,
    )


def identities(world):
    for index, plan in enumerate(world._payload["plans"]):
        for person in range(plan["people_ever"]):
            yield world._id(index, person)


def test_config_defaults_validation_and_explicit_modes():
    assert Scenario().population_model == "full"
    assert Scenario(population_model="hybrid").schema_version == 2
    assert Scenario(population_model="hybrid").hybrid.core_target_share == 0.05
    for source in (
        {"hybrid": {}},
        {"population_model": "hybrid", "backend": "reference"},
        {"population_model": "hybrid", "hybrid": {"core_max_share": 0.11}},
        {
            "population_model": "hybrid",
            "hybrid": {"core_target_share": 0.08, "core_max_share": 0.05},
        },
    ):
        with pytest.raises(ValidationError):
            Scenario.model_validate(source)


def test_demography_and_atlas_cover_all_people(world):
    assert not hasattr(world, "data")  # No retained column per ordinary person.
    summary = world.summary
    assert summary["people_ever"] == summary["initial_population"] + summary["births"]
    assert summary["population"] == summary["people_ever"] - summary["deaths"]
    for row in world.overview()["history"]:
        year = row["year"]
        assert sum(r["population"] for r in world.map_at(year)) == row["population"]
        distribution = world.distributions(year)
        assert sum(r["population"] for r in distribution["groups"]["race"]) == row["population"]
    for place in world.settlements:
        atlas = world.resident_atlas(place.id, limit=7)
        count = next(
            r["population"]
            for r in world.map_at(summary["end_year"])
            if r["settlement"] == place.id
        )
        assert atlas["total"] == atlas["matched"] == count
        assert sum(g["count"] for g in atlas["groups"]) == count
        assert len(atlas["people"]) == min(7, count)
        filtered = world.resident_atlas(place.id, min_age=18, max_age=50, sex=1, limit=5)
        assert sum(g["count"] for g in filtered["groups"]) == filtered["matched"]
        for person in filtered["people"]:
            assert person["sex"] == 1
            assert 18 <= summary["end_year"] - person["birth"] <= 50


def test_core_annual_and_historical_budgets(world):
    settings = world.config.hybrid
    for row in world.summary["core_coverage"]:
        assert row["core_alive"] <= math.floor(settings.core_max_share * row["population"])
        assert row["core_retained"] <= math.floor(
            settings.core_historical_max_share * row["people_ever"]
        )
    assert world.core.n == world.summary["core_retained"]
    for slot in range(world.core.n):
        person = world.person(world._id(0, slot))
        assert person["source"] == "core"
        assert person["importance"]["promotion_year"] >= max(world.start_year, person["birth"])


def test_source_browsing_partitions_the_census_without_changing_simulation(world):
    before = world.overview()
    for place in world.settlements:
        full = world.resident_atlas(place.id)
        core = world.resident_atlas(place.id, source="core")
        projected = world.resident_atlas(place.id, source="projected")
        assert core["matched"] + projected["matched"] == full["matched"]
        assert core["total"] == projected["total"] == full["total"]
        for source, atlas in (("core", core), ("projected", projected)):
            assert sum(row["count"] for row in atlas["groups"]) == atlas["matched"]
            for brief in atlas["people"]:
                person = world.person(brief["id"])
                assert brief["source"] == person["source"] == source
                for relative in world.lineage(person["id"])["people"]:
                    assert world.person(relative["id"])["source"] == source
        history = world.settlement_history(place.id)
        assert history == [
            {"year": row["year"], "population": next(
                p["population"] for p in world.map_at(row["year"])
                if p["settlement"] == place.id
            )}
            for row in before["history"]
        ]
    with pytest.raises(ValueError):
        world.resident_atlas(0, source="unknown")
    with pytest.raises(ValueError):
        world.settlement_history(-1)
    assert world.overview() == before


def test_parent_children_unions_and_shared_identity(world):
    child_count = 0
    for identity in identities(world):
        person = world.person(identity)
        assert person["history_known_from"] >= world.start_year
        for parent_id in (person["father"], person["mother"]):
            if parent_id is None:
                continue
            child_count += 1
            parent = world.person(parent_id)
            assert identity in {child["id"] for child in parent["children"]}
            assert parent["birth"] < person["birth"]
            assert parent["death"] is None or parent["death"] >= person["birth"]
        for union in person["unions"]:
            partner_id = union["male"] if union["female"] == identity else union["female"]
            partner = world.person(partner_id)
            assert union in partner["unions"]
        lineage = world.lineage(identity, depth=4)
        assert len({p["id"] for p in lineage["people"]}) == len(lineage["people"])
    assert child_count > 0


def test_read_order_concurrency_cache_and_reopen(world, tmp_path):
    ids = list(identities(world))[::19]
    expected = {pid: world.person(pid) for pid in ids}
    world.clear_cache()
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(pool.map(world.person, reversed(ids)))
    assert actual == [expected[pid] for pid in reversed(ids)]
    assert world.cache_bytes <= world.config.hybrid.projection_cache_mebibytes * 1024**2
    path = tmp_path / "hybrid.sqlite"
    copy_world = HybridArchive(world._payload)
    copy_world.save(path)
    restored = open_archive(path)
    assert isinstance(restored, HybridArchive)
    assert [restored.person(pid) for pid in ids] == [expected[pid] for pid in ids]
    assert restored.overview()["history"] == world.overview()["history"]
    with pytest.raises(FileExistsError):
        copy_world.save(path)
    with sqlite3.connect(path) as db:
        assert (
            db.execute("SELECT count(*) FROM sqlite_master WHERE name='people'").fetchone()[0] == 0
        )
        db.execute("UPDATE hybrid_plan SET digest='corrupted'")
    with pytest.raises(ValueError, match="checksum"):
        open_archive(path)


def test_projection_versions_identity_and_fact_isolation(world):
    pid = next(identities(world))
    with pytest.raises(ValueError):
        world.person(pid.replace(world.world_id, "0" * 64))
    with pytest.raises(ValueError):
        world.person(pid + "0")  # Noncanonical leading-zero slot.
    with pytest.raises(KeyError):
        world.person(world._id(100000, 0))
    broken = copy.deepcopy(world._payload)
    broken["version"] = "future"
    with pytest.raises(ValueError, match="version"):
        HybridArchive(broken)
    person = world.person(pid)
    person["id"] = "changed"
    person["children"].clear()
    assert world.person(pid)["id"] == pid


def test_stable_resident_pagination(world):
    for place in world.settlements:
        residents, after = [], -1
        while True:
            page = world.residents(place.id, limit=7, after=after)
            if not page:
                break
            residents.extend(page)
            after = page[-1]["id"]
        ids = [person["id"] for person in residents]
        assert len(ids) == len(set(ids)) == world.resident_atlas(place.id)["total"]
        assert all(world.residence(pid, world.summary["end_year"]) == place.id for pid in ids)


def test_anchors_preserve_identity_and_reject_budget_overflow(tmp_path):
    base = Scenario(
        population_model="hybrid",
        initial_population=80,
        years=2,
        hybrid={"promotion_rules": []},
    )
    first = generate(base, None)
    slot = next(
        i for i in range(first.core.n) if first.core.residence(i, base.start_year + 1) is not None
    )
    candidate = [first.person(first._id(0, slot))]
    source = base.model_dump(mode="json")
    source["hybrid"]["anchors"] = [
        {"segment": 0, "person": slot, "role": "hero", "year": base.start_year + 1}
    ]
    second = generate(Scenario.model_validate(source), None)
    person = second.person(second._id(0, slot))
    assert person["source"] == "core"
    assert person["birth"] == candidate[0]["birth"]
    assert person["importance"]["role"] == "hero"
    source["hybrid"]["core_max_share"] = 0
    source["hybrid"]["core_target_share"] = 0
    path = tmp_path / "rejected.sqlite"
    with pytest.raises(ValueError, match="budget"):
        generate(Scenario.model_validate(source), path)
    assert not path.exists()


def test_http_opaque_ids_and_studio_generation(world, tmp_path):
    studio = Studio(world, tmp_path / "runs")
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(world, studio))
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        pid = next(identities(world))
        with urlopen(base + "/api/person?id=" + pid) as response:
            assert json.load(response) == world.person(pid)
        year = world.summary["end_year"]
        with urlopen(base + f"/api/residence?id={pid}&year={year}") as response:
            assert json.load(response) == world.residence(pid, year)
        ordinary = world._id(len(world._payload["plans"]) - 1, 0)
        with urlopen(base + f"/api/lineage?id={ordinary}&depth=12") as response:
            assert json.load(response) == world.lineage(ordinary, depth=2)
        place = world.settlements[0].id
        with urlopen(base + f"/api/settlement-history?place={place}") as response:
            assert json.load(response) == world.settlement_history(place)
        with urlopen(base + f"/api/resident-atlas?place={place}&source=projected") as response:
            assert json.load(response) == world.resident_atlas(place, source="projected")
        with pytest.raises(HTTPError) as error:
            urlopen(base + "/api/person?id=0")
        assert error.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)
    studio.start({"scenario": {"population_model": "hybrid", "initial_population": 40, "years": 2}})
    studio.worker.join(timeout=20)
    assert studio.status()["state"] == "complete"
    key = studio.status()["archive"]
    studio.select(key)
    assert studio.config()["population_model"] == "hybrid"
    restored = Studio(world, studio.output_dir)
    assert key in restored.archives


def test_cancellation_never_publishes_archive(tmp_path):
    def cancel(year, population):
        raise RuntimeError("cancelled")

    path = tmp_path / "cancelled.sqlite"
    with pytest.raises(RuntimeError, match="cancelled"):
        generate(Scenario(population_model="hybrid", initial_population=40, years=3), path, cancel)
    assert not path.exists()
    assert not path.with_suffix(".sqlite.partial").exists()


def test_weighted_mapping_order_is_canonical_before_and_after_persistence(tmp_path):
    source = {
        "population_model": "hybrid",
        "initial_population": 100,
        "years": 30,
        "settlements": [
            {
                "id": 7,
                "name": "Port",
                "x": 0,
                "y": 0,
                "activities": {"trade": 3, "craft": 2, "agriculture": 1},
            }
        ],
    }
    first = generate(Scenario.model_validate(source), None)
    source["settlements"][0]["activities"] = {"agriculture": 1, "craft": 2, "trade": 3}
    second = generate(Scenario.model_validate(source), None)
    assert first.world_id == second.world_id
    assert first._payload["core_data"] == second._payload["core_data"]
    path = tmp_path / "ordered.sqlite"
    first.save(path)
    restored = open_archive(path)
    for pid in identities(first):
        assert restored.person(pid) == first.person(pid)


def test_fantasy_periods_events_and_bounded_regulation(tmp_path):
    from pathlib import Path

    source = load_scenario(Path("config/genealogy/fantasy.yaml")).model_dump(mode="json")
    source.update(
        population_model="hybrid",
        initial_population=100,
        target_population=120,
        years=12,
        hybrid={},
    )
    source["periods"] = [
        {"start_year": source["start_year"] + 5, "demography": {"fertility_peak": 0.25}}
    ]
    source["events"] = [
        {
            "name": "Bad harvest",
            "start_year": source["start_year"] + 3,
            "end_year": source["start_year"] + 5,
            "mortality_factor": 1.1,
        }
    ]
    config = Scenario.model_validate(source)
    world = generate(config, None)
    assert sum(world.summary["population_by_race"].values()) == world.summary["population"]
    for row in world.summary["core_coverage"]:
        assert row["population"] <= 120  # Recovery floor can be unmet after a crisis.
    path = tmp_path / "fantasy.sqlite"
    world.save(path)
    restored = open_archive(path)
    for pid in list(identities(world))[::13]:
        assert restored.person(pid) == world.person(pid)
    assert all(restored.map_at(year) == world.map_at(year) for year in world.saved_years)


def test_only_core_uses_native_engine_and_queries_never_simulate(monkeypatch, tmp_path):
    from lineage_lab import engine

    native = engine.Engine
    simulated = []

    def construct(config, archive):
        simulated.append(config.initial_population)
        return native(config, archive)

    monkeypatch.setattr(engine, "Engine", construct)
    world = generate(Scenario(population_model="hybrid", initial_population=400, years=30), None)
    assert simulated == [20]
    assert world.summary["background_persons_simulated"] == 0
    assert world.cache_bytes == 0

    def forbidden(*args, **kwargs):
        raise AssertionError("A procedural query must never start a native simulation")

    monkeypatch.setattr(engine, "Engine", forbidden)
    path = tmp_path / "no-replay.sqlite"
    world.save(path)
    restored = open_archive(path)
    pid = restored._id(len(restored._payload["plans"]) - 1, 0)
    restored.person(pid)
    restored.lineage(pid, depth=2)


def test_procedural_depth_limits_and_rank_one(world):
    pid = world._id(len(world._payload["plans"]) - 1, 0)
    assert world.lineage(pid, depth=12) == world.lineage(pid, depth=2)
    assert all(p["generation"] == 1 for p in world.lineage(pid, depth=1)["people"])
    root = world._id(1, 0)
    assert world.lineage(root, depth=12, direction="descendants") == world.lineage(
        root, depth=5, direction="descendants"
    )


def test_background_spacing_and_stable_reproductive_traits(world):
    for index, record in enumerate(world.background.records):
        if not sum(record["counts"]):
            continue
        pid = world._id(index + 1, 0)
        p = world.person(pid)
        years = sorted(c["birth"] for c in p["children"])
        assert all(b - a >= world.background.spacing for a, b in pairwise(years))
        if p["infertile"] or p["childfree"]:
            assert not p["children"]
