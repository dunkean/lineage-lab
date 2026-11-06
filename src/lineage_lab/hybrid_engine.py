"""Simulate the important core once; advance ordinary population as counts only."""

from __future__ import annotations

import hashlib
import json
import math
import time

import numpy as np

from .cohorts import Cohorts
from .config import Scenario, virtual_map
from .historical_program import CoreBudgetExceeded, ProviderError
from .memory import CENSUS_FIELDS, MemoryArchive
from .politics import PoliticalTimeline
from .projection import PROJECTION_VERSION, canonical, derive_seed, pack_arrays, uniforms


def run_core(config, settlements, size, progress):
    if size < 2:
        return None, [[config.start_year + y] + [0] * 13 for y in range(config.duration + 1)]
    from .engine import Engine

    source = config.model_dump(mode="json")
    source.update(
        schema_version=1,
        population_model="full",
        hybrid=None,
        seed=derive_seed(config.seed, "core"),
        initial_population=size,
        target_population=max(
            size, math.floor(config.target_population * size / config.initial_population)
        )
        if config.target_population
        else None,
        capacity_mode="fixed",
        settlements=[
            p.model_copy(
                update={"capacity": max(1, round(p.capacity * size / config.initial_population))}
            ).model_dump(mode="json")
            for p in settlements
        ],
    )
    weights = config.hybrid.core_status_weights
    if weights is None:
        baseline = config.society.status_weights
        weights = (
            [0.12, 0.18, 0.60, 0.10]
            if len(baseline) == 4
            else [w * (i + 1) ** 3 for i, w in enumerate(baseline)]
        )
    if weights is not None:
        source["society"]["status_weights"] = weights
        for period in source["periods"]:
            period["society"].pop("status_weights", None)
    local = Scenario.model_validate(json.loads(canonical(source)))
    activities = sorted({a for p in settlements for a in p.activities})
    archive = MemoryArchive(None, local, local.settlements, activities)
    engine = Engine(local, archive)
    engine.run(progress)
    annual = [list(r) for r in engine.rows]
    summary = engine.summary()
    archive.finish(engine.data, summary, engine.n)
    return archive, annual


def core_arrays(core):
    if core is None:
        return {}
    return {
        **{f"person_{k}": v for k, v in core.data.items()},
        "unions": core.union_table.data,
        "moves": core.move_table.data,
    }


class HybridEngine:
    def __init__(self, config):
        self.config = config
        self.settlements = virtual_map(config)

    def run(self, progress=None):
        from .hybrid_archive import HybridArchive

        started = time.perf_counter()
        config, places = self.config, self.settlements
        if config.historical:
            core_size = (
                config.initial_population
                * config.historical.limits["core_target_share_bps"]
                // 10000
            )
        else:
            core_size = math.floor(config.initial_population * config.hybrid.core_target_share)
        if core_size == 1:
            core_size = 0

        def core_progress(year, count):
            if progress:
                progress(config.start_year, config.initial_population)

        core, core_rows = run_core(config, places, core_size, core_progress)
        core_seconds = time.perf_counter() - started
        background = Cohorts(config, places)
        background.seed_founders(config.initial_population - core_size)
        indices = np.arange(len(background.records))
        state = background.initial(indices)
        survival = np.ones((len(indices), 2))
        initial_deaths = 0
        if config.historical:
            before = state.copy()
            state = background.initial_program(indices, state, survival)
            initial_deaths = int(before[:, :2].sum() - state[:, :2].sum())
        history, maps, races, groups, annual = [], {}, {}, {}, []
        integrated_censuses = []
        births_ever = config.initial_population
        coverage, pending = [], np.zeros(14, dtype=np.int64)
        core_n = core.n if core else 0
        for t in range(config.duration + 1):
            year = config.start_year + t
            births = deaths = marriages = divorces = infants = 0
            if not t:
                deaths = initial_deaths
            if t:
                indices = np.arange(len(background.records))
                before = state.copy()
                if config.target_population and config.target_mode == "bounded":
                    floor = config.initial_population - core_size
                    shortage = max(0, floor - int(state[:, :2].sum()))
                    background.factors[t][0] = min(
                        config.regulation_max_fertility_factor,
                        1 + shortage / max(1, floor) * config.regulation_response_years,
                    )
                if config.historical:
                    background.record_site_pressure(year, before)
                _, _, counts = background.bounds(indices, year, before)
                births = int(counts.sum())
                total = len(places) * len(config.races)
                columns = background.columns()
                codes = columns["place_index"] * len(config.races) + columns["race"]
                newborn = np.bincount(codes, weights=counts, minlength=total).astype(np.int64)
                nonzero = np.flatnonzero(newborn)
                demos = background.policy(year)[0]
                probabilities = np.array([d.male_birth_probability for d in demos])
                males = np.floor(
                    newborn[nonzero] * probabilities[nonzero % len(config.races)]
                    + uniforms(config.seed, nonzero, ["newborn sex", year])
                ).astype(np.int64)
                for code, male in zip(nonzero, males, strict=True):
                    p, race = divmod(int(code), len(config.races))
                    background.append(places[p].id, race, year, [male, int(newborn[code]) - male])
                added = len(background.records) - len(indices)
                if added:
                    new_counts = np.array([r["counts"] for r in background.records[-added:]])
                    before = np.vstack(
                        (before, np.column_stack((new_counts, np.zeros((added, 2), np.int64))))
                    )
                    survival = np.vstack((survival, np.ones((added, 2))))
                indices = np.arange(len(background.records))
                birth_counts = np.pad(counts, (0, added))
                old_survival = survival.copy()
                state = background.transition(indices, year, survival, before, birth_counts)
                if config.target_population and config.target_mode == "bounded":
                    ceiling = max(0, config.target_population - core_rows[t][1])
                    if state[:, :2].sum() > ceiling:
                        lo, hi = 0.0, 1.0
                        for _ in range(24):
                            factor = (lo + hi) / 2
                            background.factors[t][1] = factor
                            test_survival = old_survival.copy()
                            trial = background.transition(
                                indices, year, test_survival, before, birth_counts
                            )
                            if trial[:, :2].sum() <= ceiling:
                                lo = factor
                            else:
                                hi = factor
                        background.factors[t][1] = lo
                        survival = old_survival.copy()
                        state = background.transition(indices, year, survival, before, birth_counts)
                deaths = int(before[:, :2].sum() - state[:, :2].sum())
                marriages = int((state[:, 2] - before[:, 2]).sum())
                divorces = int((state[:, 3] - before[:, 3]).sum())
                if added:
                    infants = int(before[-added:, :2].sum() - state[-added:, :2].sum())
            if config.historical:
                columns = background.columns()
                total_places = len(places)
                places_index = columns["place_index"]
                bg_population = np.bincount(
                    places_index, weights=state[:, :2].sum(axis=1), minlength=total_places
                ).astype(np.int64)
                bg_deaths = np.bincount(
                    places_index,
                    weights=before[:, :2].sum(axis=1) - state[:, :2].sum(axis=1),
                    minlength=total_places,
                ).astype(np.int64)
                bg_births = (
                    np.bincount(
                        places_index,
                        weights=columns["counts"].sum(axis=1) * (columns["birth"] == year),
                        minlength=total_places,
                    ).astype(np.int64)
                    if t
                    else np.zeros(total_places, np.int64)
                )
                core_population = {int(p): int(n) for _, p, n in core.maps[year]} if core else {}
                for slot, place in enumerate(places):
                    births_here = deaths_here = 0
                    if core:
                        d = core.data
                        births_here = (
                            int(((d["birth"] == year) & (d["birth_place"] == place.id)).sum())
                            if t
                            else 0
                        )
                        deaths_here = int(
                            ((d["death"] == year) & (d["death_place"] == place.id)).sum()
                        )
                    simulated = core_population.get(place.id, 0)
                    integrated_censuses.append(
                        {
                            "local_id": place.id,
                            "year": year,
                            "population": simulated + int(bg_population[slot]),
                            "simulated": simulated,
                            "projected": int(bg_population[slot]),
                            "births": births_here + int(bg_births[slot]),
                            "deaths": deaths_here + int(bg_deaths[slot]),
                        }
                    )
            population = int(state[:, :2].sum()) + core_rows[t][1]
            births_ever += births + core_rows[t][2]
            core_ever = core_size + sum(r[2] for r in core_rows[: t + 1])
            if config.historical:
                share_bps = config.historical.limits["core_max_share_bps"]
                live_cap = population * share_bps // 10000
                history_cap = births_ever * share_bps // 10000
            else:
                live_cap = math.floor(population * config.hybrid.core_max_share)
                history_cap = math.floor(births_ever * config.hybrid.core_historical_max_share)
            if core_rows[t][1] > live_cap or core_ever > history_cap:
                if config.historical:
                    raise CoreBudgetExceeded(
                        "Important core exceeds its annual share budget",
                        year=year,
                        constraint="core_max_share_bps",
                    )
                raise ValueError(
                    f"Important core exceeds the annual budget in {year}; lower core_target_share"
                )
            if config.historical and births_ever > config.historical.limits["max_candidate_slots"]:
                raise ProviderError(
                    "BudgetExceeded",
                    "Candidate slot budget exceeded",
                    year=year,
                    constraint="max_candidate_slots",
                )
            coverage.append(
                {
                    "year": year,
                    "population": population,
                    "people_ever": births_ever,
                    "core_alive": core_rows[t][1],
                    "core_retained": core_ever,
                }
            )
            row = np.array(core_rows[t], np.int64)
            row[1] = population
            row[2:6] += [births, deaths, marriages, divorces]
            row[7] += infants
            ages = year - background.columns()["birth"]
            row[8] += int(state[ages < 15, :2].sum())
            row[9] += int(state[(ages >= 15) & (ages < 50), :2].sum())
            row[10] += int(state[ages >= 50, :2].sum())
            fertile = (ages >= config.demography.fertility_min_age) & (
                ages <= config.demography.fertility_max_age
            )
            row[11] += int(state[fertile, 1].sum())
            row[12] += int(
                np.maximum(
                    0,
                    np.minimum(state[fertile, 2], state[fertile, :2].min(axis=1))
                    - state[fertile, 3],
                ).sum()
            )
            row[13] += marriages
            annual.append(row.tolist())
            pending[2:8] += row[2:8]
            pending[13] += row[13]
            if t == 0 or t % config.snapshot_interval == 0 or t == config.duration:
                saved = row.copy()
                saved[2:8], saved[13] = pending[2:8], pending[13]
                span = (
                    1
                    if t == 0
                    else min(
                        config.snapshot_interval, t - (history[-1]["year"] - config.start_year)
                    )
                )
                history.append(dict(zip(CENSUS_FIELDS, [*map(int, saved), span], strict=True)))
                pending[:] = 0
                place_map = {p.id: 0 for p in places}
                race_map = {}
                distributions = {
                    "race": [0] * len(config.races),
                    "age_male": [],
                    "age_female": [],
                    "status": [],
                    "activity": [],
                    "nation": [0] * (len(config.nations) + 1),
                }
                politics = core.politics if core else PoliticalTimeline(config, places)
                owners, _, _ = politics.at(year)
                columns = background.columns()
                totals = state[:, :2].sum(axis=1)
                by_place = np.bincount(
                    columns["place_index"], weights=totals, minlength=len(places)
                )
                place_map = {p.id: int(by_place[i]) for i, p in enumerate(places)}
                codes = columns["place_index"] * len(config.races) + columns["race"]
                by_race = np.bincount(
                    codes, weights=totals, minlength=len(places) * len(config.races)
                )
                race_map = {
                    (p.id, r): int(by_race[i * len(config.races) + r])
                    for i, p in enumerate(places)
                    for r in range(len(config.races))
                }
                distributions["race"] = (
                    np.bincount(columns["race"], weights=totals, minlength=len(config.races))
                    .astype(int)
                    .tolist()
                )
                distributions["nation"] = (
                    np.bincount(
                        owners[columns["place_index"]] + 1,
                        weights=totals,
                        minlength=len(config.nations) + 1,
                    )
                    .astype(int)
                    .tolist()
                )
                for sex, kind in enumerate(("age_male", "age_female")):
                    distributions[kind] = (
                        np.bincount(ages // 5, weights=state[:, sex]).astype(int).tolist()
                    )
                distributions["status"] = [int(totals.sum())]
                activity_ids = []
                activities = sorted({a for p in places for a in p.activities})
                for p in places:
                    activity = max(sorted(p.activities), key=lambda k: p.activities[k])
                    activity_ids.append(activities.index(activity))
                distributions["activity"] = (
                    np.bincount(
                        np.array(activity_ids)[columns["place_index"]],
                        weights=totals,
                        minlength=len(activities),
                    )
                    .astype(int)
                    .tolist()
                )
                if core:
                    for _, place, n in core.maps[year]:
                        place_map[int(place)] += int(n)
                    for _, place, race, n in core.racial_maps[year]:
                        key = int(place), int(race)
                        race_map[key] = race_map.get(key, 0) + int(n)
                    for kind, values in core.groups[year].items():
                        dest = distributions.setdefault(kind, [])
                        dest.extend([0] * max(0, len(values) - len(dest)))
                        for i, n in enumerate(values):
                            dest[i] += int(n)
                maps[str(year)] = sorted(place_map.items())
                races[str(year)] = [[p, r, n] for (p, r), n in sorted(race_map.items()) if n]
                groups[str(year)] = distributions
            if progress:
                progress(year, population)
        metadata = self._importance(core)
        world = hashlib.sha256(
            canonical([PROJECTION_VERSION, config.model_dump(mode="json")]).encode()
        ).hexdigest()
        summary = {
            "population_model": "hybrid",
            "algorithm_version": PROJECTION_VERSION,
            "storage": "hybrid-cohorts-v2",
            "world_id": world,
            "first_person_id": f"h:{world}:0:0" if core else f"h:{world}:1:0",
            "initial_population": config.initial_population,
            "population": annual[-1][1],
            "start_year": config.start_year,
            "end_year": config.start_year + config.duration,
            "people_ever": births_ever,
            "births": sum(r[2] for r in annual),
            "deaths": sum(r[3] for r in annual),
            "marriages": sum(r[4] for r in annual),
            "divorces": sum(r[5] for r in annual),
            "migrations": sum(r[6] for r in annual),
            "core_retained": core_n,
            "core_alive": core_rows[-1][1],
            "core_coverage": coverage,
            "background_persons_simulated": 0,
            "core_founders_simulated": core_size,
            "simulation_runs": int(core is not None),
            "snapshot_interval": config.snapshot_interval,
            "settlement_count": len(places),
            "kinship_rejections": core.summary["kinship_rejections"] if core else 0,
            "kinship_checks": core.summary["kinship_checks"] if core else 0,
            "core_simulation_seconds": core_seconds,
            "simulation_seconds": time.perf_counter() - started,
            "total_seconds": time.perf_counter() - started,
            "disk_bytes_written": 0,
            "target_population": config.target_population,
            "target_relative_error": annual[-1][1] / config.target_population - 1
            if config.target_population
            else None,
            "population_by_race": {
                r.name: sum(
                    n for _, race, n in races[str(config.start_year + config.duration)] if race == i
                )
                for i, r in enumerate(config.races)
            },
        }
        plans = [{"people_ever": core_n}] + [
            {"people_ever": sum(r["counts"])} for r in background.records
        ]
        payload = {
            "version": PROJECTION_VERSION,
            "config": config.model_dump(mode="json"),
            "settlements": [p.model_dump(mode="json") for p in places],
            "summary": summary,
            "core": metadata,
            "core_data": pack_arrays(core_arrays(core)),
            "core_config": core.config.model_dump(mode="json") if core else None,
            "core_summary": core.summary if core else None,
            "records": background.records,
            "factors": background.factors,
            "final_counts": pack_arrays({"counts": state[:, :2]}),
            "plans": plans,
            "history": history,
            "maps": maps,
            "racial_maps": races,
            "groups": groups,
            "flows": {str(row["year"]): core.flows[row["year"]] if core else [] for row in history},
        }
        if config.historical:
            payload["integrated_censuses"] = integrated_censuses
        summary["archive_bytes"] = len(canonical(payload).encode())
        archive = HybridArchive(payload, native_core=core)
        archive.summary["total_seconds"] = time.perf_counter() - started
        return archive

    def _importance(self, core):
        metadata = {}
        if core:
            data = core.data
            adults = np.flatnonzero(
                (data["birth"] <= self.config.start_year - self.config.society.marriage_min_age)
                & (data["status"] <= 1)
            )
            used = set()
            for rule in self.config.hybrid.promotion_rules:
                if rule.role not in {"common_hero", "innkeeper"}:
                    continue
                candidates = adults
                quotas = {None: rule.per_world or 1}
                if rule.per_settlement:
                    quotas = {p.id: rule.per_settlement for p in self.settlements}
                for place, quota in quotas.items():
                    selected = 0
                    for pid in candidates:
                        pid = int(pid)
                        if pid in used or (place is not None and data["birth_place"][pid] != place):
                            continue
                        metadata[f"0:{pid}"] = {
                            "role": rule.role,
                            "priority": rule.priority,
                            "promotion_year": self.config.start_year,
                            "family_score": int(data["status"][pid]),
                            "segment": 0,
                            "person": pid,
                        }
                        used.add(pid)
                        selected += 1
                        if selected >= quota:
                            break
            for anchor in self.config.hybrid.anchors:
                if anchor.segment != 0 or anchor.person >= core.n:
                    raise ValueError(
                        "Required anchor must refer to an existing simulated core slot"
                    )
                raw = core._person(anchor.person)
                year = (
                    anchor.year
                    if anchor.year is not None
                    else max(self.config.start_year, raw["birth"])
                )
                if (
                    year < max(self.config.start_year, raw["birth"])
                    or raw["death"] is not None
                    and year >= raw["death"]
                ):
                    raise ValueError("Required anchor is not alive at promotion")
                metadata[f"0:{anchor.person}"] = {
                    "role": anchor.role,
                    "priority": 1001,
                    "promotion_year": year,
                    "family_score": raw["status"],
                    "segment": 0,
                    "person": anchor.person,
                }
        elif self.config.hybrid.anchors:
            raise ValueError("Required hybrid anchor exceeds the core budget")
        return metadata


def generate_hybrid(config, path, progress=None, rules=()):
    if tuple(rules):
        raise ValueError("Custom Python rules are unsupported by the hybrid cohort model")
    if path is not None:
        path = path.resolve()
        if path.exists() or path.with_suffix(path.suffix + ".partial").exists():
            raise FileExistsError(f"Choose a new output path: {path}")
    archive = HybridEngine(config).run(progress)
    if path is None:
        return archive
    archive.save(path)
    return archive.summary
