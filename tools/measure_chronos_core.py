"""Reproduce the S3 core audit; emits JSON diagnostics, never adjusts parameters.

Run with PYTHONPATH=src using the pinned runtime. Seeds are registered here;
extinction means at least one public census with zero living core founders/descendants.
Budget rejection and projected fallback are counted separately from extinction.
"""

import copy
import json

import numpy as np

from lineage_lab.chronos_bridge import select_core_variant
from lineage_lab.chronos_plan import CONTRACT, translate_plan
from lineage_lab.historical_program import CoreBudgetExceeded
from lineage_lab.hybrid_engine import HybridEngine

SEEDS = (*range(1, 21), 42, 11040611567489494868)
YEARS = (1300, 1325, 1350, 1375, 1399)


def trajectory(archive):
    return [
        [r["year"], r["core_alive"], r["population"] - r["core_alive"]]
        for r in archive.summary["core_coverage"]
        if r["year"] in YEARS
    ]


def fertility_probe(base):
    """Hold 2,000 fertile pairs at age 27 for 200 annual birth transitions.

    No mortality, separation, infertility, childfree choice or capacity pressure;
    newborns are excluded from the next transition's parent pool. Exercise the
    actual birth methods, isolating spacing rather than fitting population paths.
    """
    from lineage_lab.cohorts import Cohorts
    from lineage_lab.config import Scenario
    from lineage_lab.engine import Engine

    source = translate_plan(base).model_dump(mode="json")
    source.update(population_model="full", hybrid=None, initial_population=4000, years=200)
    source["historical"]["events"] = []
    source["historical"]["limits"].update(max_core_living=10000, max_core_historical=1000000)
    source["society"].update(childfree_rate=0, capacity_growth=0)
    source["races"][0]["demography"].update(female_infertility_rate=0, male_infertility_rate=0)
    for place in source["settlements"]:
        place["capacity"] = 1000000
    config = Scenario.model_validate(source)
    engine = Engine(config)
    ids = np.arange(4000)
    engine.data["sex"][ids] = ids % 2
    engine.data["partner"][ids] = ids ^ 1
    engine.data["reproduction_flags"][ids] = 0
    engine.data["last_birth"][ids] = config.start_year - 2
    cohorts = Cohorts(config, config.settlements)
    cohorts.append(config.settlements[0].id, 0, config.start_year - 27, [2000, 2000])
    state = np.array([[2000, 2000, 2000, 0]])
    births_core = births_projected = 0
    for year in range(config.start_year + 1, config.start_year + 201):
        engine.year = year
        engine.alive = ids.copy()
        engine.data["birth"][ids] = year - 27
        births_core += engine._births()[0]
        cohorts.columns()["birth"][:] = year - 27
        cohorts.record_site_pressure(year, state)
        births_projected += int(cohorts.bounds([0], year, state)[2].sum())
    return {
        "pairs": 2000,
        "years": 200,
        "probability": 0.32,
        "spacing": 2,
        "core_births": births_core,
        "projected_births": births_projected,
    }


def audit():
    base = json.loads(CONTRACT.joinpath("village-population-plan.example.json").read_bytes())
    output = {"seeds": SEEDS, "years": YEARS, "cases": {}, "direct_seed_replay": {}}
    for case in ("example", "large_no_shock"):
        rows = []
        for seed in SEEDS:
            plan = copy.deepcopy(base)
            plan["seed"] = str(seed)
            if case == "large_no_shock":
                plan["shocks"] = []
                plan["limits"].update(
                    core_target_share_bps=1000, max_core_living=2000, max_core_historical=20000
                )
                for site in plan["settlements"]:
                    site["initial_population"] *= 10
                    for episode in site["occupation_episodes"]:
                        episode["capacity"] *= 10
            config = translate_plan(plan)
            attempts = []

            def run(candidate, attempts=attempts):
                try:
                    archive = HybridEngine(candidate).run()
                except CoreBudgetExceeded as exc:
                    attempts.append(
                        {"outcome": exc.code, "year": exc.year, "constraint": exc.constraint}
                    )
                    raise
                attempts.append({"outcome": "selected"})
                return archive

            archive, _, policy = select_core_variant(plan, config, run)
            coverage = archive.summary["core_coverage"]
            core_history = archive.core.history if archive.core else []
            rows.append(
                {
                    "seed": seed,
                    "attempts": attempts,
                    "variant": policy["variant"],
                    "trajectory": trajectory(archive),
                    "extinct": archive.summary["core_founders_simulated"] > 0
                    and any(r["core_alive"] == 0 for r in coverage),
                    "core_births": archive.core.summary["births"] if archive.core else 0,
                    "core_deaths": archive.core.summary["deaths"] if archive.core else 0,
                    "projected_births": archive.summary["births"]
                    - (archive.core.summary["births"] if archive.core else 0),
                    "projected_deaths": archive.summary["deaths"]
                    - (archive.core.summary["deaths"] if archive.core else 0),
                    "core_fertile_woman_years": sum(r["fertile_women"] for r in core_history),
                    "core_partnered_fertile_woman_years": sum(
                        r["partnered_fertile_women"] for r in core_history
                    ),
                }
            )
            if seed == int(base["seed"]):
                direct = HybridEngine(config).run()
                output["direct_seed_replay"][case] = trajectory(direct)
        output["cases"][case] = rows
    output["fertility_probe"] = fertility_probe(base)
    return output


if __name__ == "__main__":
    print(json.dumps(audit(), ensure_ascii=False, indent=2))
