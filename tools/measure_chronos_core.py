"""Reproduce the S3 core audit; emits JSON diagnostics, never adjusts parameters.

Run with PYTHONPATH=src using the pinned runtime. Seeds are registered here;
extinction means at least one public census with zero living core founders/descendants.
Budget rejection and projected fallback are counted separately from extinction.
"""

import argparse
import copy
import json
import platform
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from lineage_lab.chronos_bridge import Worker, select_core_variant
from lineage_lab.chronos_plan import CONTRACT, plan_fingerprint, translate_plan
from lineage_lab.historical_program import CoreBudgetExceeded
from lineage_lab.integrated_population import IntegratedPopulation

SEEDS = (*range(1, 21), 42, 11040611567489494868)
YEARS = (1300, 1325, 1350, 1375, 1399)


def trajectory(archive):
    return [
        [r["year"], r["core_alive"], r["population"] - r["core_alive"]]
        for r in archive.summary["core_coverage"]
        if r["year"] in YEARS
    ]


def fertility_probe(base, probabilities=(0.1, 0.32, 0.7), spacings=(1, 2, 4), years=100):
    """2,000 pairs, actual birth methods, successful cooldown and ten-year pause.

    Age is held at the fertility peak; offspring are excluded from parent pools.
    No parameters are calibrated. The pause covers transitions 41 through 50.
    """
    from lineage_lab.cohort_residence import ResidentRegistry
    from lineage_lab.config import Scenario
    from lineage_lab.engine import Engine
    from lineage_lab.registry_demography import RegistryDemography

    output = []
    for probability in probabilities:
        for spacing in spacings:
            source = translate_plan(base).model_dump(mode="json")
            source.update(
                population_model="full", hybrid=None, initial_population=4000, years=years
            )
            source["historical"]["events"] = []
            source["historical"]["limits"].update(
                max_core_living=10000, max_core_historical=1000000
            )
            source["society"].update(
                childfree_rate=0, capacity_growth=0, founder_match_participation=1
            )
            source["races"][0]["demography"].update(
                female_infertility_rate=0,
                male_infertility_rate=0,
                fertility_peak=probability,
                birth_spacing=spacing,
            )
            for place in source["settlements"]:
                place["capacity"] = 1000000
            config = Scenario.model_validate(source)
            engine = Engine(config)
            ids = np.arange(4000)
            engine.data["sex"][ids] = ids % 2
            engine.data["partner"][ids] = ids ^ 1
            engine.data["reproduction_flags"][ids] = 0
            engine.data["last_birth"][ids] = config.start_year - spacing
            registry = ResidentRegistry(config, "projected")
            founders = registry.add_cohort(
                "founder", 0, 0, config.start_year - 27, ids % 2, flags=np.zeros(4000, np.int32)
            )
            registry.seal_pairs(founders)
            registry.activate_pairs(initial=True)
            registry.partition(founder=True)
            demo = RegistryDemography(registry)
            trajectory = []
            valid_spacing = True
            previous = {}
            for step in range(1, years + 1):
                year = config.start_year + step
                paused = 41 <= step <= 50
                engine.year = registry.year = year
                engine.alive = ids.copy()
                engine.data["birth"][ids] = year - 27
                engine.race_fertility_peak[:] = 0 if paused else probability
                registry.data["birth"][founders] = year - 27
                registry.config.races[0].demography.fertility_peak = 0 if paused else probability
                demo.policy = lambda year, c=registry.config: (
                    [
                        c.demography.model_copy(
                            update=c.races[0].demography.model_dump(exclude_none=True)
                        )
                    ],
                    c.society,
                )
                core_births = engine._births()[0]
                before = registry.n
                mothers, _ = demo.births()
                for mother in mothers:
                    if int(mother) in previous:
                        valid_spacing &= year - previous[int(mother)] >= spacing
                    previous[int(mother)] = year
                trajectory.append([year, core_births, len(mothers)])
                registry.data["site"][before : registry.n] = -1
                del registry.pairs[2000:]
            totals = np.asarray(trajectory)[:, 1:].sum(axis=0)
            output.append(
                {
                    "pairs": 2000,
                    "years": years,
                    "probability": probability,
                    "spacing": spacing,
                    "pause": [1341, 1351],
                    "stationary_births_per_pair_year": probability
                    / (1 + (spacing - 1) * probability),
                    "core_births": int(totals[0]),
                    "projected_births": int(totals[1]),
                    "successful_birth_spacing_valid": bool(valid_spacing),
                    "trajectory": trajectory,
                }
            )
    return output


def reproduction(registry):
    """Observed net reproduction, with all external-coverage censoring explicit.

    Denominator: girls born in coverage early enough for their entire fertility
    window AND their latest possible daughter's adulthood to be observed.
    Numerator: their recorded daughters surviving to minimum fertile age.
    Unknown survival after external departure contributes zero to a lower bound.
    """
    d, start, end = registry.data, registry.start, registry.end
    demos = RegistryPolicy(registry)
    mothers = set()
    for p in range(registry.n):
        demo = demos[int(d["race"][p])]
        latest = end - 1 - demo.fertility_max_age - demo.fertility_min_age
        if d["sex"][p] == 1 and start <= d["birth"][p] <= latest:
            mothers.add(p)
    daughters = unknown = 0
    for mother in mothers:
        for child in registry.children[mother]:
            if d["sex"][child] != 1:
                continue
            mature = int(d["birth"][child]) + demos[int(d["race"][child])].fertility_min_age
            if d["coverage_end"][child] <= mature:
                unknown += 1
            elif d["death"][child] == -(2**31) or d["death"][child] > mature:
                daughters += 1
    return {
        "female_birth_cohort": len(mothers),
        "daughters_surviving_to_fertile_age": daughters,
        "offspring_with_unknown_survival": unknown,
        "mothers_departed_during_fertile_window": sum(
            d["departure"][p] != -(2**31)
            and d["departure"][p] <= d["birth"][p] + demos[int(d["race"][p])].fertility_max_age
            for p in mothers
        ),
        "observed_nrr_lower_bound": daughters / len(mothers) if mothers else None,
        "births": len(registry.births),
        "deaths": len(registry.deaths),
    }


def RegistryPolicy(registry):
    from lineage_lab.cohorts import Cohorts

    return Cohorts(registry.config, registry.places).policy(registry.start)[0]


def core_observations(core):
    if core is None:
        return None
    return {
        "extinct": any(not row["counts"].sum() for row in core.annual.values()),
        "trajectory": [[y, int(core.annual[y]["counts"].sum())] for y in YEARS],
        "reproduction": reproduction(core.registry),
        "fertile_woman_years": sum(int(row[11]) for row in core.rows),
        "partnered_fertile_woman_years": sum(int(row[12]) for row in core.rows),
        "exposure_observation": "annual_closing_census",
    }


def enrich_core_exposure(path):
    """Add raw-core census exposure to an existing audit, without rerunning background."""
    from lineage_lab.chronos_bridge import _attempt_seed
    from lineage_lab.integrated_core import precompute_core

    output = json.loads(path.read_text(encoding="utf-8"))
    base = json.loads(CONTRACT.joinpath("village-population-plan.example.json").read_bytes())
    cache = {}
    for case, records in output["cases"].items():
        for record in records:
            for index, observation in enumerate(record["core_attempt_observations"]):
                if observation is None:
                    continue
                key = (case, record["seed"], index)
                if key not in cache:
                    plan = copy.deepcopy(base)
                    if case == "large_no_shock":
                        plan["shocks"] = []
                        plan["limits"].update(
                            core_target_share_bps=1000,
                            max_core_living=2000,
                            max_core_historical=20000,
                        )
                        for site in plan["settlements"]:
                            site["initial_population"] *= 10
                            for episode in site["occupation_episodes"]:
                                episode["capacity"] *= 10
                    plan["seed"] = str(_attempt_seed(str(record["seed"]), index))
                    core, _ = precompute_core(translate_plan(plan))
                    fresh = core_observations(core)
                    if fresh["trajectory"] != observation["trajectory"]:
                        raise RuntimeError("Core replay differs from audited trajectory")
                    cache[key] = {
                        k: fresh[k]
                        for k in (
                            "fertile_woman_years",
                            "partnered_fertile_woman_years",
                            "exposure_observation",
                        )
                    }
                observation.update(cache[key])
            path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    output["nrr_initial_boundary_note"] = (
        "Birth-year windows include age-zero founders; these are already observed survivors. "
        "NRR is an observed conditional cohort statistic, not an unbiased template calibration estimate."
    )
    return output


def audit(output_path=None, seeds=SEEDS, cases=("example", "large_no_shock")):
    import lineage_lab.integrated_population as population_module

    original_core_provider = population_module.precompute_core
    base = json.loads(CONTRACT.joinpath("village-population-plan.example.json").read_bytes())
    output = {
        "seeds": list(seeds),
        "years": list(YEARS),
        "cases": {},
        "calibration": "none; templates unchanged",
        "nrr_definition": "observed daughter survival to fertile age, completed birth cohorts; lower bound under external censoring",
    }
    for case in cases:
        output["cases"][case] = []
        for seed in seeds:
            core_cache = {}

            def cached_core(candidate, core_cache=core_cache):
                key = (candidate.seed, candidate.historical.limits["core_target_share_bps"])
                if key not in core_cache:
                    core_cache[key] = original_core_provider(candidate)
                return core_cache[key]

            # Reuse only immutable precomputed core tapes across the measurement's
            # regulation-off/on pair. Production Worker jobs never share them.
            population_module.precompute_core = cached_core
            for regulated in (False, True):
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
                    for marker in plan["baseline_markers"]:
                        marker["population"] *= 10
                config = translate_plan(plan)
                attempts = []

                def run(candidate, regulated=regulated, attempts=attempts):
                    world = IntegratedPopulation(candidate, regulation=regulated)
                    try:
                        world.run()
                    except CoreBudgetExceeded:
                        attempts.append(core_observations(world.core))
                        raise
                    attempts.append(core_observations(world.core))
                    return world

                t = time.perf_counter()
                world, _, policy = select_core_variant(plan, config, run)
                rows = world.regulator.rows
                record = {
                    "seed": seed,
                    "regulated": regulated,
                    "variant": policy["variant"],
                    "selected_attempt": policy["selected_attempt"],
                    "attempts": policy["attempts"],
                    "core_attempt_observations": attempts,
                    "trajectory": trajectory(world),
                    "annual_trajectory": world.coverage,
                    "core_extinct": world.summary["core_founders_simulated"] > 0
                    and any(not r["core_alive"] for r in world.coverage),
                    "projected_reproduction": reproduction(world.registry),
                    "core_reproduction": reproduction(world.core.registry) if world.core else None,
                    "pair_exposure": world.demography.exposure,
                    "controller_saturated_site_years": sum(r["saturated"] for r in rows),
                    "controller_correcting_site_years": sum(
                        r["multiplier_ppm"] != 1000000 for r in rows
                    ),
                    "no_eligible_pair_site_years": sum(r["no_eligible_pairs"] for r in rows),
                    "maximum_absolute_guide_error": max(
                        abs(r["error"][0] / r["error"][1]) for r in rows
                    ),
                    "seconds": time.perf_counter() - t,
                }
                # Convert NumPy scalars from measurement-only reductions.
                record = json.loads(json.dumps(record, default=lambda v: v.item()))
                output["cases"][case].append(record)
                print(
                    f"{case} seed={seed} regulated={regulated} variant={policy['variant']}",
                    file=sys.stderr,
                    flush=True,
                )
                if output_path:
                    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
            population_module.precompute_core = original_core_provider
    output["nrr_initial_boundary_note"] = (
        "Birth-year windows include age-zero founders, who are observed survivors; "
        "NRR is a conditional observed-cohort lower bound, not an unbiased calibration estimate."
    )
    return output


def peak_memory():
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    kernel, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(Counters),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(
        kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb
    ):
        raise ctypes.WinError()
    return {
        "peak_working_set_bytes": counters.PeakWorkingSetSize,
        "peak_private_commit_bytes": counters.PeakPagefileUsage,
    }


def benchmark(sites=2, maximum=True):
    plan = json.loads(CONTRACT.joinpath("village-population-plan.example.json").read_bytes())
    if maximum:
        templates = plan["settlements"]
        plan["settlements"], plan["baseline_markers"] = [], []
        for i in range(sites):
            site = copy.deepcopy(templates[i % len(templates)])
            site.update(
                site_id=f"site/maximum/{i:02}",
                provider_local_id=i,
                initial_population=20000 // sites + int(i < 20000 % sites),
            )
            for episode in site["occupation_episodes"]:
                episode["capacity"] = site["initial_population"]
            plan["settlements"].append(site)
            for year in (1300, 1399):
                plan["baseline_markers"].append(
                    {
                        "site_id": site["site_id"],
                        "year": year,
                        "population": site["initial_population"],
                        "tolerance_bps": 2000,
                    }
                )
        ids = [s["site_id"] for s in plan["settlements"]]
        plan["mobility_edges"] = [
            {
                "edge_id": f"edge/maximum/{i}",
                "from_site_id": ids[i],
                "to_site_id": ids[i + 1],
                "valid": plan["timeline"],
                "travel_days": 1,
                "bidirectional": True,
                "mode": "land",
            }
            for i in range(sites - 1)
        ]
        plan["shocks"][0]["site_ids"] = [ids[0]]
        plan["historical_program"][0].update(origin_site_id=ids[0], destination_site_id=ids[1])
        plan["boundary_ports"][0]["entry_site_id"] = ids[0]
        plan["role_demands"][0]["site_id"] = ids[0]
        plan["limits"].update(max_core_living=2000, max_core_historical=20000)
    root = Path("output")
    root.mkdir(exist_ok=True)
    results = []
    with tempfile.TemporaryDirectory(prefix="s3-benchmark-", dir=root) as temp:
        worker = Worker(temp)
        plan["provider"]["runtime_fingerprint"] = worker.runtime_id
        plan["plan_fingerprint"] = plan_fingerprint(plan)
        worker.prepare(plan)
        for label in ("cold", "warm"):
            reply = worker.materialize(plan["plan_id"], plan["plan_fingerprint"])
            path = Path(temp) / reply["job_directory"] / reply["result_ref"]["relative_path"]
            # Avoid loading the large JSON into another Python object graph just
            # to measure it. Complete materialize has already independently validated.
            results.append(
                {
                    "run": label,
                    "seconds": reply["metrics"]["materialize_seconds"],
                    "bytes": reply["result_ref"]["bytes"],
                    "result_fingerprint": reply["result_fingerprint"],
                    "memory": peak_memory(),
                }
            )
            path.unlink()
        # Slot count and selection are returned in the protocol metrics, outside
        # the canonical result and without reparsing a potentially large file.
        results[-1]["candidate_slots"] = reply["metrics"]["candidate_slots"]
        results[-1]["core_selection"] = reply["metrics"]["core_selection"]
    assert results[0]["result_fingerprint"] == results[1]["result_fingerprint"]
    return {
        "sites": sites if maximum else 2,
        "initial_population": 20000 if maximum else 1100,
        "public_years": 100,
        "hardware": {
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "system": platform.platform(),
            "threads": 1,
        },
        "max_candidate_slots": 250000,
        "max_result_bytes": 536870912,
        "runs": results,
        "note": "cold means first materialize in this process; use an empty NUMBA_CACHE_DIR to include fresh JIT",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["probe", "audit", "benchmark", "enrich-audit"], default="audit"
    )
    parser.add_argument("--sites", type=int, choices=[2, 32], default=2)
    parser.add_argument("--example", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument(
        "--cases",
        choices=["example", "large_no_shock"],
        nargs="+",
        default=["example", "large_no_shock"],
    )
    args = parser.parse_args()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    base = json.loads(CONTRACT.joinpath("village-population-plan.example.json").read_bytes())
    if args.mode == "probe":
        result = fertility_probe(base)
    elif args.mode == "benchmark":
        result = benchmark(args.sites, maximum=not args.example)
    elif args.mode == "enrich-audit":
        if not args.output:
            parser.error("--output must name the existing audit")
        result = enrich_core_exposure(args.output)
    else:
        result = audit(args.output, seeds=args.seeds, cases=args.cases)
    content = json.dumps(result, indent=2, default=lambda v: v.item())
    if args.output:
        args.output.write_text(content, encoding="utf-8")
    else:
        print(content)
