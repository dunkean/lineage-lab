"""Repeatable exact-history benchmark; RAM by default, explicit SQLite optionally."""

import argparse
import ctypes
import json
import platform
import statistics
import time
from contextlib import closing
from pathlib import Path

import numba
import numpy as np

from .archive_protocol import open_archive
from .config import Scenario
from .engine import generate
from .store import Archive


def peak_rss_bytes():
    if platform.system() == "Windows":

        class Counters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("faults", ctypes.c_ulong)] + [
                (name, ctypes.c_size_t)
                for name in (
                    "peak_rss",
                    "rss",
                    "peak_paged",
                    "paged",
                    "peak_nonpaged",
                    "nonpaged",
                    "pagefile",
                    "peak_pagefile",
                    "private",
                )
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        current = ctypes.windll.kernel32.GetCurrentProcess
        current.restype = ctypes.c_void_p
        query = ctypes.windll.psapi.GetProcessMemoryInfo
        query.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
        if query(current(), ctypes.byref(counters), counters.cb):
            return counters.peak_rss
        return None
    import resource

    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if platform.system() == "Darwin" else value * 1024


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--population", type=int, default=1_000_000)
    parser.add_argument("--places", type=int, default=500)
    parser.add_argument("--years", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", choices=["compiled", "reference"], default="compiled")
    parser.add_argument("--population-model", choices=["full", "hybrid"], default="full")
    parser.add_argument("--output", type=Path, help="Explicit SQLite output; default is RAM only")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--compare", action="store_true", help="Compare warmed full/hybrid construction"
    )
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.compare:
        if args.output or args.backend != "compiled" or not 1 <= args.repeats <= 100:
            parser.error("Comparison needs compiled backend, 1..100 repeats and no output archive")
        warmup = generate(Scenario(initial_population=100, years=2), None)
        warmup.close()
        timings = {}
        for mode in ("full", "hybrid"):
            samples = []
            for _ in range(args.repeats):
                started = time.perf_counter()
                archive = generate(
                    Scenario(
                        initial_population=args.population,
                        virtual_settlements=args.places,
                        years=args.years,
                        seed=args.seed,
                        compute_threads=args.threads,
                        population_model=mode,
                    ),
                    None,
                )
                samples.append(time.perf_counter() - started)
                archive.close()
                del archive
            timings[mode] = {"seconds": samples, "median_seconds": statistics.median(samples)}
        print(
            json.dumps(
                {
                    "founders": args.population,
                    "places": args.places,
                    "years": args.years,
                    "seed": args.seed,
                    "warm_numba": True,
                    "persistence": False,
                    "timings": timings,
                    "construction_reduction": 1
                    - timings["hybrid"]["median_seconds"] / timings["full"]["median_seconds"],
                },
                indent=2,
            )
        )
        return
    summary = generate(
        Scenario(
            initial_population=args.population,
            virtual_settlements=args.places,
            years=args.years,
            seed=args.seed,
            backend=args.backend,
            population_model=args.population_model,
            compute_threads=args.threads,
        ),
        args.output,
    )
    if args.population_model == "hybrid":
        archive = summary if args.output is None else open_archive(args.output)
        summary = archive.summary
        rows = archive.overview()["history"]
        times = []
        for segment, plan in enumerate(archive._payload["plans"][:10]):
            if not plan["people_ever"]:
                continue
            identity = archive._id(segment, plan["people_ever"] - 1)
            archive.clear_cache()
            started = time.perf_counter()
            archive.person(identity)
            archive.lineage(identity, depth=4)
            times.append(time.perf_counter() - started)
        checks = {
            "places": len(archive.settlements),
            "census_balance": sum(
                sum(p["population"] for p in archive.map_at(r["year"])) != r["population"]
                for r in rows
            ),
            "core_budget_violations": sum(
                row["core_alive"] > int(archive.config.hybrid.core_max_share * row["population"])
                or row["core_retained"]
                > int(archive.config.hybrid.core_historical_max_share * row["people_ever"])
                for row in summary["core_coverage"]
            ),
            "cold_person_and_lineage_p95_seconds": float(np.percentile(times, 95)),
            "disk_bytes_written": summary["disk_bytes_written"],
        }
    elif args.output is None:
        memory, summary = summary, summary.summary
        invalid = 0
        for start in range(0, memory.n, 100000):
            end = min(memory.n, start + 100000)
            ids = np.arange(start, end, dtype=np.int32)
            invalid += int(
                np.sum(
                    (memory.data["father"][start:end] >= ids)
                    | (memory.data["mother"][start:end] >= ids)
                )
            )
        checks = {
            "places": len(memory.settlements),
            "invalid_parent_order": invalid,
            "census_balance": sum(
                sum(r["population"] for r in memory.map_at(row["year"])) != row["population"]
                for row in memory.history
            ),
            "disk_bytes_written": summary["disk_bytes_written"],
        }
    else:
        with closing(Archive(args.output).connect()) as db:
            checks = {
                "quick_check": db.execute("PRAGMA quick_check").fetchone()[0],
                "places": db.execute("SELECT count(*) FROM settlements").fetchone()[0],
                "invalid_parent_order": db.execute(
                    "SELECT count(*) FROM people WHERE father>=id OR mother>=id"
                ).fetchone()[0],
                "census_balance": db.execute(
                    "SELECT count(*) FROM census c WHERE population != "
                    "(SELECT sum(population) FROM settlement_census s WHERE s.year=c.year)"
                ).fetchone()[0],
            }
    result = {
        "summary": summary,
        "peak_rss_bytes": peak_rss_bytes(),
        "checks": checks,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "numba": numba.__version__,
            "platform": platform.platform(),
        },
    }
    if args.output:
        args.output.with_suffix(".benchmark.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
