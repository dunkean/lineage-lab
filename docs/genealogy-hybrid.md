# Hybrid population model

`full` remains the default. Select **Hybrid** in the studio or set `population_model: hybrid`. The provided example has 2,000 founders and 75 years; it illustrates functionality rather than the hybrid model's break-even point.

```powershell
.venv\Scripts\python.exe -m lineage_lab studio --scenario config/genealogy/hybrid.yaml
.venv\Scripts\python.exe -m lineage_lab generate config/genealogy/hybrid.yaml --output output/hybrid.sqlite
.venv\Scripts\python.exe -m lineage_lab explore output/hybrid.sqlite
```

## What is simulated

The current format is `cohort-runtime-v2`. Earlier experimental hybrid archives are not silently reinterpreted.

One full engine instance simulates the core founders, normally 5% of the initial population. Their births, deaths, unions, ancestry and migrations are recorded. The background consists of integer counts by settlement, ancestry, birth year and sex. Annual cohort updates do not enumerate individual background lives or store their family links.

The summary exposes `core_founders_simulated`, `background_persons_simulated: 0` and `simulation_runs: 1` (or zero when the core budget cannot form a couple). Background cost depends mainly on cohort count: settlement count, ancestries and years, rather than raw inhabitants alone.

The core favors high social ranks using configurable `core_status_weights`. Roles include noble houses, merchant lineages, local notables, innkeepers and commoner heroes. Relatives remain simulated to preserve core families. Annual living and historical core budgets are checked; exceeding a budget rejects the scenario rather than silently dropping recorded people.

## On-demand people

An ordinary resident has a stable ID `h:<world fingerprint>:<cohort>:<slot>`. Cohort zero represents the core. Background IDs denote virtual slots, not pre-created personal objects.

A deterministic rule assigns child ordinals to compatible parental pair ordinals. Bijective permutations distribute men, women and children and can be inverted to find a parent's children. This preserves parent/child reciprocity regardless of lookup order. Birth spacing follows admissible ordinal sequences; infertility and childfree states remain lifelong.

Queries may reconstruct bounded cohort count curves, but never launch another population engine or enumerate all background people. The personal response cache is bounded by `projection_cache_mebibytes`; curve/quota caches are capped at 256 cohorts each.

Background exploration is limited to **two ancestor generations** and **five descendant generations** in both API and UI. Core records retain the full engine's exploration capabilities. Parents before the simulation's founders remain unknown in either mode.

Record identity survives reopening, cache eviction, lookup order changes and concurrent reads for the same configuration and algorithm version. SQLite persistence verifies version and checksum. Full-mode archives remain readable. Compact dense exports are full-mode only.

## Sources and roles

Existing routes accept opaque hybrid IDs. Responses distinguish `source: core|projected`; `/api/core?limit=100&role=noble_house` lists important people. Residents use cursor pagination with the full previous ID in `after`. Atlas counts cover the whole population and materialize only the requested sample.

An explicit anchor references an already simulated core person:

```yaml
hybrid:
  core_target_share: 0.05
  core_max_share: 0.10
  core_historical_max_share: 0.10
  anchors:
    - {segment: 0, person: 17, role: hero, year: 1030}
```

The person must exist and be alive at that date. Changing anchors changes the world identity without changing demographic facts under otherwise identical settings. Arbitrary retrospective promotion of a background slot into a new simulated causal history is not implemented.

## Approximations

Background couples share settlement, ancestry and birth year. They have at most one union; divorce does not produce remarriage. Deaths follow deterministic ordinals. Ordinary activity is the settlement's dominant activity, social rank is zero, and background family identity describes a close family rather than a noble house.

Cohorts account for age rates, sex-specific and maternal mortality, fertility, periods, targeted crises, divorce and population regulation. They omit individual migrations, cross-ancestry offspring, social transitions and the full engine's detailed cousin checks. Core and background do not exchange partners. Custom Python rules are not supported.

Personas of projected people are also approximated. Their values combine bounded hashes of the person, parents and grandparents instead of a recorded inheritance chain. Parent/child resemblance matches the configured heritability, but siblings are too similar and grandparents too distant. Personas report `source: approximated`; core personas report `source: inherited`, or `source: founder` without recorded parents. See [the user guide](genealogy.md#personas).

Hybrid results can therefore differ from full mode at the same seed. Hybrid is not an exact compressed replay of every full-mode life.

## Historical timing sample

An October 1, 2026 measurement used Windows, Python 3.12.13, NumPy 2.5.3, Numba 0.67.0, seed 42, 16 settlements, 60 years and four threads. Times include in-memory construction and hybrid preparation, exclude disk writes and personal lookups, use warm JIT and are medians of three runs.

| Founders | Full | Hybrid |
| ---: | ---: | ---: |
| 2,000 | 0.108 s | 0.139 s |
| 10,000 | 0.205 s | 0.171 s |
| 100,000 | 1.263 s | 0.266 s |
| 1,000,000 | 15.504 s | 0.834 s |

These are historical measurements, not a memory, storage or latency guarantee. Small worlds can be faster in full mode. Reproduce with `python -m lineage_lab.benchmark --compare --population 1000000 --places 16 --years 60 --repeats 3`.
