# Population and genealogy guide

Lineage Lab simulates births, deaths, unions, divorces and household migrations year by year. It preserves recorded parents, union episodes, birth/death locations and dated movements. It runs locally without LLM calls or credentials.

## Start and explore

Run `demo.cmd` on Windows or `python -m lineage_lab studio --port 8772` from an installed environment. The default demo uses 2,000 founders, 36 settlements and 60 years. First use may compile Numba kernels.

The configuration view supports presets, seeds, population limits, durations, settlement profiles, demographic rules, ancestries, crises and nations. A blank ancestry field inherits its global setting. The full JSON editor exposes detailed rules, explicit maps, periods and metadata; controls remain locked while unapplied JSON edits exist.

Generation runs in the background. One job can run per server. Progress reports its year and population. A completed result opens automatically if the configuration view is still active; otherwise choose **Explore result**. The active world and two recent alternatives remain available in RAM.

The world view provides a pannable/zoomable map, census timeline, settlement ranking, population/vital-event charts and distributions. The individual view provides the resident atlas, life events and a clickable family tree. Settlement resident lists refer to the final census even when the map displays an earlier year.

## Persistence

Studio generation writes no simulation archives automatically. Exporting a configuration saves settings and seed, not recorded histories. For persistence:

```powershell
.venv\Scripts\python.exe -m lineage_lab generate config/genealogy/fantasy.yaml --output output/fantasy.sqlite
.venv\Scripts\python.exe -m lineage_lab explore output/fantasy.sqlite --port 8772
.venv\Scripts\python.exe -m lineage_lab stats output/fantasy.sqlite
.venv\Scripts\python.exe -m lineage_lab person output/fantasy.sqlite 42 --depth 6
.venv\Scripts\python.exe -m lineage_lab compact output/fantasy.sqlite --output output/fantasy_compact
```

Use a fresh archive path. Compact dense export applies to `full`, not `hybrid`. The server binds to `127.0.0.1`, reads persistent archives without editing them, and validates local browser writes.

## Time and founder uncertainty

`years` overrides `generations * generation_years`. A generation is a configured time interval, not a synchronized birth cohort. Negative calendar years are supported; the internal alive marker is distinct from calendar dates.

The initial census is at `start_year`. Each following year processes:

1. Births in existing unions, including birth spacing and maternal risk.
2. Mortality, including newborn deaths, and union closure.
3. Divorces and new unions, including remarriage delays.
4. Household migrations and activity compatibility.
5. The year-end census.

A person is present in census year `y` when `birth <= y` and `death is null or death > y`. Birth and death can occur in the same year. Parents are alive at the start of their child's birth year but may die during that year.

Founders have an estimated earlier birth year, unknown parents and no invented pre-simulation migration history. Their initial unions are observed at the simulation boundary. Later genealogies remain complete back to these founders in full mode. Kinship checks apply to known ancestry at the configured depth and are never disabled for large populations.

## Settlements

Supply an explicit `settlements` list or use virtual settlement profiles. Explicit settlements define IDs, names, coordinates, type, capacity, initial weight, activities, local ancestry weights and metadata.

Virtual `settlement_types` define `kind`, `share`, `minimum_count`, `capacity`, `initial_weight`, `activities`, `races` and `metadata`. Minimum counts are allocated first; remaining settlements are assigned proportionally. Their final count equals `virtual_settlements` exactly. Profile shares count places, not people. Virtual founder weight is `capacity * initial_weight`; explicit maps use absolute `initial_weight` values.

Layouts are `dispersed`, `clustered` or `grid`. Coordinates and distances use the scenario's units. `capacity_mode: scale` adjusts capacities to the requested world size; `fixed` preserves them. Settlements remain fixed during a run: dynamic town founding and type changes are not implemented.

## Demographic and social rules

Pydantic validates unknown options, missing references, probability ranges and age bounds. Probabilities are annual unless stated otherwise. Maternal mortality is per delivery.

| Setting | Meaning |
| --- | --- |
| `infant_mortality` | Birth-year mortality |
| `child_mortality`, `child_max_age` | Annual childhood mortality and age limit |
| `adult_mortality`, aging parameters, `max_age` | Baseline/age-dependent mortality and hard lifespan bound |
| `fertility_peak`, fertility ages | Conditional fertility for eligible partnered women |
| `birth_spacing` | Minimum interval between deliveries |
| Female/male infertility | Independent lifelong biological states |
| `childfree_rate` | Lifelong preference, checked for both partners |
| Marriage/divorce rates and age limits | Union participation and eligibility |
| `kinship_depth` | Known-ancestry exclusion depth |
| Marriage/migration radius and distance scale | Spatial opportunity and decay |
| Status/activity inheritance | Configurable intergenerational continuity |

Realized fertility depends on unions, age, spacing, reproductive states, economic conditions and regulation. It is not an asserted total fertility rate. Maternal/paternal care and partner/dependent co-residence are reflected in household movements.

## Ancestries, politics and crises

The schema field `races` defines fantasy ancestry profiles and demographic overrides. Explicit crossbreeding rules determine compatible pairings and offspring profiles. Local founder ancestry weights replace global weights where supplied.

Nations have dated foundation/dissolution and capitals. Territory and sovereignty are separate from residence and ancestry. Nation contacts can regulate cross-border marriages and migrations. Explicit territories and annexations override implicit capital-based regions. Political changes do not themselves teleport residents.

Crises can target dates, places and ancestries, adding mortality and modifying fertility/migration. Negative events are not silently cancelled by fertility recovery.

## Population regulation

There is a single simulation, without calibration pilots. Founders remain fixed. If a bounded target is enabled, the maximum is a strict ceiling enforced by mortality; the initial population supplies a recovery floor. The ceiling is not a promise to reach that population. Crises can push population below the floor. Fertility recovery is bounded by biological and social eligibility and suspended during harmful events. Without a maximum, regulation is inactive.

## APIs and extension points

The common archive interface exposes `overview`, `person`, `lineage`, `residence`, `residents`, `resident_atlas`, `map_at`, `settlement_history` and `distributions`. Explorer routes mirror these operations under `/api/`.

Use opaque IDs returned by the hybrid API unchanged. IDs, world configuration and algorithm version define identity. Names currently remain simple display placeholders; no external naming engine is connected.

Full-mode Python rule extensions are available in `rules.py`. The hybrid model does not support arbitrary custom Python rules. See [the hybrid contract](genealogy-hybrid.md) before choosing it for a world.

## Verification and limits

Run `python -m pytest`, `python -m ruff check src tests`, then the documented browser checks. Tests cover seeded histories, kinship, archive parity, regulation, reproduction, politics, spatial counts, cancellation, world selection and both population models.

Full mode retains historical people, including the dead. A complete 50-million-person world across 1,000 years remains a target rather than a verified capability. Settings are fiction parameters rather than a historically validated population model. See [historical measurements](genealogy_review.md).
