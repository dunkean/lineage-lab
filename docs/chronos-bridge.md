# Chronos bridge, complete S3 registry

`Worker.materialize` now publishes **`population-result-json-2`**, with
`result_type="PopulationResult"`, `complete=true`, and independently checked
exported tables. Format 2 holds the facts of format 1 in columns: each long or
repeated text is an index into `strings`, where U+0001 stands for `domain_id`
(`compact_result` and `expand_result` in `population_export.py`). The tables
are validated in row form, before compaction. The protocol remains `chronos-lineage-1`; `prepare` announces
complete coverage. Publication is atomic as `population-result.json`. Runtime
and result fingerprints must be repinned; the bundled input schema/example and
requirements lock have not changed. The stage-1 notes below are historical
implementation records, not the current worker's result contract.

## Complete result and Rust integration

The top-level tables are:

- `candidate_table`: columns sorted by `origin_id` UTF-8 bytes. `source` is
  `simulated` or `projected`; ancestry, sealed origin/sex ordinal, original
  `family_key`, birth/death facts, coverage, departure facts, parent references,
  reproductive exclusions and established core attributes are explicit.
- `origin_cohorts`: immutable sealed origin-sex universes, including arrivals.
- `residence_index` and `residence_offsets`: episodes sorted by candidate row,
  year and phase; offsets have `candidate_slots + 1` entries.
- `household_episodes`, `union_allocations`, `guardian_assignments`:
  resident membership and union intervals, formation sources, declared founder
  custody and synthetic care, without invented founder biological parentage.
- `birth_allocations`, `death_allocations`, `parentage`, `children_index`:
  reciprocal links; a birth names its original parental union and child origin
  ordinal, regardless of where the mother moved.
- `residence_transfers`: one frozen member list per actual household move, with
  its operation or statistical port. Local endpoints represent the same event.
  Statistical arrivals have no named reservoir person or bridge identity.
- `annual_censuses` and `cohort_residence_counts`: all public years, counts by
  site/component and sparse origin/current-site/sex counts.
- `diagnostics` and `validation_proof`: exact controller ratios, migration
  realizations, core attempts/coverage, partnership exposure, bounded lineage
  scope, and independent validation results.

Person references are **zero-based rows in the canonical candidate table**,
with JSON null for unknown references. Public identity is a lossless JSON-array
address encoded as a string, never a table position or a hash. Row numbers may
change when composing an export; identities, family keys and links do not.

Intervals compare `(year, phase)` lexicographically. The ordinary half-open
annual `valid` fields are supplemented by `start_phase` and `end_phase`.
A coverage end uses phase 0. Do not discard equal-year birth/death intervals or
visits: phase 20 birth precedes phase 30 death and phase 40 movement. Annual
censuses observe phase 80. `departed_external` closes local coverage, preserves
identity, and leaves subsequent death unknown. A widow's household ID survives
membership-episode closure. Synthetic care is stationary.

Rust should verify the file SHA-256 and byte size, then the result fingerprint
(the canonical object excluding `result_fingerprint`), and independently check
the tables before assigning social attributes or offices. The Python validator
is exposed as `population_export.validate_population_result(result, limits)`;
it recomputes balances, sparse counts, reciprocal family references, residence
and household membership from exported facts, without simulation counters.
`validation_proof` is evidence of that executed check, not permission for the
host to skip its own validation. Metrics are outside canonical content.

## Execution and versions

`IntegratedPopulation` is the production integrated entry point. The standalone
full and lazy hybrid implementations, including their seeded golden histories,
remain unchanged. `partial_result` retains the stage-1 diagnostic path for
regression comparison; it is not used by the worker and does not provide the
registry contract.

The integrated implementation separates immutable slot/residence storage
(`cohort_residence.py`), projected transitions (`registry_demography.py`), core
precomputation/tape (`integrated_core.py`), movement (`registry_mobility.py`),
exact site regulation (`site_regulation.py`), and export/validation
(`population_export.py`). Translation retains operations, statistical ports,
inline kinship/site profiles, graph mode/edge identity, recovery metadata and
regulator settings. There is no Chronos repository import or new dependency.

The manifest declares `chronos-resident-registry-1`,
`chronos-annual-present-mortality-2`, `chronos-successful-birth-spacing-1` and
`chronos-residual-rational-1`. Annual projected death quotas use current present
homogeneous groups, stochastic `floor(N*p+u)` rounding, and a year-keyed uint64
bijection of immutable sex ordinals, selected with serial fixed-width radix
passes. Maternal risk applies only to actual mothers. Core demographic facts
remain Engine facts, with integrated household commits and a phase-stamped
birth/death/union/household/transfer tape.

Projected births use active co-resident unions from the preceding publication,
per-ancestry successful-birth spacing, actual parental slots and the mother's
current site. No union or birth is created to repair a demographic target.
The regulator uses exact fractions and half-up ppm quantization. It observes
only the previous publication; suspends through `end_year + recovery_years`
inclusive; and rebases after that year's transfers. A zero baseline remains
neutral until its first positive phase-80 anchor. Migration is incorporated
once through signed `M` and `M_ref`, including core transfers. Saturation,
negative raw guides, core above guide and unavailable fertility remain visible.

Only `CoreBudgetExceeded` advances a core attempt. An extinct compliant
`required_core` is accepted; only a core without any founders is refused.
Malformed UTF-8 lines, isolated-surrogate request IDs and ill-typed materialize
plan IDs return validation envelopes without destroying prepared plans.
Bootstrap diagnostics judge composed effects, including mutually cancelling
mortality multipliers.

## Explicit implementation choices and measured limitations

Founder dependants are ordered by birth year then semantic ID. Compatible
households use least load then semantic household ID; every age-compatible
adult in the chosen household is a declared guardian. Foster limits are per
household and taken from the site's inline profile. No later foster adoption
is invented. Incoming statistical household sizes/ancestries use inline weights;
up to two adults have uniform ages in the declared range and form one opposite-
sex union; remaining members are dependants with unknown biological parents and
explicit guardians. No exterior marriage is simulated. These are construction
choices where the input profile does not provide a complete individual model.

There are no annual births, ordinary migration or rate-port transitions at the
initial boundary: only founder construction, the composed initial shock delta,
marriage relocations and explicitly dated operations. Ordinary projected
migration uses the resolved cultural rate once, historical modifiers, opening
household denominators and carried fractional remainders. Unserved integer
demand is diagnostic, not perpetual debt. Incoming and marriage households
cannot move again in the phase.

Slots and scan/grouping/death-order kernels use NumPy/int32 and serial Numba.
Household structural bookkeeping remains Python. Bootstrap least-load selection
and semantic export ordering use heaps/comparison sorting, adding `O(L log L)`
and `O(E log E)` terms; this is **not** a claim of strict linear total work.
Storage is slots plus emitted episodes/events, without a person-by-year matrix.
The result-file byte budget is distinct from process working set/private commit;
the measurement tool reports both, and does not claim a 512 MiB RAM guarantee.

Reproduce measurements with the prescribed Python and `PYTHONPATH=src`:

```powershell
python tools/measure_chronos_core.py --mode probe --output output/s3-fertility-probe.json
python tools/measure_chronos_core.py --mode audit --output output/s3-seed-audit.json
python tools/measure_chronos_core.py --mode benchmark --example --output output/s3-example-benchmark.json
python tools/measure_chronos_core.py --mode benchmark --sites 32 --output output/s3-maximum-32-benchmark.json
```

Use a fresh workspace-local `NUMBA_CACHE_DIR` for cold JIT measurements. Benchmarks
include simulation, validation, canonical serialization and atomic publication;
worker startup and prepare are excluded. The audit registers seeds 1–20, 42 and
11040611567489494868, compares regulation disabled/enabled, reports attempted-
core extinction separately from budget rejection/fallback, complete annual
trajectories and controller saturation. Observed net reproduction counts girls
born early enough for their entire fertility window and their latest daughter's
adulthood to be observable. External coverage losses make its reported NRR a
lower bound; censoring counts are explicit. No template has been recalibrated.

# Historical stage-1 notes


Run the local worker with the pinned environment:

```powershell
$env:NUMBA_CACHE_DIR = '.numba-cache'
python -m lineage_lab.chronos_bridge --runtime
python -m lineage_lab.chronos_bridge --job-root output/chronos-jobs
```

`--runtime` reports the runtime manifest and its fingerprint. Normal operation
reads one JSON object per stdin line and writes one response per stdout line.
Both streams explicitly use strict UTF-8, independent of the Windows locale
and `PYTHONIOENCODING`; the subprocess regression sends `é` in the plan and
`東京` in the request ID under a forced cp1252 environment.
Provider output goes to stderr. The host supplies a private job root. Each
successful materialization gets a separate directory beneath it. No Chronos
repository imports, network requests, or extra Python dependencies are needed.
The bundled schema, example and `requirements.lock` are package data with frozen
byte fingerprints tested in `tests/test_chronos_bridge.py`.

Example envelopes (replace the example placeholders before execution):

```json
{"protocol":"chronos-lineage-1","request_id":"p1","method":"prepare","plan":{"...":"complete PopulationPlan"}}
{"protocol":"chronos-lineage-1","request_id":"m1","method":"materialize","plan_id":"pop/demo/domain-0/v1","payload_ref":"<plan_fingerprint>"}
```

`prepare` returns `result.payload_ref`, the runtime fingerprint and the partial
result format. The `plan` field is the full plan, not a file path. The optional
envelope `plan_id` must match it. `materialize` requires that exact prepared
fingerprint. Request IDs, paths, job ordering and elapsed times do not enter the
simulation seeds. Each request is processed synchronously; concurrent calls to
`Worker.handle` receive `ProviderUnavailable` while another job is running.

Successful `materialize` responses contain `job_directory` relative to the host's
job root, and `result_ref {relative_path, file_sha256, bytes, format}` relative to
that job directory. Resolve both beneath the host-owned root, reject escapes,
and verify size and SHA-256 before reading. The worker chooses the filename;
requests cannot supply result paths. Publication uses a flushed staging file
and atomic replacement. Failed publications remove the staging file and their
private directory. Engine exceptions during `materialize` are provider failures
(`ProviderUnavailable`), not plan-validation failures.
Durations appear only in the envelope's `metrics`.

The artifact uses **`chronos-lineage-partial-json-1`**, with
`result_type=PartialPopulationDiagnostics`, `coverage=partial`, `complete=false`
and `stage=S3-step-1`. It is not `population-result-json-2` or a complete
`PopulationResult`; it must not pass the host's `VerifiedPopulation` gate.
It includes:

- Provider/runtime manifests and the complete applied configuration.
- Annual post-transition counts for each site, with disjoint simulated and
  projected counts, births and deaths. Site migration is deferred until an
  independently recorded movement register exists in step 2; no residual is
  presented as a measured flow. The domain birth/death balance is tested.
- Initial shock delta, annual core budgets, selected attempt and diagnostics
  listing every deferred operation and port.

Canonical JSON accepts only integers, strings, booleans, null, lists and objects.
Keys follow UTF-8 order, without whitespace, BOM or unnecessary escaping. The
host supplies canonical list ordering; episodes and registers retain their
explicit order. `plan_fingerprint` excludes `plan_id` and itself;
`result_fingerprint` excludes itself. The file SHA-256 includes the entire file.
Applied binary64 settings are represented as `{ "binary64": "<float.hex()>" }`
inside this integer-only manifest; normalized input profiles are also retained.
The runtime includes package source/data hashes, the bundled lock hash, actual
dependency versions, Python, OS/architecture, CPU target and thread count. It
contains no hostname, username, absolute path or processor serial number.
The first target is generic CPU, with no extra CPU features, and one thread.

`chronos_plan.validate_plan(plan, runtime_fingerprint)` performs schema and
semantic validation. `translate_plan(plan)` produces a validated `Scenario`;
call validation first. Recognized profile keys are frozen explicitly, with the
normative `_bps`, `_ppm`, `_mm` and `_years` conversions. Effect names use the
separate §26.4 mapping. Society IDs are recomputed from resolved parameters and
weights; social weights follow contiguous provider ranks. External pack byte
verification and resolution of culture variants to this single society remain
host-owned: the worker has references, not the external packs. It validates
inline references and syntax and never fetches a catalogue.

Standalone scenarios do not activate this extension. `Scenario.historical` is
explicit, omitted from serialized standalone settings when absent, and rejects
standalone targets/events when active. Golden histories were captured **before
engine edits**, on base `41a573c11ee05e7b6e5e1bfc304369480a134360`, for seeds 42 and
123 in both full and hybrid modes. The digests cover people/core arrays, unions,
moves, annual censuses, maps and cohort records.

Soft shocks are already locally exposed. Their inclusive translated intervals
are shared by the core and cohorts; no exposure is drawn again. Multipliers
compose in source-event UTF-8 order, followed by independent additional survival.
The `apply_initial_program` guard runs once per constructed component before
its first census. Cohort reconstruction reads the same initial delta and saved
annual capacity pressures. It cannot derive pressure from a queried subset.
Explicit episode capacities are multiplied by
`(1 + society.capacity_growth) ** (year - start_year)`, with the standalone
semantics on both components. Only the standalone target and initial floor are
disabled. Integrated founder and share thresholds use integer multiplication
and division by 10,000, including the zero-share fallback's applied limits.

Choices where the design leaves a stage-1 detail open:

- The dedicated partial format above prevents accidental acceptance as a full
  provider result. The `job_directory` envelope field makes the relative path's
  resolution base explicit.
- Event multipliers retain Lineage Lab's cumulative-hazard semantics:
  `q_modified = 1 - (1 - q_base)^factor`. Independent extra mortality is composed
  afterward. Maximum lifespan remains a hard bound.
- Founders are pre-shock survivors. Bootstrap applies additional mortality and
  the excess mortality hazard `max(0, factor - 1)`, without an extra ordinary
  annual mortality pass or bootstrap births. For a shock confined to `start_year`,
  fertility/capacity/migration multipliers and mortality multipliers below one
  have no annual transition on which to act. Nonneutral effects are listed with
  source event, sites, integer input values and reason in
  `diagnostics.initial_delta.effects_without_consequence`. Longer episodes can
  affect subsequent simulated years.
- Attempt seed `i` is the first eight SHA-256 bytes, little endian, of canonical
  `["chronos-lineage-core-attempt-1", seed_decimal_string, i]`. The selected
  scenario uses that seed; the standalone provider's core substream derivation
  is retained. The rule is versioned and part of the provider source fingerprint.

`free_try3_then_projected` declares exactly four attempts: target share at indices
0, 1 and 2, then zero share at index 3. Only the typed `CoreBudgetExceeded`
exception advances this policy. Slot/cohort/result-byte/memory failures and
unsupported biology do not activate it. An extinct compliant core is accepted.
`required_core` allows the declared attempt count, at most eight, with no
zero-share fallback. A core with no founders in this policy raises
`PopulationConstraintsUnsatisfied` before publication; extinction within budget is
accepted on the first successful attempt. An empty founder count
is not advertised as `target_core`. Free empty variants declare `projected_only`
and explicitly exclude simulated and long hereditary lineages in `lineage_scope`.
This scope is bounded by provider coverage and the declared projection depths.

Step 2 must implement the residence and household registries, founder guardian
assignments using the **inline** foster limit, household movements, statistical
ports, per-site residual regulation and full bounded candidate/kinship export.
In this artifact the baseline remains unmodified input, all target correction is
disabled, ordinary role demands are validated planning inputs only, and deferred movement
and port IDs are explicitly reported. Background residence remains its origin
site for this step. Only `historical_soft_v1` is advertised as executed; accepting
ordinary S3 planning fields does not claim their deferred capabilities. Valid
S6 conditional/exact fields, S7 bridges, S8 dynamic settlements/procedural modes,
unsupported reproduction and `role_demands[].policy="required"` are rejected
with `UnsupportedCapability`.

The former `test_marker_does_not_anticipate_shock` did not exercise a marker
consumer: `HistoricalProgram.baseline` has none in step 1. That acceptance test
is explicitly deferred to step 2 with the site regulator. The narrower
`test_future_shock_does_not_change_realized_past` checks the actual materialize
path before a future shock. Other regression checks read actual core deaths,
use partial bootstrap mortality with factors below and above one, exercise
the real annual share ceiling, and replay two capacity episodes by cohort in
reverse query order.

## Core audit, correction pass 1 (2026-11-06)

Reproduce with the pinned interpreter and `PYTHONPATH=src`:

```powershell
$env:NUMBA_CACHE_DIR = '.numba-cache'
& D:/workspace/self/chronos/.venv/Scripts/python.exe tools/measure_chronos_core.py
```

The registered seeds are **1 through 20, 42, 11040611567489494868** (22 total).
The last is the example's population seed, distinct from its world seed 42.
Each run covers 1300–1399. The example keeps the supplied shock and budgets;
the large case multiplies initial populations and capacities by ten, removes
shocks, sets target/max core shares to 1,000 bps, and declares absolute budgets
2,000 living / 20,000 historical core people. No other demographic parameters
are changed. This is a fixed diagnostic sample, not a calibration or a bound
on extinction probability.

Measurements were taken before the behavioral corrections and repeated after
them. The original published example is reproduced: **55 → 6** simulated;
1,045 → 1,627 projected before the capacity-growth correction, 1,628 afterward.
The direct engine large case with the original population seed reproduces
**1,100 → 510 simulated; 9,900 → 15,183 projected**. Calling the worker uses
the indexed attempt seed instead: this same large plan selects attempt 0 and
ends at 766 / 15,262. Direct and worker paths must not be conflated.

Across the 22 actual worker selections, extinction (at least one census with
zero living core, conditional on nonempty founders) is **0/22 in each case**;
projected-only fallbacks are **0/22**. All 22 small-case first attempts succeed.
The large case rejects **1/23 attempted core variants (4.35%)**: seed 17,
attempt 0, year 1301, annual relative share ceiling; attempt 1 succeeds. Thus
1/22 jobs needs a retry, and 0/22 jobs ultimately fails. These counts are the
same before and after the corrections. No extinction observed over this horizon
means neither survival indefinitely nor a healthy replacement rate.

Post-correction trajectories below are means of the selected variants; core
ranges show the full seed sample. Counts are disjoint simulated/projected
people, not interchangeable estimates of one population.

| Year | Example core mean [min, max] | Example projected mean | Large core mean [min, max] | Large projected mean |
|---|---:|---:|---:|---:|
| 1300 | 55.00 [55, 55] | 1,045.00 | 1,100.00 [1100, 1100] | 9,900.00 |
| 1325 | 44.36 [29, 58] | 1,166.55 | 1,012.32 [940, 1069] | 10,849.91 |
| 1350 | 33.23 [15, 55] | 1,383.77 | 898.36 [819, 992] | 12,176.82 |
| 1375 | 19.45 [5, 37] | 1,364.95 | 776.05 [670, 873] | 13,711.27 |
| 1399 | 12.23 [1, 24] | 1,612.09 | 673.27 [560, 790] | 15,212.09 |

The original population seed on the worker path follows:

| Year | Example simulated / projected | Large simulated / projected |
|---|---:|---:|
| 1300 | 55 / 1045 | 1100 / 9900 |
| 1325 | 49 / 1169 | 1067 / 10843 |
| 1350 | 35 / 1403 | 992 / 12177 |
| 1375 | 23 / 1399 | 873 / 13748 |
| 1399 | 6 / 1628 | 766 / 15262 |

The divergence has an established fertility mechanism, independent of shocks,
capacity and the small population's matching noise. `Engine._births` and
`kernels.fertile_ids` enforce a delay after a **successful** birth. In contrast,
`Cohorts.bounds` rotates approximately `1 / spacing` of paired ordinals each
year and multiplies their probability by `spacing`. A failed birth opportunity
still waits for the next rotation, while the probability compensation largely
removes the aggregate spacing penalty. With a constant annual probability p
and spacing s, the core's stationary rate is `p / (1 + (s - 1) * p)`; the
projection's is `min(p, 1 / s)` before other factors. For p=0.32, s=2, these
are approximately 0.2424 and 0.32 births per pair-year.

The audit includes a controlled probe through both actual birth methods:
2,000 pairs held at age 27 for 200 transitions, with no deaths, infertility,
childfree choice, separation, capacity pressure or newborns entering the parent
pool. It produces **97,017 core births versus 128,000 projected births**
(0.2425425 versus 0.32 per pair-year). This identifies a systematic projection
approximation defect in **`cohorts.py:Cohorts.bounds`**, not a second shock
application or an extinction-budget defect. The probe does not attribute every
person in the century-long discrepancy to spacing alone.

Other model differences remain explicit. Real core unions can be absent or
closed, matching uses age gaps/status/kinship, and both partners' reproduction
flags must allow births. Cohorts approximate unions using paired counts and
use rounded cohort-level trait exclusions, which differ from individual flags.
The original example has only **332 partnered fertile woman-years out of 717**,
and 48 births versus 97 deaths. Matching scarcity and demographic randomness
therefore amplify its decline; they are not explained by its plague alone.

No fertility, pairing, mortality or budget tuning is included in this pass.
Repairing the projection's birth-spacing semantics requires a separate,
versioned decision about allocation/reconstruction and the stage-2 union
registry; silently changing it would also break preserved standalone hybrid
histories. The requested capacity-growth correction is implemented independently.

## Initial delivery validation (before correction pass 1)

Validation measured on 2026-11-06: installed wheel, run from outside the checkout,
using the prescribed Chronos Python 3.13.9 environment, Windows 11, Intel Core
i9-12900K (24 logical CPUs), 127.82 GiB RAM; generic CPU target, one compute
thread. The unshortened example covers two sites and 100 public annual states.
After replacing the runtime and plan-fingerprint placeholders, `materialize`
took **5.603624 s cold** (empty dedicated JIT cache) and **0.172787 s warm**
(second job in the same worker). Worker startup and `prepare` are excluded;
compilation, simulation, canonical serialization and atomic file publication
are included. Both produced identical 59,132-byte canonical files and selected
attempt 0. These timings concern the partial artifact, with the deferred work
listed above; they are not a measurement of the complete S3 provider.

Final checks: `pytest -q -p no:cacheprovider --basetemp=.pytest-tmp` passed
169 tests in 56.48 s; `ruff check src tests` and `git diff --check` passed.
The wheel was built with the already installed system setuptools, then installed
with `--no-deps --no-compile --target` inside the worktree and exercised from
`C:/Windows` with the prescribed interpreter. The Chronos environment was not
modified.

## Correction pass 1 verification

The new regressions were run against the uncorrected implementation first:
growth was identical with or without `capacity_growth_ppm`, bootstrap diagnostics
were missing, required roles and empty required cores were accepted, the empty
variant claimed `target_core`, materialize exceptions were misclassified,
the integer founder boundary returned 28 instead of 29, non-ASCII subprocess
round-trips failed, and a publication failure left an empty job directory.
The removed migration-residual assertion also failed its replacement contract.

With the prescribed Python 3.13.9 interpreter and commands, the final full suite
reports **184 passed, 1 failed in 122.52 s**. The failure is
`test_cancelled_job_does_not_publish_archive_and_can_retry`: after its 10-second
join, the asynchronous job still reports `running`. That test passes alone
(1 passed in 5.86 s), and its entire existing studio module passes alone
(7 passed in 26.31 s). Its full-suite timing failure remains unresolved; it is
not counted as a successful check. An additional bridge/studio diagnostic run
with `faulthandler_timeout=8` aborted with a Windows access violation while
dumping thread stacks; it supplies no reliable causal diagnosis of the timeout.
The final bridge plus standalone golden subset passes **77/77 in 9.72 s**.
`ruff check src tests`, a separate lint of the audit script, and
`git diff --check` pass. Seeded standalone full/hybrid golden histories remain
unchanged. No packaging or dependency changes were made in this pass.

The corrected checkout's full example was also measured in a fresh process on
the Intel Core i9-12900K / Windows 11 machine, generic CPU target, one compute
thread: **6.807521 s cold**, using an empty dedicated Numba cache, then
**0.299386 s warm** in the same worker. Both canonical artifacts were identical,
53,738 bytes (SHA-256
`1d896cf6626f042343aa3a42a83d9f267188a970c38a3baec8b6dd862dd7caff`).
These times exclude worker startup and prepare, and include simulation,
serialization and atomic publication; they cover the partial stage-1 artifact.

## Local re-pairing

`chronos-local-repairing-1` is published as `provider_manifest.repairing_rule_version`.
It is a flagged departure from design §26.2 and §28.3, which keep sealed pairs and
forbid projected remarriage. Sealed pairs are drawn at birth inside one cohort, so
without re-pairing about 44 % of women reaching adulthood lost their partner before
18 and the population fell to 16 % of its start in a century.

At phase 35, after sealed pairs have had their chance, projected adults within the
marriage ages form a pool per site and ancestry if they have no union and no
reachable sealed partner: alive, co-resident, of marriageable age and still free.
This includes widowed and divorced adults. Women enter at `marriage_rate` and men
always, so a union draws that hazard once. Pools are matched in order of birth year
then slot ID, within `max_age_gap`, excluding kin within `kinship_depth` and former
partners. A new union closes both unformed sealed pairs, keeps the mother's last
birth for spacing, reproduces from `Y+1`, and has a family key marked `local_repair`.
Regulation still creates no union. The sealed pair remains the first choice; a closed
pair never forms again. `diagnostics.local_repair_unions` counts these unions.
