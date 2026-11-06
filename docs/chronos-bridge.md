# Chronos bridge, S3 step 1

Run the local worker with the pinned environment:

```powershell
$env:NUMBA_CACHE_DIR = '.numba-cache'
python -m lineage_lab.chronos_bridge --runtime
python -m lineage_lab.chronos_bridge --job-root output/chronos-jobs
```

`--runtime` reports the runtime manifest and its fingerprint. Normal operation
reads one JSON object per stdin line and writes one response per stdout line.
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
and atomic replacement. Failed jobs do not publish a canonical result.
Durations appear only in the envelope's `metrics`.

The artifact uses **`chronos-lineage-partial-json-1`**, with
`result_type=PartialPopulationDiagnostics`, `coverage=partial`, `complete=false`
and `stage=S3-step-1`. It is not `population-result-json-1` or a complete
`PopulationResult`; it must not pass the host's `VerifiedPopulation` gate.
It includes:

- Provider/runtime manifests and the complete applied configuration.
- Annual post-transition counts for each site, with disjoint simulated and
  projected counts, births, deaths and realized net internal migration.
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
Explicit episode capacities replace automatic standalone capacity growth.

Choices where the design leaves a stage-1 detail open:

- The dedicated partial format above prevents accidental acceptance as a full
  provider result. The `job_directory` envelope field makes the relative path's
  resolution base explicit.
- Event multipliers retain Lineage Lab's cumulative-hazard semantics:
  `q_modified = 1 - (1 - q_base)^factor`. Independent extra mortality is composed
  afterward. Maximum lifespan remains a hard bound.
- Founders are pre-shock survivors. Bootstrap applies additional mortality and
  the excess mortality hazard `max(0, factor - 1)`, without an extra ordinary
  annual mortality pass or bootstrap births. Nonmortality initial effects remain
  active context; they do not invent an initial birth transition.
- Attempt seed `i` is the first eight SHA-256 bytes, little endian, of canonical
  `["chronos-lineage-core-attempt-1", seed_decimal_string, i]`. The selected
  scenario uses that seed; the standalone provider's core substream derivation
  is retained. The rule is versioned and part of the provider source fingerprint.

`free_try3_then_projected` declares exactly four attempts: target share at indices
0, 1 and 2, then zero share at index 3. Only the typed `CoreBudgetExceeded`
exception advances this policy. Slot/cohort/result-byte/memory failures and
unsupported biology do not activate it. An extinct compliant core is accepted.
`required_core` allows the declared attempt count, at most eight, with no
zero-share fallback. Selection happens before publication.

Step 2 must implement the residence and household registries, founder guardian
assignments using the **inline** foster limit, household movements, statistical
ports, per-site residual regulation and full bounded candidate/kinship export.
In this artifact the baseline remains unmodified input, all target correction is
disabled, role demands are validated planning inputs only, and deferred movement
and port IDs are explicitly reported. Background residence remains its origin
site for this step. Only `historical_soft_v1` is advertised as executed; accepting
ordinary S3 planning fields does not claim their deferred capabilities. Valid
S6 conditional/exact fields, S7 bridges, S8 dynamic settlements/procedural modes,
and unsupported reproduction are rejected with `UnsupportedCapability`.

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
