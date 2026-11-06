Recommend a **versioned integrated registry**, while preserving the standalone full and hybrid paths. Core precomputation can remain, subject to the independence rule below.

**1. Representation**

Make the registry authoritative; derive cohort counts from it. Reconstructing nominal lives afterward from the existing survival prefixes would lose migration and household history.

| Table | Essential fields |
|---|---|
| `OriginCohort` | Immutable semantic ID, origin type, ancestry, birth year, sealed sex counts |
| `Slot` | Immutable origin ID, sex ordinal, `simulated/projected`, birth/death facts, coverage, departure state, parental references, original family key |
| `PotentialPair` | Stable pair key, two slot references, `unformed/active/closed`, reproductive exclusions, last successful birth |
| `UnionAllocation` | Union ID, partners, formation/end, formation site, source |
| `Household` / membership episodes | Persistent household ID, site, members, validity, formation reason |
| Event ledgers | Birth allocations, deaths, guardian assignments, residence episodes, transfers |
| Published annual state | Counts by site/component; sparse origin-cohort × current-site × sex counts; controller state |

Use dense `int32` row references internally; public IDs must be lossless semantic identities, never hashes or row positions. Seal each cohort before defining its permutations; later arrivals create new origin cohorts.

Indexes: ID→row; living rows; household→members; site→households; current risk group→slots; person→residence episodes; parent→children. Replace integrated `parents()`/`children()` reconstruction with reciprocal indexes over recorded birth allocations. Freeze the original family key across residence changes.

Use NumPy columns and serial Numba kernels for scans, grouping, selection and membership updates. Keep plan validation, rational controller arithmetic, semantic IDs and JSON in Python. Separate core/background random streams; avoid Python `hash()` and container iteration order.

Cost is honestly `O(L + E + Σ living-slot-years + T·S³)`, memory `O(L + E)`, for historical lives `L` and emitted records `E`. With the fixed century and ≤32 sites, this is linear in represented lives. Do not allocate an `L×T` personal-state matrix. Strictly linear export ordering requires radix sorting; ordinary sorting adds `O(L log L)`.

**2. Annual order and core precomputation**

Let `O(Y)` be the immutable state published at phase 80 of `Y−1`.

| Phase | Operations and inputs |
|---|---|
| **10** | Resolve dated profiles, capacities, routes and locally supplied shocks. Snapshot opening household counts from `O(Y)` for rate denominators. Compute projected fertility correction from `O(Y)` and shocks starting now. |
| **20** | Core and projected births read opening survivors and unions already active in `O(Y)`. Apply successful-birth spacing. Record newborns and reciprocal parentage at the mother’s current site; attach newborns to her household. |
| **30** | Apply mortality once, including newborn and maternal mortality, using residence before phase-40 movement. Record death sites; close affected unions and repair surviving dependants’ guardianship. |
| **35** | Divorces, then unions. Projected potential pairs activate only when both partners survive, qualify and are co-resident. Adults left with no union and no reachable sealed partner then enter local re-pairing (`chronos-local-repairing-1`, see `chronos-bridge.md`). Newly activated unions cannot reproduce until `Y+1`. Stage household changes and core marriage relocations. |
| **40** | Commit the household partition; complete marriage relocations; execute programmed operations; statistical ports; ordinary mobility. Each actual transfer updates residence and both site counters atomically. |
| **50** | Chronos institutions/vacancies use current survivors and post-movement residence. |
| **55** | Chronos cohesion/scission. Its annual exposure/vacancy inputs remain the previously published state, as specified by §26.6. |
| **60** | Chronos assignments use current eligible residents and preceding institutional decisions. |
| **70** | Other relations use the state resulting from preceding phases. |
| **80** | Verify counts, memberships, references and budgets; perform any recovery rebase after transfers; publish the annual state. |

The demographic worker supplies phases 10–40 and the demographic publication; it does not choose officeholders.

**The residual regulator and `move_households` do not themselves require interleaving.** Section 25.3 explicitly permits preallocation to the core. Retaining precomputation requires:

- Core fertility and movement choices depend only on its state and fixed plan inputs.
- The core emits a phase-stamped event tape, including actual household transfers and operation IDs. Its annual population series alone is insufficient.
- The background consumes only events effective now. Its controller never reads the precomputed core’s current-year closing population.
- For an operation requesting `H` households, background demand is `max(0, H − actual_core_households_moved)`, not a subtraction of people or allocated quota.

The present [`run_core` arrangement](D:/workspace/self/_worktrees/lineage-s3/src/lineage_lab/hybrid_engine.py:103) can support that contract after adding these records and integrated hooks.

There is a specification conflict if destination weights using **combined realized population** also govern core decisions. That requires annual interleaving. My S3 recommendation is to scope those weights to projected mobility and preserve core independence. Otherwise introduce a phase-steppable integrated driver. The current core uses its own population and attraction formula in [`_migrations`](D:/workspace/self/_worktrees/lineage-s3/src/lineage_lab/engine.py:935).

**3. Mortality: linear selection and the defect**

For an immutable cohort-sex universe of size `N`, define each slot’s site-specific rank once. With the existing affine permutation:

```text
rank(slot, site) = inverse_a × (sex_ordinal − b) mod N
parameters keyed by (origin_cohort_id, site_id, sex, mortality_rule_version)
```

Each year:

1. Gather only currently present eligible slots, grouped by homogeneous hazard.
2. Compute each group’s death quota from its **current present count**.
3. Select the quota’s lowest stable ranks with fixed-width radix selection: linear work, without sorting.
4. Record deaths permanently.

Never redefine `N` as the current population. Never retain a forward-only permutation cursor: an absent slot skipped previously may return. Never continue origin-based cumulative survival after migration.

Specify quota rounding explicitly; for example, indexed stochastic rounding keyed by group and year. Maternal mortality must target actual mothers, rather than unrelated females in their cohort.

**The fixed ordering has a statistical defect under migration.** Established survivors have already exhausted low site ranks; newcomers have not. Consequently newcomers can suffer systematically higher mortality despite identical age/sex hazards. This is a fixed susceptibility model, not equal annual risk.

Recommended amendment: retain immutable slot ordinals, but key the **annual death ordering additionally by year**. This changes §25.2 semantics and needs a new mortality-rule version. A stable identity does not require a lifelong death priority.

**4. Pairs, households and guardians**

At bootstrap: create founder adult unions and household units; assign founder dependants using the inline age compatibility, least-load rule and foster limit; assign unresolved dependants to `synthetic_care`; then execute movements. Integrated bootstrap must insert this before the current core’s immediate marriage movements.

At each structural transition, compute a complete destination membership partition, then commit it atomically:

- Remove deaths and terminate unions.
- Preserve the survivor’s household ID after widowhood; close its membership episode when membership changes.
- Resolve divorce, majority and new-union households.
- Assign each dependant to an eligible declared guardian, otherwise a known co-resident parent by minimum ID, otherwise care.
- Close emptied households **after** surviving dependants have destinations.

The integrated core must consult declared founder guardians; its current [`_dependents`](D:/workspace/self/_worktrees/lineage-s3/src/lineage_lab/engine.py:385) only searches biological parents.

A transfer freezes the household’s member list and moves that list as one unit. Household identity survives migration. Mark marriage relocations and incoming households as already moved for phase 40, preventing another move under a newly created household ID.

Projected pairs have one lifetime union: divorce or death makes them permanently closed. Only never-formed pairs may activate after a later reunion.

Define care eligibility explicitly; I recommend stationary `synthetic_care` units in S3, excluded from mobility selection. Incoming formed households need explicit unions/guardians without invented biological parentage.

**5. Site regulator**

Use exact rational arithmetic for interpolation, `q` and comparisons; quantize the resulting multiplier to ppm once, with specified rounding.

To match the explicit 1365 example, adopt:

```text
shock effects: [start_year, end_year)
rebase_year: end_year + recovery_years
controller suspended through rebase_year, inclusive
```

Thus the example rebases in phase 80 of 1365 and resumes in 1366. This convention also means recovery zero includes the end boundary; state that explicitly.

```text
Initial state per site:
    q = 1
    M = M_ref = 0
    zero = (baseline(start_year) == 0)
    freeze_until = NONE

Phase 10(Y):
    old = published(site, Y−1)
    freeze_until = old.freeze_until

    for shock starting at this site in Y:
        freeze_until = max_optional(
            freeze_until,
            shock.end_year + shock.recovery_years
        )

    frozen = freeze_until is not NONE
    P = old.core + old.background

    G_raw = P if old.zero else (
        old.q * baseline(Y−1) + old.M − old.M_ref
    )
    G = max(0, G_raw)
    error = (G − old.core) − old.background
    scale = max(1, G)
    tolerance = interpolated_tolerance_bps(Y−1)

    multiplier_ppm = 1_000_000
    if not frozen and not old.zero:
        if 10_000 * abs(error) > tolerance * scale:
            multiplier_ppm = clamp(
                round_half_up(
                    1_000_000 * (1 + error / (response_years * scale))
                ),
                min_fertility_multiplier_ppm,
                max_fertility_multiplier_ppm
            )

Phase 20:
    Apply this multiplier only to eligible projected birth probabilities.
    Zero eligibility remains zero.
    Multiplier 1 is an identity operation.
    The probability cap limits corrective increases.

Phase 40:
    Record actual incoming/outgoing PERSON counts, including core transfers.

Phase 80(Y):
    P = current_core + current_background
    M = old.M + incoming_people − outgoing_people
    B = baseline(Y)
    recovery_complete = (freeze_until == Y)

    retain old q, M_ref and zero unless changed below

    if B == 0:
        zero = true
        q = NONE
        M_ref = M
    else if recovery_complete or (old.zero and not frozen):
        q = exact_ratio(P, B)
        M_ref = M
        zero = false

    if recovery_complete:
        freeze_until = NONE

    verify:
        P == old.P + births − deaths + incoming_people − outgoing_people

    publish counts and controller state
```

Processing newly starting shocks before testing completion handles overlapping recoveries, including a shock starting on the old rebase date. It neither rebases early nor freezes before a future shock starts.

Zero-baseline mode is neutral. The first positive annual interpolated baseline anchors at phase 80; correction starts next year. An inflow updates realized population and migration counters but cannot itself define `P/0`.

Keep signed `G_raw` for diagnostics, while controlling against nonnegative `G`. Report saturation, core exceeding guide, and lack of eligible pairs. The controller cannot guarantee a demographic target.

**6. Fertility correction**

The established discrepancy is a projection-spacing defect; it does not establish that the core’s spacing rule is wrong. The measured mechanism is documented in the [fertility audit](D:/workspace/self/_worktrees/lineage-s3/docs/chronos-bridge.md:211).

| Option | Assessment |
|---|---|
| Align integrated `Cohorts.bounds` | **Recommended first.** Replace periodic opportunities with eligibility based on the last successful birth. The registry now makes this practical. |
| Recalibrate the template | Necessary afterward if its intended demography is stable/growing but its unregulated process is below replacement. Version and apply the resolved profile consistently to both components. |
| Preserve and publish the discrepancy | Suitable for preserved legacy histories; unsuitable as the integrated default. Regulation would conceal a known systematic error. |

Simply removing multiplication by spacing gives `p/s`, which also disagrees with the core. Substituting the stationary formula alone fails to reproduce changing fertility, cooldown after shocks and union formation.

With permanent pairs and constant conditions, both integrated mechanisms should approach:

```text
births per pair-year = p / (1 + (s−1)p)
```

Validation must first repeat the controlled 2,000-pair probe with regulation disabled, across several probabilities/spacings and a temporary fertility interruption. Verify successful-birth cooldown directly.

Then measure unregulated net reproduction, partnership exposure, century trajectories and core extinction across the registered seeds. If the template intends replacement, calibrate toward that intention; do not require the small core to track a target. Finally measure controller saturation and unmet guide error. The regulator should not be the sole mechanism sustaining an intentionally stable population.

**7. Specification amendments required**

Add these explicit rules to the cited sections:

- **§25.3:** “Core movement choices are independent of projected population; combined-population destination weights apply to projected mobility.” Otherwise mandate interleaving.
- **§25.2:** “Permutation universes are sealed origin-sex cohorts; annual mortality selection includes the year in its key.” Also specify quota rounding and maternal eligibility.
- **§25.2/§26.4:** “Household episode closure does not necessarily close household identity. Survivor reassignment precedes household closure.” Define marriage relocation within phase 40 and the once-per-phase rule.
- **§7.4/§25.3/§26.4:** Adopt the recovery boundary and controller formula above, including observation year, negative guide handling and zero-baseline exit. “Or an explicit inflow” presently does not define a valid zero-denominator rebase.
- **§8.4/§26.2:** “The original family key identifies the parental pair allocation, independent of the child’s birth site/cohort.” Define keys for incoming formed unions whose partners belong to different birth cohorts.
- **§8.2/§24.3:** Retain intra-year event ordering internally. Preserve births/deaths and visits with empty annual intervals; departed people have bounded coverage and unknown subsequent death, rather than a fabricated death date.
- **§8.4/§25.3:** Define incoming-household construction, care mobility, core allocation for port destinations, and whether arrivals count as founders for historical share budgets. Count arrivals in the historical slot budget.

The current translation also drops operations, ports, regulation and recovery metadata from executable settings; extending [`HistoricalSettings` and translation](D:/workspace/self/_worktrees/lineage-s3/src/lineage_lab/chronos_plan.py:501) is a prerequisite.

Implementation order, with a testable exit for each step:

1. Freeze the amended temporal, random and regulator contracts; add reference vectors and retain standalone golden histories.
2. Extend integrated settings without losing plan fields; test serialization and boundary conversions.
3. Implement immutable slots, sealed cohorts and residence indexes; test identity stability and birth-year deaths.
4. Implement unions, households, founder guardians and care; test widowhood, divorce, majority and simultaneous guardian deaths.
5. Implement annual birth/death transitions; validate spacing, mortality selection and reciprocal parentage.
6. Add the core event tape and household-operation accounting; test replay and residual household requests.
7. Implement programmed moves, ports and ordinary mobility; prove exact site/domain balances and one movement per household.
8. Implement the regulator; test overlapping shocks, the 1365 transfer/rebase, zero baseline, emigration and unavailable fertility.
9. Export and independently validate the complete result; test query/job ordering, budgets and maximum-domain performance. Run `python -m pytest` and `ruff check src tests`; verify an installed wheel outside the checkout if packaging changes.