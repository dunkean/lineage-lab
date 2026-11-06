"""Canonical complete S3 tables, followed by independent table validation."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .historical_program import ProviderError
from .registry_contract import NO_YEAR

RESULT_FORMAT = "population-result-json-1"


def export_tables(registries, core=None):
    # Python's stable comparison sort is explicit: O(L log L) export ordering.
    # Simulation/grouping/death selection use fixed-width linear kernels.
    people = sorted(
        ((r.ids[p], i, p) for i, r in enumerate(registries) for p in range(r.n)),
        key=lambda value: value[0].encode("utf-8"),
    )
    mappings = [{} for _ in registries]
    for row, (_, component, person) in enumerate(people):
        mappings[component][person] = row
    columns = {
        f: []
        for f in (
            "origin_id",
            "origin_cohort_id",
            "sex_ordinal",
            "source",
            "ancestry_id",
            "sex",
            "birth_year",
            "death_year",
            "death_site_id",
            "coverage_start_year",
            "coverage_end_year",
            "departure_state",
            "departure_year",
            "mother_slot",
            "father_slot",
            "family_key",
            "reproduction_flags",
            "core_attributes",
        )
    }
    for identity, component, person in people:
        r, mapping = registries[component], mappings[component]
        d = r.data

        def nullable(field, d=d, person=person):
            value = int(d[field][person])
            return None if value == NO_YEAR else value

        death, departure = nullable("death"), nullable("departure")
        attributes = None
        if core is not None and r.source == "simulated":
            original = core.person_map[person]
            attributes = {
                "status_rank": int(core.data["status"][original]),
                "activity_id": core.activities[int(core.data["activity"][original])],
                "observation": "last_simulated_state",
            }
        values = {
            "origin_id": identity,
            "origin_cohort_id": r.cohorts[d["cohort"][person]]["origin_cohort_id"],
            "sex_ordinal": int(d["ordinal"][person]),
            "source": r.source,
            "ancestry_id": r.config.races[d["race"][person]].name,
            "sex": int(d["sex"][person]),
            "birth_year": int(d["birth"][person]),
            "death_year": death,
            "death_site_id": None if death is None else r.site_ids[d["death_site"][person]],
            "coverage_start_year": int(d["coverage_start"][person]),
            "coverage_end_year": int(d["coverage_end"][person]),
            "departure_state": "departed_external"
            if departure is not None
            else "dead"
            if death is not None
            else "resident_at_coverage_end",
            "departure_year": departure,
            "mother_slot": mapping.get(int(d["mother"][person])),
            "father_slot": mapping.get(int(d["father"][person])),
            "family_key": r.family[person],
            "reproduction_flags": int(d["flags"][person]),
            "core_attributes": attributes,
        }
        for field, column in columns.items():
            column.append(values[field])
    tables = {
        "candidate_table": columns,
        "origin_cohorts": [],
        "residence_index": [],
        "household_episodes": [],
        "union_allocations": [],
        "birth_allocations": [],
        "death_allocations": [],
        "residence_transfers": [],
        "guardian_assignments": [],
    }
    single_refs = {"person_slot", "child_slot", "mother_slot", "father_slot", "guardian_slot"}
    multiple_refs = {"member_slots", "partner_slots", "origin_slots"}
    fields = {
        "residence_index": "residences",
        "household_episodes": "episodes",
        "union_allocations": "unions",
        "birth_allocations": "births",
        "death_allocations": "deaths",
        "residence_transfers": "transfers",
        "guardian_assignments": "guardians",
    }
    for i, r in enumerate(registries):
        mapping = mappings[i]
        tables["origin_cohorts"].extend(r.cohorts)
        for table, field in fields.items():
            for source in getattr(r, field):
                record = {}
                for key, value in source.items():
                    if key in single_refs:
                        record[key] = mapping.get(value)
                    elif key in multiple_refs:
                        record[key] = sorted(mapping[v] for v in value)
                    else:
                        record[key] = value
                tables[table].append(record)
    tables["origin_cohorts"].sort(key=lambda c: c["origin_cohort_id"].encode())
    tables["residence_index"].sort(
        key=lambda r: (r["person_slot"], r["valid"]["start_year"], r["start_phase"])
    )
    for key, id_field in (
        ("household_episodes", "household_id"),
        ("union_allocations", "union_id"),
    ):
        tables[key].sort(
            key=lambda r: (r[id_field].encode(), r["valid"]["start_year"], r["start_phase"])
        )
    for key, field in (
        ("birth_allocations", "child_slot"),
        ("death_allocations", "person_slot"),
        ("guardian_assignments", "child_slot"),
    ):
        tables[key].sort(
            key=lambda r: (r[field], r["year"], r["phase"], r.get("guardian_slot") or -1)
        )
    unions_by_family = defaultdict(list)
    for union in tables["union_allocations"]:
        unions_by_family[union["family_key"]].append(union)
    for birth in tables["birth_allocations"]:
        child, year = birth["child_slot"], birth["year"]
        birth["child_origin_cohort_id"] = columns["origin_cohort_id"][child]
        birth["child_sex_ordinal"] = columns["sex_ordinal"][child]
        active_unions = [
            u
            for u in unions_by_family[birth["family_key"]]
            if u["valid"]["start_year"] < year
            and (u["valid"]["end_year"], u["end_phase"]) > (year, 20)
        ]
        birth["parental_union_id"] = active_unions[0]["union_id"] if active_unions else None
    tables["residence_transfers"].sort(
        key=lambda r: (r["year"], r["phase"], r["transfer_id"].encode())
    )
    offsets = [0]
    count_by_person = np.bincount(
        [r["person_slot"] for r in tables["residence_index"]], minlength=len(people)
    )
    for count in count_by_person:
        offsets.append(offsets[-1] + int(count))
    tables["residence_offsets"] = offsets
    children = [[] for _ in people]
    parentage = []
    for child in range(len(people)):
        mother, father = columns["mother_slot"][child], columns["father_slot"][child]
        if mother is not None or father is not None:
            parentage.append(
                {
                    "child_slot": child,
                    "mother_slot": mother,
                    "father_slot": father,
                    "precision": "simulated"
                    if columns["source"][child] == "simulated"
                    else "projected",
                }
            )
            for parent in (mother, father):
                if parent is not None:
                    children[parent].append(child)
    tables.update(parentage=parentage, children_index=children)
    return tables


def validate_population_result(result, limits):
    """Recompute site-year balances and memberships from exported facts only.

    This validator never consumes simulation counters or registry objects. Its
    reference checks and event sweeps include birth-year deaths and empty visits.
    A mutation to a census, death, membership or transfer invalidates the proof.
    """

    def require(condition, message):
        if not condition:
            raise ProviderError("InvalidPopulationResult", message, constraint="export_validation")

    c = result["candidate_table"]
    n = len(c["origin_id"])
    require(n <= limits["max_candidate_slots"], "Candidate slot budget exceeded")
    require(all(len(v) == n for v in c.values()), "Candidate columns have different lengths")
    require(
        c["origin_id"] == sorted(set(c["origin_id"]), key=lambda s: s.encode()),
        "Candidate IDs must be unique and sorted",
    )
    start, end = (result["timeline"][k] for k in ("start_year", "end_year"))
    cohorts = {r["origin_cohort_id"]: r for r in result["origin_cohorts"]}
    require(
        len(cohorts) == len(result["origin_cohorts"]) <= limits["max_cohorts"],
        "Invalid sealed cohorts",
    )
    sex_ordinals = defaultdict(set)
    for p in range(n):
        origin = c["origin_cohort_id"][p]
        require(origin in cohorts, "Unknown origin cohort")
        cohort = cohorts[origin]
        require(
            cohort["sealed"]
            and cohort["birth_year"] == c["birth_year"][p]
            and cohort["ancestry_id"] == c["ancestry_id"][p],
            "Origin facts differ from sealed cohort",
        )
        sex_ordinals[origin, c["sex"][p]].add(c["sex_ordinal"][p])
        require(
            start <= c["coverage_start_year"][p] <= c["coverage_end_year"][p] <= end,
            "Invalid person coverage",
        )
        if c["departure_state"][p] == "departed_external":
            require(
                c["death_year"][p] is None and c["departure_year"][p] == c["coverage_end_year"][p],
                "An external departure must not invent a death",
            )
    for origin, cohort in cohorts.items():
        for sex in (0, 1):
            require(
                sex_ordinals[origin, sex] == set(range(cohort["sex_counts"][sex])),
                "Sealed sex ordinals are incomplete",
            )

    def ref(p):
        require(type(p) is int and 0 <= p < n, "Dangling person reference")

    residence_events = defaultdict(list)
    household_events = defaultdict(list)
    residences = defaultdict(list)
    for row in result["residence_index"]:
        p = row["person_slot"]
        ref(p)
        a, b = (
            (row["valid"]["start_year"], row["start_phase"]),
            (row["valid"]["end_year"], row["end_phase"]),
        )
        require(a <= b and start <= a[0] < end and b[0] <= end, "Invalid residence interval")
        require(not residences[p] or residences[p][-1][1] == a, "Residence gap or overlap")
        residences[p].append((a, b, row["site_id"], row["end_reason"]))
        if a < b:
            residence_events[a].append((1, p, row["site_id"]))
            residence_events[b].append((-1, p, row["site_id"]))
    expected_offsets = [0]
    for p in range(n):
        require(bool(residences[p]), "Candidate has never resided in domain")
        require(
            residences[p][0][0][0] == c["coverage_start_year"][p],
            "Coverage starts outside residence",
        )
        expected_offsets.append(expected_offsets[-1] + len(residences[p]))
    require(expected_offsets == result["residence_offsets"], "Residence index offsets differ")
    for row in result["household_episodes"]:
        a, b = (
            (row["valid"]["start_year"], row["start_phase"]),
            (row["valid"]["end_year"], row["end_phase"]),
        )
        require(a <= b, "Invalid household interval")
        require(
            len(row["member_slots"]) == len(set(row["member_slots"])), "Duplicate household member"
        )
        for p in row["member_slots"]:
            ref(p)
            if a < b:
                household_events[a].append((1, p, row["site_id"]))
                household_events[b].append((-1, p, row["site_id"]))
    birth_counts, death_counts, incoming, outgoing = [defaultdict(int) for _ in range(4)]
    birth_rows, death_rows = {}, {}
    unions = {u["union_id"]: u for u in result["union_allocations"]}
    require(len(unions) == len(result["union_allocations"]), "Duplicate union identity")

    def present_at(person, stamp, site):
        return any(a <= stamp < b and s == site for a, b, s, _ in residences[person])

    for row in result["birth_allocations"]:
        p = row["child_slot"]
        ref(p)
        require(
            p not in birth_rows and row["year"] == c["birth_year"][p], "Invalid birth allocation"
        )
        require(row["family_key"] == c["family_key"][p], "Original family key changed")
        require(
            row["child_origin_cohort_id"] == c["origin_cohort_id"][p]
            and row["child_sex_ordinal"] == c["sex_ordinal"][p],
            "Birth ordinal differs from sealed origin",
        )
        require(row["parental_union_id"] in unions, "Birth has no recorded parental union")
        union = unions[row["parental_union_id"]]
        require(
            union["family_key"] == row["family_key"]
            and union["valid"]["start_year"] < row["year"]
            and (union["valid"]["end_year"], union["end_phase"]) > (row["year"], 20),
            "Birth predates active union or follows its closure",
        )
        require(
            set(union["partner_slots"]) == {row["mother_slot"], row["father_slot"]},
            "Birth parents differ from union partners",
        )
        for name in ("mother_slot", "father_slot"):
            parent = row[name]
            require(parent == c[name][p], "Birth parent differs from candidate")
            if parent is not None:
                ref(parent)
                require(
                    c["birth_year"][parent] < c["birth_year"][p], "Parent is not older than child"
                )
                require(
                    present_at(parent, (row["year"], 20), row["site_id"]),
                    "Birth parents are not co-resident",
                )
        require(residences[p][0][2] == row["site_id"], "Birth is outside initial residence")
        birth_rows[p] = row
        birth_counts[row["site_id"], row["year"]] += 1
    for row in result["death_allocations"]:
        p = row["person_slot"]
        ref(p)
        require(
            p not in death_rows
            and c["death_year"][p] == row["year"]
            and c["death_site_id"][p] == row["site_id"],
            "Invalid death allocation",
        )
        require(
            residences[p][-1][1] == (row["year"], 30) and residences[p][-1][2] == row["site_id"],
            "Death did not occur in exposed residence",
        )
        death_rows[p] = row
        death_counts[row["site_id"], row["year"]] += 1
    for p in range(n):
        require((c["death_year"][p] is not None) == (p in death_rows), "Missing death allocation")
        require(
            (cohorts[c["origin_cohort_id"][p]]["origin_type"] == "birth") == (p in birth_rows),
            "Missing birth allocation",
        )
    moved = set()
    transfer_ids = set()
    household_endpoints = defaultdict(list)
    for h in result["household_episodes"]:
        household_endpoints[h["household_id"]].append(h)
    for row in result["residence_transfers"]:
        require(row["transfer_id"] not in transfer_ids, "Duplicate transfer ID")
        transfer_ids.add(row["transfer_id"])
        require(row["from_site"] != row["to_site"], "Transfer has identical endpoints")
        require(
            len(row["origin_slots"]) == len(set(row["origin_slots"])) > 0,
            "Transfer must freeze distinct resident members",
        )
        endpoint = "end_year" if row["from_site"] is not None else "start_year"
        phase = "end_phase" if row["from_site"] is not None else "start_phase"
        require(
            any(
                h["valid"][endpoint] == row["year"]
                and h[phase] == 40
                and set(h["member_slots"]) == set(row["origin_slots"])
                for h in household_endpoints[row["household_id"]]
            ),
            "Transfer does not match frozen household membership",
        )
        for p in row["origin_slots"]:
            ref(p)
            require((p, row["year"]) not in moved, "Person moved more than once in phase 40")
            moved.add((p, row["year"]))
            stamp = (row["year"], 40)
            if row["from_site"] is not None:
                require(
                    any(b == stamp and site == row["from_site"] for a, b, site, _ in residences[p]),
                    "Transfer departure lacks a residence endpoint",
                )
                outgoing[row["from_site"], row["year"]] += 1
            if row["to_site"] is not None:
                require(
                    any(a == stamp and site == row["to_site"] for a, b, site, _ in residences[p]),
                    "Transfer arrival lacks a residence endpoint",
                )
                incoming[row["to_site"], row["year"]] += 1
            else:
                require(c["departure_year"][p] == row["year"], "External exit lacks departure fact")
    expected_children = [[] for _ in range(n)]
    expected_parentage = []
    for p in range(n):
        mother, father = c["mother_slot"][p], c["father_slot"][p]
        for parent in (mother, father):
            if parent is not None:
                ref(parent)
                expected_children[parent].append(p)
        if mother is not None or father is not None:
            expected_parentage.append(
                {
                    "child_slot": p,
                    "mother_slot": mother,
                    "father_slot": father,
                    "precision": c["source"][p],
                }
            )
    require(
        expected_children == result["children_index"] and expected_parentage == result["parentage"],
        "Family links are not reciprocal",
    )
    for row in result["union_allocations"]:
        require(len(row["partner_slots"]) == 2, "Union must have two partners")
        for p in row["partner_slots"]:
            ref(p)
            bound = (
                (c["death_year"][p], 30)
                if c["death_year"][p] is not None
                else (c["departure_year"][p], 40)
                if c["departure_year"][p] is not None
                else (end, 0)
            )
            require(
                (row["valid"]["end_year"], row["end_phase"]) <= bound,
                "Union extends beyond a partner's death or coverage",
            )
        if row["source"] != "simulated":
            require(
                all(
                    present_at(
                        p,
                        (row["valid"]["start_year"], row["start_phase"]),
                        row["site_at_formation"],
                    )
                    for p in row["partner_slots"]
                ),
                "Projected union formed before co-residence",
            )
    for row in result["guardian_assignments"]:
        ref(row["child_slot"])
        if row["guardian_slot"] is not None:
            ref(row["guardian_slot"])
    census_rows = {(r["site_id"], r["year"]): r for r in result["annual_censuses"]}
    sites = sorted({r["site_id"] for r in result["annual_censuses"]})
    require(
        len(census_rows) == len(result["annual_censuses"]) == len(sites) * (end - start),
        "Incomplete annual census grid",
    )
    # Sweep exported interval boundaries, closing old episodes before opening new ones.
    positions = {"residence": np.full(n, -1, np.int32), "household": np.full(n, -1, np.int32)}
    active = {"residence": np.zeros(n, np.int32), "household": np.zeros(n, np.int32)}
    site_index = {site: i for i, site in enumerate(sites)}
    events = sorted(
        [
            (stamp, kind, delta, p, site)
            for kind, table in (("residence", residence_events), ("household", household_events))
            for stamp, rows in table.items()
            for delta, p, site in rows
        ],
        key=lambda e: (e[0], e[2]),
    )
    previous = defaultdict(int)
    for p in range(n):
        if cohorts[c["origin_cohort_id"][p]]["origin_type"] == "founder":
            previous[residences[p][0][2]] += 1
    cursor = 0
    sparse = {
        (r["origin_cohort_id"], r["site_id"], r["sex"], r["source"], r["year"]): r["count"]
        for r in result["cohort_residence_counts"]
    }
    require(len(sparse) == len(result["cohort_residence_counts"]), "Duplicate sparse cohort count")
    recomputed_sparse = {}
    for year in range(start, end):
        while cursor < len(events) and events[cursor][0] <= (year, 80):
            _, kind, delta, p, site = events[cursor]
            require(site in site_index, "Unknown residence or household site")
            active[kind][p] += delta
            positions[kind][p] = site_index[site] if delta == 1 else -1
            cursor += 1
        require(
            np.all((active["residence"] == 0) | (active["residence"] == 1)),
            "Overlapping residences",
        )
        require(
            np.array_equal(active["residence"], active["household"]),
            "Every resident must have one household",
        )
        require(
            np.array_equal(positions["residence"], positions["household"]),
            "Household is outside member residence",
        )
        counts = defaultdict(int)
        for p in np.flatnonzero(active["residence"]):
            counts[sites[positions["residence"][p]], c["source"][p]] += 1
            key = (
                c["origin_cohort_id"][p],
                sites[positions["residence"][p]],
                c["sex"][p],
                c["source"][p],
                year,
            )
            recomputed_sparse[key] = recomputed_sparse.get(key, 0) + 1
        for site in sites:
            row = census_rows[site, year]
            core, projected = counts[site, "simulated"], counts[site, "projected"]
            births, deaths = birth_counts[site, year], death_counts[site, year]
            ins, outs = incoming[site, year], outgoing[site, year]
            population = core + projected
            require(
                population == previous[site] + births - deaths + ins - outs,
                "Table-derived site-year balance does not close",
            )
            expected = {
                "population": population,
                "simulated": core,
                "projected": projected,
                "births": births,
                "deaths": deaths,
                "incoming": ins,
                "outgoing": outs,
            }
            require(
                all(row[k] == value for k, value in expected.items()),
                "Census differs from independently recomputed tables",
            )
            previous[site] = population
    require(sparse == recomputed_sparse, "Sparse cohort counts differ from resident tables")
    return {
        "validator": "chronos-export-validator-1",
        "valid": True,
        "candidate_slots": n,
        "site_years": len(census_rows),
        "checks": [
            "sealed_origins",
            "reciprocal_parentage",
            "residence_intervals",
            "one_household_per_resident",
            "exposed_death_sites",
            "actual_transfers",
            "independent_site_year_balances",
        ],
    }
