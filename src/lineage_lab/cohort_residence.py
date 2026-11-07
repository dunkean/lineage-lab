"""Authoritative integrated slots, sealed origin cohorts and residence episodes.

Dense int32 references are private. Public addresses encode semantic origins
without hashing. Events retain phase stamps even when annual intervals are empty.
There is no person-by-year state matrix and no post-hoc survival reconstruction.
"""

from __future__ import annotations

import json
from collections import defaultdict
from heapq import heappop, heappush

import numpy as np

from .historical_program import ProviderError
from .registry_contract import (
    NO_YEAR,
    REPAIRING_VERSION,
    address,
    encoded,
    key_prefix,
    opening_draws,
    stream,
)
from .registry_kernels import count_living_sites, radix_order


class ResidentRegistry:
    FIELDS = (
        "birth",
        "death",
        "death_site",
        "site",
        "race",
        "sex",
        "ordinal",
        "cohort",
        "mother",
        "father",
        "guardian",
        "household",
        "coverage_start",
        "coverage_end",
        "departure",
        "flags",
        "last_birth",
        "adult_initialized",
    )

    def __init__(self, config, source):
        self.config, self.source = config, source
        self.settings = config.historical
        self.domain = self.settings.domain_id
        self.start, self.end = config.start_year, config.start_year + config.duration + 1
        self.places = config.settlements
        self.site_ids = [self.settings.site_ids[p.id] for p in self.places]
        self.local_index = {p.id: i for i, p in enumerate(self.places)}
        self.public_index = {p: i for i, p in enumerate(self.site_ids)}
        self.capacity = self.settings.limits["max_candidate_slots"]
        self.data = {f: np.full(self.capacity, -1, np.int32) for f in self.FIELDS}
        self.n = 0
        self.living_rows = np.empty(0, np.int32)
        self.site_households = [set() for _ in self.places]
        self.ids, self.family = [], []
        self.origin_keys, self.slot_keys = [], []
        self.id_rows = {}
        self.cohorts, self.cohort_rows = [], {}
        self.residences, self.residence_index = [], {}
        self.births, self.deaths, self.transfers, self.guardians = [], [], [], []
        self.pairs, self.unions, self.households, self.episodes = [], [], [], []
        # `pairs` in order, without those already closed: a closed pair never reopens.
        self.open_pairs = []
        self.key_prefix = key_prefix(config.seed)
        self.hh_keys, self.hh_episode = {}, {}
        self.children = defaultdict(list)
        self.declared_guardians = defaultdict(list)
        self.partner = {}
        self.sealed_pair = {}
        self.former_partners = set()
        self.moved = set()
        self.year = self.start
        self.counters = np.zeros((len(self.places), 4), np.int64)  # births, deaths, in, out

    def living(self):
        d = self.data
        return self.living_rows[d["site"][self.living_rows] >= 0]

    def counts(self):
        return count_living_sites(self.living_rows, self.data["site"], len(self.places))

    def cohort_counts(self):
        d, living = self.data, self.living()
        if not len(living):
            return []
        size = len(self.places)
        codes = (
            (d["cohort"][living].astype(np.int64) * size + d["site"][living]) * 2 + d["sex"][living]
        ).astype(np.uint64)
        order = radix_order(codes)
        keys = codes[order]
        boundaries = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1]) + 1, len(keys)]
        people = living[order[boundaries[:-1]]]
        return [
            {
                "origin_cohort_id": self.cohorts[cohort]["origin_cohort_id"],
                "site_id": self.site_ids[site],
                "sex": sex,
                "source": self.source,
                "year": self.year,
                "count": count,
            }
            for cohort, site, sex, count in zip(
                d["cohort"][people].tolist(),
                d["site"][people].tolist(),
                d["sex"][people].tolist(),
                np.diff(boundaries).tolist(),
                strict=True,
            )
        ]

    def open_year(self, year):
        self.year = year
        self.moved.clear()
        self.counters.fill(0)

    def add_cohort(
        self,
        kind,
        site,
        race,
        birth,
        sexes,
        *,
        year=None,
        token=None,
        parents=None,
        families=None,
        flags=None,
        origin_type=None,
        phase=None,
    ):
        year = self.year if year is None else year
        phase = phase if phase is not None else 20 if kind == "birth" else 10
        sexes = np.asarray(sexes, np.int32)
        count = len(sexes)
        if self.n + count > self.capacity:
            raise ProviderError(
                "BudgetExceeded",
                "Candidate slot budget exceeded",
                year=year,
                constraint="max_candidate_slots",
            )
        key = [
            self.domain,
            self.source,
            kind,
            self.site_ids[site],
            self.config.races[race].name,
            birth,
            token,
        ]
        cohort_id = address("origin", *key)
        if cohort_id in self.cohort_rows:
            raise ProviderError("InvalidPopulationResult", "Origin cohort was already sealed")
        if len(self.cohorts) >= self.settings.limits["max_cohorts"]:
            raise ProviderError(
                "BudgetExceeded", "Cohort budget exceeded", constraint="max_cohorts"
            )
        cohort = len(self.cohorts)
        self.cohort_rows[cohort_id] = cohort
        counts = [int((sexes == sex).sum()) for sex in (0, 1)]
        self.cohorts.append(
            {
                "origin_cohort_id": cohort_id,
                "origin_type": origin_type or kind,
                "ancestry_id": self.config.races[race].name,
                "birth_year": birth,
                "sex_counts": counts,
                "sealed": True,
            }
        )
        self.origin_keys.append(key[2:])
        rows = np.arange(self.n, self.n + count, dtype=np.int32)
        self.n += count
        self.living_rows = np.concatenate((self.living_rows, rows))
        d = self.data
        for field, value in (
            ("birth", birth),
            ("death", NO_YEAR),
            ("site", site),
            ("race", race),
            ("cohort", cohort),
            ("coverage_start", year),
            ("coverage_end", self.end),
            ("departure", NO_YEAR),
            ("flags", 0),
            ("last_birth", self.start - 10000),
            ("adult_initialized", 0),
        ):
            d[field][rows] = value
        d["sex"][rows] = sexes
        ordinals = [0, 0]
        for i, row in enumerate(rows):
            sex = int(sexes[i])
            ordinal = ordinals[sex]
            ordinals[sex] += 1
            d["ordinal"][row] = ordinal
            slot_id = address("slot", *key, sex, ordinal)
            self.id_rows[slot_id] = int(row)
            self.ids.append(slot_id)
            self.slot_keys.append([*key[2:], sex, ordinal])
            self.family.append(
                families[i]
                if families is not None
                else address("family", *key, "unallocated", sex, ordinal)
            )
            self._open_residence(int(row), site, year, phase)
            if parents is not None:
                mother, father = map(int, parents[i])
                d["mother"][row], d["father"][row] = mother, father
                for parent in (mother, father):
                    if parent >= 0:
                        self.children[parent].append(int(row))
                self.births.append(
                    {
                        "child_slot": int(row),
                        "mother_slot": mother,
                        "father_slot": father,
                        "family_key": self.family[row],
                        "site_id": self.site_ids[site],
                        "year": year,
                        "phase": 20,
                    }
                )
                self.counters[site, 0] += 1
        if flags is not None:
            d["flags"][rows] = flags
        return rows

    def _open_residence(self, row, site, year, phase):
        self.residence_index[row] = len(self.residences)
        self.residences.append(
            {
                "person_slot": row,
                "site_id": self.site_ids[site],
                "valid": {"start_year": year, "end_year": self.end},
                "start_phase": phase,
                "end_phase": 0,
                "end_reason": "coverage_end",
            }
        )

    def _close_residence(self, row, year, phase, reason):
        episode = self.residences[self.residence_index[row]]
        episode["valid"]["end_year"] = year
        episode.update(end_phase=phase, end_reason=reason)

    def die(self, rows, year, *, cause="annual_mortality"):
        d = self.data
        rows = list(map(int, rows))
        for row in rows:
            if d["site"][row] < 0 or d["death"][row] != NO_YEAR:
                raise ProviderError("InvalidPopulationResult", "Death requires a current resident")
            site = int(d["site"][row])
            d["death"][row], d["death_site"][row] = year, site
            self._close_residence(row, year, 30, "death")
            self.deaths.append(
                {
                    "person_slot": row,
                    "site_id": self.site_ids[site],
                    "year": year,
                    "phase": 30,
                    "cause": cause,
                }
            )
            self.counters[site, 1] += 1
            d["site"][row] = -1
            partner = self.partner.pop(row, None)
            if partner is not None:
                self.partner.pop(partner, None)
        if rows:
            self.living_rows = self.living_rows[d["death"][self.living_rows] == NO_YEAR]
            self._repair_death_memberships()

    def _columns(self, *fields):
        """Python copies of registry columns, for loops that read them row by row."""
        return [self.data[field][: self.n].tolist() for field in fields]

    def _commit_all(self, memberships, site_of, phase):
        """Commit every household holding or receiving members, in table order.

        A household that stays empty on its site has nothing to commit.
        """
        touched = set(memberships).union(*self.site_households)
        for hh in sorted(touched):
            if hh >= 0:
                members = tuple(memberships.get(hh, ()))
                self._commit_household(hh, members, site_of(hh, members), phase)

    def _repair_death_memberships(self):
        """Remove dead members and repair custody at phase 30, before new unions."""
        d, living = self.data, self.living()
        rows = living.tolist()
        grown = (self.year - d["birth"][living] >= self.config.society.dependent_age).tolist()
        adult_set = {row for row, adult in zip(rows, grown, strict=True) if adult}
        site_of, guardian_of, household_of, mother_of, father_of = self._columns(
            "site", "guardian", "household", "mother", "father"
        )
        memberships = defaultdict(list)
        for row, adult in zip(rows, grown, strict=True):
            hh = household_of[row]
            if not adult:
                site = site_of[row]
                guardian = guardian_of[row]
                if guardian not in adult_set or site_of[guardian] != site:
                    declared = [
                        p
                        for p in self.declared_guardians[row]
                        if p in adult_set and site_of[p] == site
                    ]
                    options = declared or [
                        parent
                        for parent in (mother_of[row], father_of[row])
                        if parent in adult_set and site_of[parent] == site
                    ]
                    guardian = min(options, key=lambda p: self.ids[p]) if options else -1
                    if guardian != guardian_of[row]:
                        self.guardians.append(
                            {
                                "child_slot": row,
                                "guardian_slot": guardian,
                                "year": self.year,
                                "phase": 30,
                                "source": "declared_guardian"
                                if declared
                                else "known_parent"
                                if guardian >= 0
                                else "synthetic_care",
                            }
                        )
                        d["guardian"][row] = guardian
                hh = (
                    household_of[guardian]
                    if guardian >= 0
                    else self.new_household(
                        ["care", self.site_ids[site]], site, "synthetic_care", "synthetic_care"
                    )
                )
                d["household"][row] = hh
            if hh >= 0:
                memberships[hh].append(row)
        self._commit_all(memberships, lambda hh, _: self.households[hh]["site"], 30)

    def new_household(self, key, site, kind, reason):
        household_id = address("household", self.domain, self.source, *key)
        if household_id not in self.hh_keys:
            self.hh_keys[household_id] = len(self.households)
            self.households.append(
                {
                    "household_id": household_id,
                    "site": site,
                    "kind": kind,
                    "reason": reason,
                    "members": (),
                }
            )
        return self.hh_keys[household_id]

    def _single(self, row, reason="independent_adult"):
        d = self.data
        key = ["single", *self.slot_keys[row], self.year]
        return self.new_household(key, int(d["site"][row]), "single", reason)

    def core_family_key(self, mother, father):
        return address("core_family", self.domain, *self.slot_keys[mother], *self.slot_keys[father])

    def partition(self, *, founder=False, force_single=(), phase=35):
        """Compute a complete destination partition, then commit atomically."""
        d, living = self.data, self.living()
        grown = self.year - d["birth"][living] >= self.config.society.dependent_age
        adults = living[grown]
        adult_rows = adults.tolist()
        adult_set, forced = set(adult_rows), set(map(int, force_single))
        established_adults = set(adults[d["adult_initialized"][adults] != 0].tolist())
        site_of, household_of = self._columns("site", "household")
        destinations = {}
        for row in adult_rows:
            partner = self.partner.get(row, -1)
            old = household_of[row]
            if partner in adult_set and site_of[partner] == site_of[row]:
                if (
                    old >= 0
                    and old == household_of[partner]
                    and self.households[old]["kind"] == "union"
                ):
                    hh = old
                else:
                    hh = destinations.get(partner)
                    if hh is None:
                        a, b = sorted((row, partner), key=lambda p: self.ids[p])
                        key = ["union", *self.slot_keys[a], *self.slot_keys[b], self.year]
                        hh = self.new_household(key, site_of[row], "union", "union")
                destinations[row] = hh
            else:
                keep = (
                    old >= 0
                    and row not in forced
                    and row in established_adults
                    and self.households[old]["kind"] != "synthetic_care"
                )
                if keep:
                    # Widowhood retains the survivor's identity; divorce is explicit.
                    occupants = [
                        p
                        for p in self.households[old]["members"]
                        if p in established_adults and self.partner.get(p, -1) < 0
                    ]
                    keep = len(occupants) <= 1
                destinations[row] = old if keep else self._single(row)
        d["adult_initialized"][adults] = 1
        dependants = living[~grown].tolist()
        if founder:
            self._founder_guardians(dependants, adults, destinations)
        guardian_of, mother_of, father_of = self._columns("guardian", "mother", "father")
        for row in dependants:
            site = site_of[row]
            guardian = guardian_of[row]
            if guardian not in destinations or site_of[guardian] != site:
                declared = [
                    p
                    for p in self.declared_guardians[row]
                    if p in destinations and site_of[p] == site
                ]
                options = declared or [
                    parent
                    for parent in (mother_of[row], father_of[row])
                    if parent in destinations and site_of[parent] == site
                ]
                guardian = min(options, key=lambda p: self.ids[p]) if options else -1
                if guardian != guardian_of[row]:
                    self.guardians.append(
                        {
                            "child_slot": row,
                            "guardian_slot": guardian,
                            "year": self.year,
                            "phase": 35,
                            "source": "declared_guardian"
                            if declared
                            else "known_parent"
                            if guardian >= 0
                            else "synthetic_care",
                        }
                    )
                    d["guardian"][row] = guardian
            if guardian >= 0:
                destinations[row] = destinations[guardian]
            else:
                destinations[row] = self.new_household(
                    ["care", self.site_ids[site]], site, "synthetic_care", "synthetic_care"
                )
        memberships = defaultdict(list)
        rows = living.tolist()
        chosen = [destinations[row] for row in rows]
        d["household"][living] = chosen
        for row, hh in zip(rows, chosen, strict=True):
            memberships[hh].append(row)

        def site_after(hh, members):
            return site_of[members[0]] if members else self.households[hh]["site"]

        self._commit_all(memberships, site_after, phase if not founder else 10)

    def _founder_guardians(self, dependants, adults, destinations):
        d = self.data
        profiles = {p["id"]: p for p in self.settings.kinship_profiles}
        members = defaultdict(list)
        for p in map(int, adults):
            members[destinations[p]].append(p)
        for site, place in enumerate(self.places):
            profile_id = self.settings.kinship_by_site.get(place.id)
            profile = profiles[profile_id] if profile_id else next(iter(profiles.values()))
            limit, parent_age = profile["founder_foster_children_limit"], profile["min_parent_age"]
            units = sorted(
                (min(int(d["birth"][p]) for p in people), self.households[hh]["household_id"], hh)
                for hh, people in members.items()
                if self.households[hh]["site"] == site
            )
            # Stable priority: older dependants first, semantic ID for ties.
            children = sorted(
                (p for p in dependants if d["site"][p] == site),
                key=lambda p: (int(d["birth"][p]), self.ids[p]),
            )
            heap, cursor = [], 0
            for row in children:
                threshold = int(d["birth"][row]) - parent_age
                while cursor < len(units) and units[cursor][0] <= threshold:
                    _, key, hh = units[cursor]
                    if limit:
                        heappush(heap, (0, key, hh))
                    cursor += 1
                if not heap:
                    continue
                load, key, hh = heappop(heap)
                declared = sorted(
                    (p for p in members[hh] if d["birth"][p] <= threshold),
                    key=lambda p: self.ids[p],
                )
                d["guardian"][row] = declared[0]
                self.declared_guardians[row] = declared
                for candidate in declared:
                    self.guardians.append(
                        {
                            "child_slot": row,
                            "guardian_slot": candidate,
                            "year": self.year,
                            "phase": 10,
                            "source": "founder_assignment",
                        }
                    )
                if load + 1 < limit:
                    heappush(heap, (load + 1, key, hh))

    def _commit_household(self, hh, members, site, phase):
        household = self.households[hh]
        if household["members"] == members and household["site"] == site:
            return
        self.site_households[household["site"]].discard(hh)
        if hh in self.hh_episode:
            episode = self.episodes[self.hh_episode[hh]]
            episode["valid"]["end_year"] = self.year
            episode["end_phase"] = phase
        household["members"], household["site"] = members, site
        if members:
            self.site_households[site].add(hh)
            self.hh_episode[hh] = len(self.episodes)
            self.episodes.append(
                {
                    "household_id": household["household_id"],
                    "member_slots": list(members),
                    "site_id": self.site_ids[site],
                    "valid": {"start_year": self.year, "end_year": self.end},
                    "start_phase": phase,
                    "end_phase": 0,
                    "formation_reason": household["reason"],
                    "kind": household["kind"],
                }
            )
        else:
            self.hh_episode.pop(hh, None)

    def eligible_households(self, site):
        return [
            hh
            for hh in sorted(self.site_households[site])
            if self.households[hh]["kind"] != "synthetic_care" and hh not in self.moved
        ]

    def transfer(self, hh, destination, *, reason, operation_id=None, port_id=None):
        if hh in self.moved:
            return False
        household = self.households[hh]
        members = tuple(household["members"])
        origin = household["site"]
        if not members or destination == origin or household["kind"] == "synthetic_care":
            return False
        d = self.data
        for row in members:
            if d["site"][row] != origin or d["death"][row] != NO_YEAR:
                raise ProviderError("InvalidPopulationResult", "Transfer has a nonresident member")
        transfer_id = address(
            "transfer",
            self.domain,
            self.source,
            self.year,
            reason,
            operation_id,
            port_id,
            *json.loads(household["household_id"])[3:],
            self.site_ids[origin],
            None if destination is None else self.site_ids[destination],
        )
        self.transfers.append(
            {
                "transfer_id": transfer_id,
                "household_id": household["household_id"],
                "origin_slots": list(members),
                "from_site": self.site_ids[origin],
                "to_site": None if destination is None else self.site_ids[destination],
                "port_id": port_id,
                "year": self.year,
                "phase": 40,
                "reason": reason,
                "operation_id": operation_id,
                "source": self.source,
            }
        )
        self.counters[origin, 3] += len(members)
        for row in members:
            self._close_residence(
                row, self.year, 40, "departed_external" if destination is None else "transfer"
            )
            if destination is None:
                d["site"][row] = -1
                d["departure"][row] = self.year
                d["coverage_end"][row] = self.year
            else:
                d["site"][row] = destination
                self._open_residence(row, destination, self.year, 40)
        self._commit_household(
            hh,
            () if destination is None else members,
            origin if destination is None else destination,
            40,
        )
        if destination is not None:
            self.counters[destination, 2] += len(members)
        else:
            self.living_rows = self.living_rows[d["departure"][self.living_rows] == NO_YEAR]
            member_set = set(members)
            for pair in self.open_pairs:
                if pair["state"] == "active" and (
                    pair["a"] in member_set or pair["b"] in member_set
                ):
                    pair["state"] = "closed"
                    union = self.unions[pair["union"]]
                    union["valid"]["end_year"] = self.year
                    union.update(end_phase=40, end_reason="coverage_departure")
                    self.partner.pop(pair["a"], None)
                    self.partner.pop(pair["b"], None)
        self.moved.add(hh)
        return True

    def leave_to_core(self, row):
        """Phase 40 exit of one resident who marries into the simulated core.

        The person leaves this component's coverage without a death or a site
        move; the core opens a new slot at the same site and stamp, so site
        totals are unchanged and no flow is counted. Any active union closes and
        the remaining household members keep their household.
        """
        d = self.data
        site = int(d["site"][row])
        if site < 0 or d["death"][row] != NO_YEAR:
            raise ProviderError("InvalidPopulationResult", "Core marriage requires a resident")
        self._close_residence(row, self.year, 40, "departed_external")
        d["site"][row] = -1
        d["departure"][row] = d["coverage_end"][row] = self.year
        self.living_rows = self.living_rows[self.living_rows != row]
        for pair in self.pairs:
            if pair["state"] == "active" and row in (pair["a"], pair["b"]):
                pair["state"] = "closed"
                self.former_partners.add(frozenset((pair["a"], pair["b"])))
                union = self.unions[pair["union"]]
                union["valid"]["end_year"] = self.year
                union.update(end_phase=40, end_reason="coverage_departure")
        partner = self.partner.pop(row, None)
        if partner is not None:
            self.partner.pop(partner, None)
        hh = int(d["household"][row])
        members = tuple(p for p in self.households[hh]["members"] if p != row)
        self._commit_household(hh, members, site, 40)
        d["household"][row] = -1

    def add_pair(self, pair):
        pair["encoded_key"] = encoded(pair["key"])
        self.pairs.append(pair)
        self.open_pairs.append(pair)

    def seal_pairs(self, rows):
        """Stable potential pairs over one sealed cohort, without forming unions."""
        d = self.data
        cohort = int(d["cohort"][rows[0]])
        key = self.cohorts[cohort]["origin_cohort_id"]
        rng = stream(self.config.seed, "potential_pairs", key)
        men, women = [rng.permutation(rows[d["sex"][rows] == sex]) for sex in (0, 1)]
        for ordinal, (man, woman) in enumerate(zip(men, women)):
            pair = {
                "key": address(
                    "family", self.domain, self.source, *self.origin_keys[cohort], ordinal
                ),
                "a": int(man),
                "b": int(woman),
                "state": "unformed",
                "formed": None,
                "last_birth": self.start - 10000,
                "union": None,
            }
            self.sealed_pair[int(man)] = self.sealed_pair[int(woman)] = pair
            self.add_pair(pair)

    def activate_pairs(self, *, initial=False):
        society = self.config.society
        site_of, death_of, birth_of = self._columns("site", "death", "birth")
        self.open_pairs = [pair for pair in self.open_pairs if pair["state"] != "closed"]
        low, high = society.marriage_min_age, society.marriage_max_age
        divorce_key = self.key_prefix + b'"divorce",'
        union_key = self.key_prefix + b'"form_union",'
        tail = b",%d]" % self.year
        steps, keys = [], []
        for pair in self.open_pairs:
            a, b = pair["a"], pair["b"]
            if pair["state"] == "active":
                steps.append(pair)
                if not initial:
                    keys.append(divorce_key + pair["encoded_key"] + tail)
                continue
            if site_of[a] < 0 or site_of[b] < 0 or site_of[a] != site_of[b]:
                if death_of[a] != NO_YEAR or death_of[b] != NO_YEAR:
                    pair["state"] = "closed"
                continue
            if low <= self.year - birth_of[a] <= high and low <= self.year - birth_of[b] <= high:
                steps.append(pair)
                keys.append(union_key + pair["encoded_key"] + tail)
        # Keyed draws do not depend on the order they are taken in.
        draws = iter(opening_draws(keys)[0].tolist())
        rate = society.founder_match_participation if initial else society.marriage_rate
        forced = []
        for pair in steps:
            a, b = pair["a"], pair["b"]
            if pair["state"] == "active":
                present = site_of[a] >= 0 and site_of[b] >= 0
                dead = death_of[a] != NO_YEAR or death_of[b] != NO_YEAR
                divorced = not initial and next(draws) < society.divorce_rate
                if dead or divorced or not present:
                    pair["state"] = "closed"
                    self.former_partners.add(frozenset((a, b)))
                    self.unions[pair["union"]]["valid"]["end_year"] = self.year
                    self.unions[pair["union"]]["end_phase"] = 30 if dead else 35
                    self.unions[pair["union"]]["end_reason"] = "death" if dead else "divorce"
                    self.partner.pop(a, None)
                    self.partner.pop(b, None)
                    if divorced and not dead:
                        forced.extend((a, b))
                continue
            if next(draws) >= rate:
                continue
            pair.update(state="active", formed=self.year, union=len(self.unions))
            self.partner[a], self.partner[b] = b, a
            self.unions.append(
                {
                    "union_id": address("union", *json.loads(pair["key"])[1:]),
                    "partner_slots": [a, b],
                    "family_key": pair["key"],
                    "valid": {"start_year": self.year, "end_year": self.end},
                    "start_phase": 10 if initial else 35,
                    "end_phase": 0,
                    "end_reason": "coverage_end",
                    "site_at_formation": self.site_ids[site_of[a]],
                    "source": "projected",
                }
            )
        return forced

    def _kin(self, a, b, known=None):
        """Shared ancestor within the society's kinship depth, or direct descent."""
        d, depth = self.data, self.config.society.kinship_depth
        known = {} if known is None else known

        def lineage(row):
            people = known.get(row)
            if people is None:
                people, frontier = {row}, [row]
                for _ in range(depth):
                    frontier = [
                        int(d[f][p]) for p in frontier for f in ("mother", "father") if d[f][p] >= 0
                    ]
                    people.update(frontier)
                known[row] = people
            return people

        return not lineage(a).isdisjoint(lineage(b))

    def _awaits_sealed_partner(self, row, site_of, birth_of):
        pair = self.sealed_pair.get(row)
        if pair is None or pair["state"] != "unformed":
            return False
        society = self.config.society
        other = pair["a"] if pair["b"] == row else pair["b"]
        age = self.year - birth_of[other]
        return (
            site_of[other] == site_of[row]
            and other not in self.partner
            and society.marriage_min_age <= age <= society.marriage_max_age
        )

    def repair_local(self):
        """Phase 35 local re-pairing, `chronos-local-repairing-1`.

        Sealed pairs are drawn at birth inside one cohort, so a partner's death,
        a sex surplus or a move would otherwise leave the survivor single for
        life. Adults with no union and no reachable sealed partner form a
        site/ancestry pool. Women enter it at the society's marriage rate and men
        always, so a union draws that hazard once, as a sealed pair does. Each
        pool is matched in age order, within the age gap, outside kin and former
        partners. Draws are keyed on slot and year, never on table order.
        """
        society = self.config.society
        site_of, birth_of, sex_of, race_of = self._columns("site", "birth", "sex", "race")
        low, high = society.marriage_min_age, society.marriage_max_age
        pools = defaultdict(lambda: ([], []))
        prefix = self.key_prefix + encoded(REPAIRING_VERSION) + b","
        tail = b",%d]" % self.year
        single, keys = [], []
        for row in self.living().tolist():
            if (
                row in self.partner
                or not low <= self.year - birth_of[row] <= high
                or self._awaits_sealed_partner(row, site_of, birth_of)
            ):
                continue
            single.append(row)
            if sex_of[row] != 0:
                keys.append(prefix + encoded(self.ids[row]) + tail)
        # Keyed draws do not depend on the order they are taken in.
        draws = iter(opening_draws(keys)[0].tolist())
        for row in single:
            sex = sex_of[row]
            if sex == 0 or next(draws) < society.marriage_rate:
                pools[site_of[row], race_of[row]][sex].append(row)
        lineages = {}
        for (site, _), (men, women) in sorted(pools.items()):
            men.sort(key=lambda p: (birth_of[p], self.ids[p]))
            women.sort(key=lambda p: (birth_of[p], self.ids[p]))
            for woman in women:
                for index, man in enumerate(men):
                    gap = birth_of[man] - birth_of[woman]
                    if gap > society.max_age_gap:
                        break
                    if (
                        gap >= -society.max_age_gap
                        and frozenset((man, woman)) not in self.former_partners
                        and not self._kin(man, woman, lineages)
                    ):
                        del men[index]
                        self._repair_union(man, woman, site)
                        break

    def _repair_union(self, man, woman, site):
        for row in (man, woman):
            pair = self.sealed_pair.get(row)
            if pair is not None and pair["state"] == "unformed":
                pair["state"] = "closed"
        key = address(
            "family",
            self.domain,
            self.source,
            "local_repair",
            *self.slot_keys[man],
            *self.slot_keys[woman],
            self.year,
        )
        self.add_pair(
            {
                "key": key,
                "a": man,
                "b": woman,
                "state": "active",
                "formed": self.year,
                "last_birth": int(self.data["last_birth"][woman]),
                "union": len(self.unions),
            }
        )
        self.partner[man], self.partner[woman] = woman, man
        self.unions.append(
            {
                "union_id": address("union", *json.loads(key)[1:]),
                "partner_slots": [man, woman],
                "family_key": key,
                "valid": {"start_year": self.year, "end_year": self.end},
                "start_phase": 35,
                "end_phase": 0,
                "end_reason": "coverage_end",
                "site_at_formation": self.site_ids[site],
                "source": "projected",
            }
        )
