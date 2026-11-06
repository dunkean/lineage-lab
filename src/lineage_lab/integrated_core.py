"""Independent core precomputation with a phase-stamped resident event tape.

The Engine still chooses simulated births, deaths, unions and ordinary movement.
Integrated household commits supply founder custody and once-per-phase movement.
No projected population, closing background state or controller enters this core.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy

import numpy as np

from .cohort_residence import ResidentRegistry
from .config import Scenario
from .engine import Engine
from .projection import derive_seed
from .registry_contract import NO_YEAR, address


class IntegratedCore(Engine):
    def __init__(self, config, domain_config):
        self.registry = ResidentRegistry(domain_config, "simulated")
        self.slot_map, self.person_map, self.union_map = {}, {}, {}
        self.pending_marriages = []
        self.annual, self.operations = {}, []
        self.domain_config = domain_config
        self.tape = []
        super().__init__(config)

    def _sync_slots(self):
        r, d = self.registry, self.data
        r.year = self.year
        groups = defaultdict(list)
        for person in range(len(self.slot_map), self.n):
            groups[
                int(d["birth_place"][person]), int(d["race"][person]), int(d["birth"][person])
            ].append(person)
        for (place, race, birth), people in sorted(groups.items()):
            founder = birth <= self.config.start_year and not self.slot_map
            # Founder classification is based on engine parentage, including age-zero founders.
            founder = all(d["mother"][p] < 0 and d["father"][p] < 0 for p in people)
            parents = (
                None
                if founder
                else [
                    (self.slot_map[int(d["mother"][p])], self.slot_map[int(d["father"][p])])
                    for p in people
                ]
            )
            families = None if founder else [r.core_family_key(a, b) for a, b in parents]
            rows = r.add_cohort(
                "founder" if founder else "birth",
                r.local_index[place],
                race,
                birth,
                d["sex"][people],
                parents=parents,
                families=families,
                flags=d["reproduction_flags"][people],
            )
            for person, row in zip(people, rows, strict=True):
                self.slot_map[person], self.person_map[int(row)] = int(row), person
                if not founder:
                    mother = int(r.data["mother"][row])
                    r.data["guardian"][row] = mother
                    r.data["household"][row] = r.data["household"][mother]
        dead = [
            row
            for person, row in self.slot_map.items()
            if d["death"][person] != NO_YEAR and r.data["death"][row] == NO_YEAR
        ]
        if dead:
            r.die(dead, self.year)

    def _sync_unions(self):
        r = self.registry
        for uid, a, b, year, site in self.new_unions:
            if uid in self.union_map:
                continue
            a, b = self.slot_map[a], self.slot_map[b]
            self.union_map[uid] = len(r.unions)
            r.unions.append(
                {
                    "union_id": address(
                        "core_union", r.domain, year, *r.slot_keys[a], *r.slot_keys[b]
                    ),
                    "partner_slots": [a, b],
                    "family_key": r.core_family_key(b, a),
                    "valid": {"start_year": year, "end_year": r.end},
                    "start_phase": 10 if year == r.start else 35,
                    "end_phase": 0,
                    "end_reason": "coverage_end",
                    "site_at_formation": self.registry.settings.site_ids[site],
                    "source": "simulated",
                }
            )
        divorced = []
        for year, reason, uid in self.closed_unions:
            if uid not in self.union_map:
                continue
            union = r.unions[self.union_map[uid]]
            union["valid"]["end_year"] = year
            union.update(end_phase=30 if reason == "death" else 35, end_reason=reason)
            if reason == "divorce":
                divorced.extend(union["partner_slots"])
        r.partner = {
            self.slot_map[int(p)]: self.slot_map[int(self.data["partner"][p])]
            for p in self.alive
            if self.data["partner"][p] >= 0
        }
        return divorced

    def _births(self):
        births, maternal = super()._births()
        self._sync_slots()
        if births:
            r = self.registry
            members = defaultdict(list)
            for row in map(int, r.living()):
                members[int(r.data["household"][row])].append(row)
            for hh, people in members.items():
                if hh >= 0:
                    r._commit_household(hh, tuple(people), r.households[hh]["site"], 20)
        return births, maternal

    def _deaths(self, maternal):
        answer = super()._deaths(maternal)
        self._sync_slots()
        return answer

    def _move_households_batch(self, adults, slots, reason):
        if reason == "marriage":
            self.pending_marriages.extend(
                (int(p), int(s)) for p, s in zip(adults, slots, strict=True)
            )
        else:
            super()._move_households_batch(adults, slots, reason)

    def _marriages(self, initial=False):
        self._sync_slots()
        if not initial:
            self.registry.partition(force_single=self._sync_unions())
        result = super()._marriages(initial)
        self._sync_unions()
        self.registry.partition(founder=initial)
        return result

    def _move_registered(self, hh, destination, reason, operation_id=None):
        r = self.registry
        members = r.households[hh]["members"]
        if r.transfer(hh, destination, reason=reason, operation_id=operation_id):
            people = np.array([self.person_map[p] for p in members], np.int64)
            slots = np.full(len(people), destination, np.int32)
            super()._apply_moves_batch(people, slots, reason)
            return True
        return False

    def _commit_marriages(self):
        r = self.registry
        affected = set()
        for person, destination in self.pending_marriages:
            row = self.slot_map[person]
            if r.data["site"][row] < 0 or self.data["partner"][person] < 0:
                continue
            hh = int(r.data["household"][row])
            if r.households[hh]["site"] != destination:
                members = r.households[hh]["members"]
                if self._move_registered(hh, destination, "marriage"):
                    affected.update(members)
        self.pending_marriages.clear()
        r.partition(phase=40)
        # A marriage may replace singleton identities with a union identity.
        r.moved.update(
            int(r.data["household"][row]) for row in affected if r.data["site"][row] >= 0
        )

    def _migrations(self):
        r = self.registry
        self._commit_marriages()
        self._programmed()
        counts = r.counts()
        factors = self.program.factors(self.year, self.place_ids)
        attraction = np.array([p.capacity for p in self.settlements]) * factors[4] / (counts + 1)
        for site, place in enumerate(self.settlements):
            for hh in list(r.eligible_households(site)):
                if self.rng.random() >= min(1, self.society.migration_rate * factors[3][site]):
                    continue
                options = [
                    (distance, self.place_index[p])
                    for distance, p in self.neighbors[place.id]
                    if distance <= self.society.migration_radius
                ]
                if not options:
                    continue
                weights = np.array(
                    [
                        np.exp(-distance / self.society.distance_scale) * attraction[p]
                        for distance, p in options
                    ]
                )
                if weights.sum() > 0:
                    dest = options[int(self.rng.choice(len(options), p=weights / weights.sum()))][1]
                    self._move_registered(hh, dest, "ordinary")

    def _programmed(self):
        r = self.registry
        # Fixed preallocation. Ports are projected-only, so no core quota there.
        share = self.domain_config.historical.limits["core_target_share_bps"]
        for op in sorted(
            self.domain_config.historical.historical_program, key=lambda op: op["op_id"]
        ):
            if op["year"] != self.year:
                continue
            quota = (
                op["target_households"] * share // 10000
                if op["destination_site_id"] is not None
                else 0
            )
            origin = r.public_index[op["origin_site_id"]]
            destination = r.public_index.get(op["destination_site_id"])
            moved = 0
            choices = r.eligible_households(origin)
            for hh in self.rng.permutation(choices)[:quota]:
                moved += self._move_registered(int(hh), destination, "programmed", op["op_id"])
            self.operations.append(
                {
                    "op_id": op["op_id"],
                    "year": self.year,
                    "phase": 40,
                    "allocated_core_households": quota,
                    "actual_core_households": moved,
                }
            )

    def _end_union(self, identities, reason):
        result = super()._end_union(identities, reason)
        # Bootstrap flushes Engine buffers before its first census. Capture the
        # closure while it is still present, including initial-shock widowhood.
        self._sync_unions()
        return result

    def _random_stream(self, process):
        if process == "births":
            self.registry.open_year(self.year)
        return super()._random_stream(process)

    def _census(self, *args):
        self._sync_slots()
        forced = self._sync_unions()
        if self.year == self.config.start_year:
            self.registry.partition(force_single=forced)
            self._commit_marriages()
            self._random_stream("migrations")
            self._programmed()
        super()._census(*args)
        r = self.registry
        self.annual[self.year] = {
            "counts": r.counts().copy(),
            "counters": r.counters.copy(),
            "historical": r.n,
            "cohort_counts": r.cohort_counts(),
        }
        # The tape is consumable by effective phase, and includes real household transfers.
        for kind, rows in (
            ("birth", r.births),
            ("death", r.deaths),
            ("transfer", r.transfers),
            ("guardian", r.guardians),
        ):
            for record in rows:
                if record["year"] == self.year:
                    self.tape.append({"kind": kind, **record})
        for kind, records, id_field in (
            ("household", r.episodes, "household_id"),
            ("union", r.unions, "union_id"),
        ):
            for record in records:
                for boundary in ("start", "end"):
                    if record["valid"][boundary + "_year"] == self.year:
                        self.tape.append(
                            {
                                "kind": kind + "_" + boundary,
                                id_field: record[id_field],
                                "year": self.year,
                                "phase": record[boundary + "_phase"],
                                "record": deepcopy(record),
                            }
                        )


def precompute_core(config):
    size = config.initial_population * config.historical.limits["core_target_share_bps"] // 10000
    if size < 2:
        return None, 0
    source = config.model_dump(mode="json")
    source.update(
        population_model="full",
        hybrid=None,
        initial_population=size,
        seed=derive_seed(config.seed, "core"),
    )
    source["society"]["status_weights"] = (
        config.hybrid.core_status_weights or config.society.status_weights
    )
    for place in source["settlements"]:
        place["capacity"] = max(1, round(place["capacity"] * size / config.initial_population))
    core = IntegratedCore(Scenario.model_validate(source), config)
    core.run()
    core.tape.sort(key=lambda e: (e["year"], e["phase"], e["kind"]))
    return core, size
