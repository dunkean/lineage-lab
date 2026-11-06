"""Annual projected transitions over actual present slots, never survival prefixes."""

from __future__ import annotations

from collections import defaultdict
from itertools import pairwise

import numpy as np

from .cohorts import Cohorts
from .engine import mortality_table, scaled_probability
from .historical_program import HistoricalProgram, mortality_probability
from .registry_contract import MORTALITY_VERSION, stream
from .registry_kernels import annual_ranks, radix_order


class RegistryDemography:
    def __init__(self, registry):
        self.registry = registry
        self.config = registry.config
        self.program = HistoricalProgram(self.config)
        # Policy resolution is shared with the existing core; no aggregate lives
        # or allocations from Cohorts.bounds are used by this integrated path.
        self.policy = Cohorts(self.config, self.config.settlements).policy
        self.exposure = {
            "eligible_pair_years": 0,
            "active_pair_years": 0,
            "separated_potential_pair_years": 0,
            "births": 0,
        }

    def traits(self, rows):
        r, d = self.registry, self.registry.data
        demos, society = self.policy(r.year)
        for race, demo in enumerate(demos):
            for sex in (0, 1):
                group = rows[(d["race"][rows] == race) & (d["sex"][rows] == sex)]
                if not len(group):
                    continue
                cohort = int(d["cohort"][group[0]])
                rng = stream(self.config.seed, "traits", r.cohorts[cohort]["origin_cohort_id"], sex)
                infertility = (
                    demo.male_infertility_rate if sex == 0 else demo.female_infertility_rate
                )
                d["flags"][group] = (rng.random(len(group)) < infertility).astype(np.int32) | (
                    (rng.random(len(group)) < society.childfree_rate).astype(np.int32) * 2
                )

    def seed(self, core_size):
        r = self.registry
        founder_cohorts = Cohorts(self.config, self.config.settlements)
        founder_cohorts.seed_founders(self.config.initial_population - core_size)
        for cohort in founder_cohorts.records:
            sexes = np.repeat([0, 1], cohort["counts"])
            rows = r.add_cohort(
                "founder", r.local_index[cohort["place"]], cohort["race"], cohort["birth"], sexes
            )
            self.traits(rows)
            r.seal_pairs(rows)
        r.activate_pairs(initial=True)
        r.partition(founder=True)

    def births(self, multiplier_ppm=None, opening_core=None):
        r, d = self.registry, self.registry.data
        demos, society = self.policy(r.year)
        counts = r.counts() + (0 if opening_core is None else opening_core)
        factors = self.program.factors(r.year, [p.id for p in r.places])
        capacities = np.array([p.capacity for p in r.places], float) * factors[4]
        capacities *= (1 + society.capacity_growth) ** (r.year - r.start)
        pressure = np.minimum(1, capacities / np.maximum(counts, 1))
        multiplier_ppm = (
            np.full(len(r.places), 1000000) if multiplier_ppm is None else multiplier_ppm
        )
        cap = self.config.historical.regulation["max_birth_probability_bps"] / 10000
        allocations = defaultdict(list)
        mothers = []
        eligible_by_site = np.zeros(len(r.places), np.int64)
        for pair in r.pairs:
            a, b = pair["a"], pair["b"]
            if (
                pair["state"] == "unformed"
                and d["site"][a] >= 0
                and d["site"][b] >= 0
                and d["site"][a] != d["site"][b]
            ):
                self.exposure["separated_potential_pair_years"] += 1
            if pair["state"] != "active" or pair["formed"] >= r.year:
                continue
            if d["site"][a] < 0 or d["site"][a] != d["site"][b]:
                continue
            self.exposure["active_pair_years"] += 1
            mother = b if d["sex"][b] == 1 else a
            father = a if mother == b else b
            race = int(d["race"][mother])
            demo = demos[race]
            age = r.year - int(d["birth"][mother])
            if (
                d["flags"][a]
                or d["flags"][b]
                or not demo.fertility_min_age <= age <= demo.fertility_max_age
                or r.year - pair["last_birth"] < demo.birth_spacing
            ):
                continue
            self.exposure["eligible_pair_years"] += 1
            site = int(d["site"][mother])
            base = demo.fertility_peak * np.exp(
                -0.5 * ((age - demo.fertility_peak_age) / demo.fertility_width) ** 2
            )
            probability = float(scaled_probability(base, factors[2][site] * pressure[site]))
            if probability <= 0:
                continue
            eligible_by_site[site] += 1
            multiplier = multiplier_ppm[site] / 1000000
            if multiplier != 1:
                corrected = probability * multiplier
                # A cap only limits corrective increases; identity preserves the
                # entire unregulated probability, even above the corrective cap.
                probability = min(max(probability, cap), corrected) if multiplier > 1 else corrected
            if stream(self.config.seed, "birth", pair["key"], r.year).random() >= probability:
                continue
            pair["last_birth"] = r.year
            d["last_birth"][mother] = r.year
            sex = int(
                stream(self.config.seed, "birth_sex", pair["key"], r.year).random()
                >= demo.male_birth_probability
            )
            allocations[site, race].append((mother, father, pair["key"], sex))
            mothers.append(mother)
        for (site, race), entries in sorted(allocations.items()):
            rows = r.add_cohort(
                "birth",
                site,
                race,
                r.year,
                [e[3] for e in entries],
                parents=[e[:2] for e in entries],
                families=[e[2] for e in entries],
            )
            self.traits(rows)
            r.seal_pairs(rows)
            # Newborn membership exists before deaths, including birth-year deaths.
            for child, entry in zip(rows, entries, strict=True):
                mother = entry[0]
                d["guardian"][child] = mother
                d["household"][child] = d["household"][mother]
            self.exposure["births"] += len(rows)
        # Commit newborns without applying majorities or union changes before phase 35.
        if mothers:
            membership = defaultdict(list)
            for row in map(int, r.living()):
                membership[int(d["household"][row])].append(row)
            for hh, members in membership.items():
                if hh >= 0:
                    r._commit_household(hh, tuple(members), r.households[hh]["site"], 20)
        return np.asarray(mothers, np.int32), eligible_by_site

    def deaths(self, mothers=(), *, initial=False):
        r, d = self.registry, self.registry.data
        if initial:
            if getattr(r, "initial_program_applied", False):
                raise RuntimeError("Initial historical program has already been applied")
            r.initial_program_applied = True
        living = r.living()
        if not len(living):
            return
        demos, _ = self.policy(r.year)
        maternal = np.zeros(r.n, bool)
        maternal[np.asarray(mothers, np.int32)] = True
        size = len(r.places)
        # Cohort fixes age/ancestry. Maternal exposure is a separate hazard group.
        codes = (
            (
                (d["cohort"][living].astype(np.int64) * size + d["site"][living]) * 2
                + d["sex"][living]
            )
            * 2
            + maternal[living]
        ).astype(np.uint64)
        order = radix_order(codes)
        ordered = living[order]
        keys = codes[order]
        boundaries = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1]) + 1, len(keys)]
        dead = []
        factors = self.program.factors(r.year, [p.id for p in r.places])
        for left, right in pairwise(boundaries):
            group = ordered[left:right]
            sample = int(group[0])
            site, race, sex, cohort = [int(d[f][sample]) for f in ("site", "race", "sex", "cohort")]
            demo = demos[race]
            age = r.year - int(d["birth"][sample])
            base = mortality_table(demo)[min(age, demo.max_age)]
            base = min(1, base * (demo.male_mortality_factor if sex == 0 else 1))
            probability = float(
                mortality_probability(base, factors[0][site], factors[1][site], initial=initial)
            )
            if not initial and maternal[sample]:
                probability = 1 - (1 - probability) * (1 - demo.maternal_mortality)
            if age >= demo.max_age:
                probability = 1
            key = [
                MORTALITY_VERSION,
                r.cohorts[cohort]["origin_cohort_id"],
                r.site_ids[site],
                sex,
                r.year,
                bool(maternal[sample]),
            ]
            rng = stream(self.config.seed, *key)
            quota = int(np.floor(len(group) * probability + rng.random()))
            if quota:
                salt = rng.integers(0, 2**64, dtype=np.uint64)
                ranks = annual_ranks(d["ordinal"][group], salt)
                dead.extend(group[radix_order(ranks)[:quota]].tolist())
        r.die(dead, r.year, cause="initial_shock_delta" if initial else "annual_mortality")
