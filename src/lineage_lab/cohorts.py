"""Population counts at construction; ordinary persons exist only at query time.

One row represents a place/race/birth-year cohort, with two sex counts. A
household ordinal pairs two independently permuted sex slots. Annual births
select distinct eligible household ordinals; the inverse gives the same child
from either parent. No ordinary person, life, union or pedigree is enumerated
while building the world.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from .config import Demography, Society
from .historical_program import (
    HistoricalProgram,
    ProviderError,
    apply_initial_program,
    mortality_probability,
)
from .projection import inverse, permutation, permute, uniforms


def apportion(total, weights):
    weights = np.asarray(weights, float)
    quotas = total * weights / weights.sum()
    counts = np.floor(quotas).astype(np.int64)
    order = np.argsort(-(quotas - counts), kind="stable")
    counts[order[: total - int(counts.sum())]] += 1
    return counts


class Cohorts:
    def __init__(self, config, settlements, records=None, factors=None):
        self.config, self.settlements = config, settlements
        self.program = HistoricalProgram(config) if config.historical else None
        self.seed, self.start = config.seed, config.start_year
        self.end = self.start + config.duration
        self.records = records or []
        self.factors = factors or [[1.0, 1.0] for _ in range(config.duration + 1)]
        self._policies, self._curves, self._births = {}, OrderedDict(), OrderedDict()
        self._columns = None
        self._column_storage = None
        self._place_ids = {p.id: i for i, p in enumerate(settlements)}
        self._lookup = {(r["place"], r["race"], r["birth"]): i for i, r in enumerate(self.records)}
        self.spacing = max(
            [config.demography.birth_spacing]
            + [
                p.demography.birth_spacing
                for p in config.periods
                if p.demography.birth_spacing is not None
            ]
            + [r.demography.birth_spacing or config.demography.birth_spacing for r in config.races]
        )

    def policy(self, year):
        key = tuple(p.start_year for p in self.config.periods if p.start_year <= year)
        if key not in self._policies:
            demo, society = self.config.demography, self.config.society
            for period in sorted(self.config.periods, key=lambda p: p.start_year):
                if period.start_year <= year:
                    demo = Demography.model_validate(
                        {**demo.model_dump(), **period.demography.model_dump(exclude_none=True)}
                    )
                    society = Society.model_validate(
                        {**society.model_dump(), **period.society.model_dump(exclude_none=True)}
                    )
            races = [
                Demography.model_validate(
                    {**demo.model_dump(), **r.demography.model_dump(exclude_none=True)}
                )
                for r in self.config.races
            ]
            self._policies[key] = races, society
        return self._policies[key]

    def seed_founders(self, total):
        demos, _ = self.policy(self.start)
        weights, keys = [], []
        race_names = [r.name for r in self.config.races]
        for p in self.settlements:
            race_weights = (
                [p.races.get(n, 0) for n in race_names]
                if p.races
                else [r.initial_weight for r in self.config.races]
            )
            for race, demo in enumerate(demos):
                ages = np.arange(demo.max_age)
                mortality = self._mortality(demo, ages)
                survivals = [
                    np.cumprod(1 - np.minimum(1, mortality * demo.male_mortality_factor)),
                    np.cumprod(1 - mortality),
                ]
                for sex, survival in enumerate(survivals):
                    sex_weight = (
                        demo.male_birth_probability if sex == 0 else 1 - demo.male_birth_probability
                    )
                    for age, value in enumerate(survival / survival.sum()):
                        keys.append((p.id, race, self.start - age, sex))
                        weights.append(
                            p.initial_weight
                            * race_weights[race]
                            / sum(race_weights)
                            * sex_weight
                            * value
                        )
        counts = apportion(total, weights)
        if self.program:
            initial = [p.metadata["chronos_initial_population"] for p in self.settlements]
            core = apportion(self.config.initial_population - total, initial)
            counts[:] = 0
            for p, target in zip(self.settlements, np.array(initial) - core, strict=True):
                positions = [i for i, key in enumerate(keys) if key[0] == p.id]
                counts[positions] = apportion(int(target), np.asarray(weights)[positions])
        combined = {}
        for (place, race, birth, sex), count in zip(keys, counts, strict=True):
            if count:
                record = combined.setdefault((place, race, birth), [0, 0])
                record[sex] = int(count)
        for (place, race, birth), counts in sorted(combined.items()):
            self.append(place, race, birth, counts)

    def append(self, place, race, birth, counts):
        index = len(self.records)
        if self.program and index >= self.config.historical.limits["max_cohorts"]:
            raise ProviderError(
                "BudgetExceeded", "Cohort budget exceeded", constraint="max_cohorts"
            )
        demos, society = self.policy(max(self.start, birth))
        d = demos[race]
        trait_lower = max(
            int(counts[0] * d.male_infertility_rate),
            int(counts[1] * d.female_infertility_rate),
        ) + int(min(counts) * society.childfree_rate)
        self.records.append(
            {
                "place": place,
                "race": race,
                "birth": birth,
                "counts": list(map(int, counts)),
                "trait_lower": trait_lower,
            }
        )
        self._lookup[place, race, birth] = index
        if self._column_storage is not None:
            if index >= len(self._column_storage["birth"]):
                for name, old in self._column_storage.items():
                    shape = (len(old) * 2, 2) if name == "counts" else len(old) * 2
                    grown = np.empty(shape, np.int64)
                    grown[: len(old)] = old
                    self._column_storage[name] = grown
            values = {
                "place": place,
                "race": race,
                "birth": birth,
                "place_index": self._place_ids[place],
                "counts": counts,
                "trait_lower": trait_lower,
            }
            for name, value in values.items():
                self._column_storage[name][index] = value
        self._columns = None

    def columns(self):
        if self._column_storage is None:
            capacity = len(self.records) + max(
                256,
                min(8192, len(self.settlements) * len(self.config.races) * self.config.duration),
            )
            self._column_storage = {
                name: np.empty((capacity, 2) if name == "counts" else capacity, np.int64)
                for name in ("birth", "race", "place", "place_index", "counts", "trait_lower")
            }
            for i, r in enumerate(self.records):
                for name in self._column_storage:
                    self._column_storage[name][i] = (
                        self._place_ids[r["place"]] if name == "place_index" else r[name]
                    )
        if self._columns is None:
            self._columns = {
                name: values[: len(self.records)] for name, values in self._column_storage.items()
            }
        return self._columns

    def event_mask(self, event, columns, indices, ages):
        mask = (ages >= event.min_age) & (ages <= event.max_age)
        if event.settlements:
            mask &= np.isin(columns["place"][indices], event.settlements)
        if event.races:
            races = [i for i, r in enumerate(self.config.races) if r.name in event.races]
            mask &= np.isin(columns["race"][indices], races)
        return mask

    @staticmethod
    def _mortality(demo, ages):
        rates = demo.adult_mortality + demo.aging_coefficient * np.exp(
            np.minimum(demo.aging_exponent * ages, 700)
        )
        rates = np.where(ages <= demo.child_max_age, demo.child_mortality, rates)
        rates = np.where(ages == 0, demo.infant_mortality, rates)
        return np.where(ages >= demo.max_age, 1.0, np.clip(rates, 0, 1))

    def record_site_pressure(self, year, previous):
        """Publish annual capacity pressure once; queries reuse the same context."""
        columns = self.columns()
        by_place = np.bincount(
            columns["place_index"],
            weights=previous[:, :2].sum(axis=1),
            minlength=len(self.settlements),
        )
        share = sum(sum(r["counts"]) for r in self.records if r["birth"] <= self.start)
        share /= self.config.initial_population
        places = [p.id for p in self.settlements]
        capacity = self.program.factors(year, places)[4]
        society = self.policy(year)[1]
        capacity *= (1 + society.capacity_growth) ** (year - self.start)
        capacities = np.array([p.capacity for p in self.settlements]) * share * capacity
        pressure = np.minimum(1, capacities / np.maximum(1, by_place))
        factors = self.factors[year - self.start]
        values = {str(p): float(q) for p, q in zip(places, pressure, strict=True)}
        if len(factors) == 2:
            factors.append(values)
        else:
            factors[2] = values

    def bounds(self, indices, year, previous):
        """Birth selection is a periodic range, so spacing has a closed inverse."""
        indices = np.asarray(indices, dtype=np.int64)
        demos, _society = self.policy(year)
        columns = self.columns()
        births, race_ids = columns["birth"][indices], columns["race"][indices]

        def parameter(name):
            return np.array([getattr(d, name) for d in demos])[race_ids]

        lower = np.maximum(columns["trait_lower"][indices], previous[:, 3])
        lower += (year - lower) % self.spacing
        upper = np.minimum(previous[:, :2].min(axis=1), previous[:, 2])
        available = np.maximum(0, (upper - 1 - lower) // self.spacing + 1)
        ages = year - births
        probability = parameter("fertility_peak") * np.exp(
            -0.5 * ((ages - parameter("fertility_peak_age")) / parameter("fertility_width")) ** 2
        )
        valid = (ages >= parameter("fertility_min_age")) & (ages <= parameter("fertility_max_age"))
        for event in self.config.events:
            if event.start_year <= year < event.end_year:
                mask = self.event_mask(event, columns, indices, ages)
                if event.sex in {"all", "female"}:
                    probability[mask] *= event.fertility_factor
        if self.program:
            _, _, fertility, _, _ = self.program.factors(year, columns["place"][indices])
            factors = self.factors[year - self.start]
            if len(factors) < 3 and np.any(available):
                if len(indices) != len(self.records):
                    raise ProviderError(
                        "InvalidPopulationResult",
                        "Missing annual site pressure for cohort reconstruction",
                    )
                self.record_site_pressure(year, previous)
            pressure = (
                np.array([factors[2][str(int(p))] for p in columns["place"][indices]])
                if len(factors) > 2
                else 1
            )
            probability = mortality_probability(probability, fertility * pressure, 0)
        factor = self.factors[year - self.start][0]
        count = np.floor(
            available * np.minimum(1, probability * self.spacing * factor) * valid
            + uniforms(self.seed, indices, ["birth", year])
        ).astype(np.int64)
        return lower, available, np.minimum(count, available)

    def transition(self, indices, year, survival, previous, newborns=None):
        indices = np.asarray(indices, np.int64)
        demos, society = self.policy(year)
        columns = self.columns()
        original = columns["counts"][indices]
        ages = year - columns["birth"][indices]
        race_ids = columns["race"][indices]
        rates = np.zeros_like(survival)
        for race, demo in enumerate(demos):
            mask = race_ids == race
            basic = self._mortality(demo, ages[mask])
            rates[mask, 0] = np.minimum(1, basic * demo.male_mortality_factor)
            rates[mask, 1] = basic
        for event in self.config.events:
            if event.start_year <= year < event.end_year:
                mask = self.event_mask(event, columns, indices, ages)
                for sex in (0, 1):
                    if event.sex == "all" or event.sex == ("male" if sex == 0 else "female"):
                        q = rates[mask, sex]
                        rates[mask, sex] = 1 - (1 - q) ** event.mortality_factor * (
                            1 - event.extra_mortality
                        )
        if self.program:
            factor, extra, _, _, _ = self.program.factors(year, columns["place"][indices])
            rates = mortality_probability(rates, factor[:, None], extra[:, None])
            for race, demo in enumerate(demos):
                rates[(race_ids == race) & (ages >= demo.max_age)] = 1
        # Maternal risk is a cohort hazard, based on the same birth selection.
        if newborns is None:
            _, _, newborns = self.bounds(indices, year, previous)
        maternal = np.array([d.maternal_mortality for d in demos])[race_ids]
        rates[:, 1] = 1 - (1 - rates[:, 1]) * (
            1 - newborns * maternal / np.maximum(1, previous[:, 1])
        )
        survival *= (1 - rates) * self.factors[year - self.start][1]
        rounding = np.column_stack([uniforms(self.seed, indices, ["survival", s]) for s in (0, 1)])
        alive = np.floor(original * survival + rounding).astype(np.int64)
        alive = np.minimum(alive, previous[:, :2])
        paired = previous[:, 2].copy()
        minimum = np.array(
            [r.marriage_min_age or society.marriage_min_age for r in self.config.races]
        )[race_ids]
        maximum = np.array(
            [r.marriage_max_age or society.marriage_max_age for r in self.config.races]
        )[race_ids]
        eligible = (minimum <= ages) & (ages <= maximum)
        unmatched = np.maximum(0, alive.min(axis=1) - paired)
        paired += np.floor(
            unmatched * society.marriage_rate * eligible
            + uniforms(self.seed, indices, ["union", year])
        ).astype(np.int64)
        active = np.maximum(0, np.minimum(paired, alive.min(axis=1)) - previous[:, 3])
        divorced = previous[:, 3] + np.floor(
            active * society.divorce_rate + uniforms(self.seed, indices, ["divorce", year])
        ).astype(np.int64)
        return np.column_stack((alive, paired, divorced))

    def initial(self, indices):
        indices = np.asarray(indices, np.int64)
        columns = self.columns()
        counts = columns["counts"][indices]
        _, society = self.policy(self.start)
        age = self.start - columns["birth"][indices]
        minimum = np.array(
            [r.marriage_min_age or society.marriage_min_age for r in self.config.races]
        )[columns["race"][indices]]
        eligible = age >= minimum
        paired = np.floor(
            counts.min(axis=1) * society.founder_match_participation * eligible
            + uniforms(self.seed, indices, "founder union")
        ).astype(np.int64)
        return np.column_stack((counts, paired, np.zeros(len(indices), np.int64)))

    def initial_program(self, indices, state, survival):
        def apply():
            return self._initial_shock(indices, state, survival)

        return apply_initial_program(self, apply)

    def _initial_shock(self, indices, state, survival):
        columns = self.columns()
        demos, _ = self.policy(self.start)
        ages, races = self.start - columns["birth"][indices], columns["race"][indices]
        base = np.zeros((len(indices), 2))
        for race, demo in enumerate(demos):
            mask = races == race
            q = self._mortality(demo, ages[mask])
            base[mask, 0] = np.minimum(1, q * demo.male_mortality_factor)
            base[mask, 1] = q
        factor, extra, _, _, _ = self.program.factors(self.start, columns["place"][indices])
        rates = mortality_probability(base, factor[:, None], extra[:, None], initial=True)
        survival *= 1 - rates
        rounding = np.column_stack([uniforms(self.seed, indices, ["survival", s]) for s in (0, 1)])
        result = state.copy()
        result[:, :2] = np.minimum(
            state[:, :2],
            np.floor(columns["counts"][indices] * survival + rounding).astype(np.int64),
        )
        return result

    def curve(self, index):
        """Reconstruct a cohort's count curve, never its population of persons."""
        if index in self._curves:
            result = self._curves.pop(index)
            self._curves[index] = result
            return result
        record = self.records[index]
        result = np.zeros((self.config.duration + 1, 4), np.int64)
        survival = np.ones((1, 2))
        if record["birth"] <= self.start:
            state = self.initial([index])
            if self.program:
                state = self._initial_shock(np.array([index]), state, survival)
            result[0] = state[0]
            begin = self.start + 1
        else:
            state = np.array([[*record["counts"], 0, 0]], np.int64)
            begin = record["birth"]
        for year in range(begin, self.end + 1):
            state = self.transition([index], year, survival, state)
            result[year - self.start] = state[0]
        self._curves[index] = result
        # Count curves are a runtime cache, not retained personal states.
        while len(self._curves) > 256:
            self._curves.popitem(last=False)
        return result

    def birth_plan(self, index):
        if index in self._births:
            return self._births[index]
        child = self.records[index]
        year = child["birth"]
        if year <= self.start:
            return []
        demos, _ = self.policy(year)
        demo = demos[child["race"]]
        indices = [
            self._lookup[key]
            for age in range(demo.fertility_min_age, demo.fertility_max_age + 1)
            if (key := (child["place"], child["race"], year - age)) in self._lookup
        ]
        indices.sort()
        previous = np.array(
            [self.curve(i)[year - self.start - 1] for i in indices], np.int64
        ).reshape(-1, 4)
        lower, available, counts = self.bounds(indices, year, previous)
        plan, offset = [], 0
        for parent, lo, size, count in zip(indices, lower, available, counts, strict=True):
            if count:
                plan.append((parent, int(lo), int(size), int(count), offset))
                offset += int(count)
        if offset != sum(child["counts"]):
            raise ValueError("Procedural birth quotas do not match the demographic registry")
        self._births[index] = plan
        while len(self._births) > 256:
            self._births.popitem(last=False)
        return plan

    def slot(self, index, sex, rank):
        record = self.records[index]
        size = record["counts"][sex]
        slot = permute(rank, size, permutation(self.seed, ["slot", index, sex], size))
        return slot + (record["counts"][0] if sex else 0)

    def sex_rank(self, index, slot):
        sizes = self.records[index]["counts"]
        sex = int(slot >= sizes[0])
        local = slot - sizes[0] if sex else slot
        if not 0 <= local < sizes[sex]:
            raise KeyError(slot)
        return sex, inverse(
            local, sizes[sex], permutation(self.seed, ["slot", index, sex], sizes[sex])
        )

    def parents(self, index, slot):
        record = self.records[index]
        total = sum(record["counts"])
        rank = permute(slot, total, permutation(self.seed, ["child", index], total))
        for parent, lower, size, count, offset in self.birth_plan(index):
            if offset <= rank < offset + count:
                pair = lower + self.spacing * permute(
                    rank - offset,
                    size,
                    permutation(self.seed, ["birth selection", parent, record["birth"]], size),
                )
                return (
                    (parent + 1, self.slot(parent, 0, pair)),
                    (parent + 1, self.slot(parent, 1, pair)),
                )
        return None, None

    def children(self, index, pair):
        parent = self.records[index]
        children = []
        epochs = [self.start, self.end] + [p.start_year for p in self.config.periods]
        max_age = max(d.fertility_max_age for y in epochs for d in self.policy(y)[0])
        min_age = min(d.fertility_min_age for y in epochs for d in self.policy(y)[0])
        for year in range(
            max(self.start + 1, parent["birth"] + min_age),
            min(self.end, parent["birth"] + max_age) + 1,
        ):
            child = self._lookup.get((parent["place"], parent["race"], year))
            if child is None:
                continue
            for source, lower, size, count, offset in self.birth_plan(child):
                if source == index and pair >= lower and (pair - lower) % self.spacing == 0:
                    ordinal = (pair - lower) // self.spacing
                    if ordinal < size:
                        rank = inverse(
                            ordinal,
                            size,
                            permutation(self.seed, ["birth selection", index, year], size),
                        )
                        if rank < count:
                            total = sum(self.records[child]["counts"])
                            slot = inverse(
                                offset + rank,
                                total,
                                permutation(self.seed, ["child", child], total),
                            )
                            children.append((child + 1, slot))
        return children
