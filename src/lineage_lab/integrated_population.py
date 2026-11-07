"""Complete integrated S3 population; standalone hybrid histories stay separate."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .cohort_residence import ResidentRegistry
from .historical_program import CoreBudgetExceeded, ProviderError
from .integrated_core import precompute_core
from .population_export import export_tables
from .registry_demography import RegistryDemography
from .registry_mobility import RegistryMobility
from .site_regulation import SiteRegulator


class IntegratedPopulation:
    def __init__(self, config, *, regulation=True):
        self.config = config
        self.regulation_enabled = regulation
        self.censuses, self.coverage, self.cohort_counts = [], [], []
        self.summary = {}
        self.in_marriage_rows = []

    def run(self):
        config = self.config
        self.core, core_size = precompute_core(config)
        r = self.registry = ResidentRegistry(config, "projected")
        demo = self.demography = RegistryDemography(r)
        mobility = self.mobility = RegistryMobility(r, demo)
        regulator = self.regulator = SiteRegulator(config, enabled=self.regulation_enabled)
        demo.seed(core_size)
        zero = np.zeros(len(config.settlements), np.int64)
        core_previous = zero.copy()
        if self.core:
            cr = self.core.registry
            for row in cr.residences:
                if row["valid"]["start_year"] == r.start and row["start_phase"] == 10:
                    core_previous[cr.public_index[row["site_id"]]] += 1
        background_previous = r.counts()
        for year in range(r.start, r.end):
            r.open_year(year)
            opening = mobility.opening()
            multiplier = regulator.begin(year, core_previous, background_previous)
            eligible = zero.copy()
            if year == r.start:
                demo.deaths(initial=True)
                forced = r.activate_pairs(initial=True)
            else:
                mothers, eligible = demo.births(multiplier, opening_core=core_previous)
                demo.deaths(mothers)
                forced = r.activate_pairs()
                r.repair_local()
            r.partition(force_single=forced)
            # These current-year core facts first become visible to background
            # movement in phase 40. The phase-10 controller used only old counts.
            core_counts = self.core.annual[year]["counts"] if self.core else zero
            core_counters = (
                self.core.annual[year]["counters"] if self.core else np.zeros_like(r.counters)
            )
            if self.core:
                self._in_marriage_departures(year)
            mobility.programmed(self.core.operations if self.core else ())
            if year != r.start:
                mobility.ports(opening)
                mobility.ordinary(opening, core_counts)
            background = r.counts()
            counters = r.counters + core_counters
            self._budgets(year, int(core_counts.sum()), int(background.sum()))
            population = core_counts + background
            if not np.array_equal(
                population,
                core_previous
                + background_previous
                + counters[:, 0]
                - counters[:, 1]
                + counters[:, 2]
                - counters[:, 3],
            ):
                raise ProviderError(
                    "InvalidPopulationResult", "Annual registry balance failed", year=year
                )
            regulator.publish(
                year, core_counts, background, counters[:, 2], counters[:, 3], eligible
            )
            for site, site_id in enumerate(r.site_ids):
                self.censuses.append(
                    {
                        "site_id": site_id,
                        "year": year,
                        "population": int(population[site]),
                        "simulated": int(core_counts[site]),
                        "projected": int(background[site]),
                        "births": int(counters[site, 0]),
                        "deaths": int(counters[site, 1]),
                        "incoming": int(counters[site, 2]),
                        "outgoing": int(counters[site, 3]),
                        "precision": "realized",
                        "coverage": "complete",
                    }
                )
            self.coverage.append(
                {
                    "year": year,
                    "core_alive": int(core_counts.sum()),
                    "population": int(population.sum()),
                    "core_historical": self.core.annual[year]["historical"] if self.core else 0,
                    "background_historical": r.n,
                }
            )
            self.cohort_counts.extend(r.cohort_counts())
            if self.core:
                self.cohort_counts.extend(self.core.annual[year]["cohort_counts"])
            core_previous, background_previous = core_counts.copy(), background.copy()
        self.censuses.sort(key=lambda row: (row["site_id"].encode(), row["year"]))
        self.cohort_counts.sort(
            key=lambda row: (
                row["origin_cohort_id"].encode(),
                row["site_id"].encode(),
                row["sex"],
                row["year"],
            )
        )
        self.summary = {
            "core_founders_simulated": core_size,
            "core_coverage": self.coverage,
            "population": int((core_previous + background_previous).sum()),
            "people_ever": r.n + (self.core.registry.n if self.core else 0),
        }
        return self

    def _in_marriage_departures(self, year):
        """Phase 40 counterpart of `chronos-core-in-marriage-1`.

        For each spouse the core drew, one background resident of the same site
        and sex leaves the background, so the person counts once. Preferred:
        unpartnered, same ancestry and of marriage age; then the nearest birth
        year, someone living alone, and the semantic slot ID. No such resident
        rejects the core variant through the existing budget rule.
        """
        requests = sorted(self.core.in_marriages.get(year, ()), key=lambda q: q[4].encode())
        if not requests:
            return
        r, society = self.registry, self.config.society
        d = r.data
        by_site = defaultdict(list)
        for row in map(int, r.living()):
            if year - int(d["birth"][row]) >= society.dependent_age:
                by_site[int(d["site"][row])].append(row)

        def rank(row, race, birth):
            age = year - int(d["birth"][row])
            preferred = (
                row not in r.partner
                and int(d["race"][row]) == race
                and society.marriage_min_age <= age <= society.marriage_max_age
            )
            household = int(d["household"][row])
            shared = household >= 0 and len(r.households[household]["members"]) > 1
            return (not preferred, abs(int(d["birth"][row]) - birth), shared, r.ids[row])

        for site, sex, race, birth, core_id in requests:
            candidates = [p for p in by_site[site] if d["sex"][p] == sex]
            if not candidates:
                raise CoreBudgetExceeded(
                    "No background resident can marry into the core",
                    year=year,
                    constraint="core_in_marriage_background",
                )
            row = min(candidates, key=lambda p, race=race, birth=birth: rank(p, race, birth))
            fallback = rank(row, race, birth)[0]
            by_site[site].remove(row)
            r.leave_to_core(row)
            self.in_marriage_rows.append(
                {
                    "year": year,
                    "site_id": r.site_ids[site],
                    "core_slot": core_id,
                    "background_slot": r.ids[row],
                    "birth_year_gap": abs(int(d["birth"][row]) - birth),
                    "fallback": fallback,
                }
            )

    def _budgets(self, year, core_alive, background_alive):
        limits = self.config.historical.limits
        core_ever = self.core.annual[year]["historical"] if self.core else 0
        total_ever = core_ever + self.registry.n
        if total_ever > limits["max_candidate_slots"]:
            raise ProviderError(
                "BudgetExceeded",
                "Historical slot budget exceeded",
                year=year,
                constraint="max_candidate_slots",
            )
        cohorts = len(self.registry.cohorts)
        if self.core:
            cohorts += sum(c["birth_year"] <= year for c in self.core.registry.cohorts)
        if cohorts > limits["max_cohorts"]:
            raise ProviderError(
                "BudgetExceeded",
                "Origin cohort budget exceeded",
                year=year,
                constraint="max_cohorts",
            )
        if 10000 * core_alive > limits["core_max_share_bps"] * (core_alive + background_alive):
            raise CoreBudgetExceeded(
                "Living core share exceeds ceiling", year=year, constraint="core_max_share_bps"
            )
        if 10000 * core_ever > limits["core_max_share_bps"] * total_ever:
            raise CoreBudgetExceeded(
                "Historical core share exceeds ceiling",
                year=year,
                constraint="core_historical_max_share_bps",
            )

    def tables(self):
        registries = [self.registry] + ([self.core.registry] if self.core else [])
        return export_tables(registries, self.core)

    def diagnostics(self):
        extra = {}
        if self.config.historical.core_in_marriage:
            rows = self.in_marriage_rows
            extra["core_in_marriage"] = {
                "rule_version": self.config.historical.core_in_marriage,
                "spouses": len(rows),
                "fallback_matches": sum(row["fallback"] for row in rows),
                "birth_year_gap_total": sum(row["birth_year_gap"] for row in rows),
                "background_exit": "departed_external at phase 40; core arrival cohort",
            }
        return {
            **extra,
            "core_coverage": self.coverage,
            "regulation": self.regulator.rows,
            "mobility": self.mobility.diagnostics,
            "fertility": self.demography.exposure,
            "projected_pairs": {
                state: sum(p["state"] == state for p in self.registry.pairs)
                for state in ("unformed", "active", "closed")
            },
            "local_repair_unions": sum(
                '"local_repair"' in u["family_key"] for u in self.registry.unions
            ),
            "role_demands": "planning_only; no_office_holder_selected",
            "projection_limits": {
                k: self.config.historical.limits[k]
                for k in ("max_projection_ancestors", "max_projection_descendants")
            },
            "known_parentage": "recorded_within_domain_coverage; projected_links_are_approximations",
            "export_ordering": "comparison_sort; O(L log L + E log E)",
            "simulation_work": "slot scans and radix selection; bootstrap O(L log L); independent core uses Engine matching",
            "core_event_tape": {
                "version": "chronos-core-phase-tape-1",
                "events": len(self.core.tape) if self.core else 0,
                "independent_of_projected_population": True,
            },
        }
