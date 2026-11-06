"""Soft historical effects shared by the simulated core and projected cohorts.

This module has no Chronos repository dependency. Event intervals are inclusive
here, as in the standalone engine; the bridge translates half-open plan spans.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np


class ProviderError(Exception):
    def __init__(self, code, message, *, owner=None, year=None, constraint=None, plan_id=None):
        super().__init__(message)
        self.code, self.owner, self.year = code, owner, year
        self.constraint, self.plan_id = constraint, plan_id

    def as_dict(self):
        return {
            "code": self.code,
            "message": str(self),
            "owner": self.owner,
            "year": self.year,
            "constraint": self.constraint,
            "plan_id": self.plan_id,
        }


class CoreBudgetExceeded(ProviderError):
    def __init__(self, message, **context):
        super().__init__("CoreBudgetExceeded", message, **context)


def apply_initial_program(owner, apply):
    """Run only the bootstrap shock delta, once per newly constructed component."""
    if getattr(owner, "_initial_program_applied", False):
        raise RuntimeError("Initial historical program has already been applied")
    owner._initial_program_applied = True
    return apply()


def mortality_probability(base, factor, extra, *, initial=False):
    # Preserve the provider's event semantics: factors scale cumulative hazard.
    # At bootstrap founders are pre-shock survivors: only excess hazard is new.
    if initial:
        factor = np.maximum(0, factor - 1)
    return 1 - np.power(1 - np.clip(base, 0, 1), factor) * (1 - extra)


class HistoricalProgram:
    def __init__(self, config):
        self.config = config
        self.settings = config.historical
        self.events = sorted(self.settings.events, key=lambda e: e.name.encode("utf-8"))

    def factors(self, year, places):
        places = np.asarray(places)
        mortality, fertility, migration, capacity = [np.ones(places.shape) for _ in range(4)]
        survival = np.ones(places.shape)
        for event in self.events:
            if event.start_year <= year <= event.end_year:
                mask = np.isin(places, event.settlements)
                mortality[mask] *= event.mortality_factor
                fertility[mask] *= event.fertility_factor
                migration[mask] *= event.migration_factor
                capacity[mask] *= event.capacity_factor
                survival[mask] *= 1 - event.extra_mortality
        for place, episodes in self.settings.occupation_episodes.items():
            for episode in episodes:
                span = episode["valid"]
                if span["start_year"] <= year < span["end_year"]:
                    capacity[places == place] *= episode["capacity"] / episodes[0]["capacity"]
                    break
        return mortality, 1 - survival, fertility, migration, capacity

    def baseline(self, site_id, year):
        """Linear, shock-independent baseline, represented as an integer ratio."""
        rows = sorted(
            (m for m in self.settings.baseline_markers if m.site_id == site_id),
            key=lambda m: m.year,
        )
        if not rows:
            raise ValueError("Missing baseline markers")
        if year <= rows[0].year:
            return rows[0].population, 1
        for left, right in pairwise(rows):
            if year <= right.year:
                span = right.year - left.year
                return (
                    left.population * (right.year - year) + right.population * (year - left.year),
                    span,
                )
        return rows[-1].population, 1

    def neighbors(self, year, settlements, society):
        """Shortest travel costs, with no coordinate fallback for isolated sites."""
        ids = [p.id for p in settlements]
        costs = {(i, i): 0 for i in ids}
        for edge in self.settings.mobility_edges:
            span = edge["valid"]
            if span["start_year"] <= year < span["end_year"]:
                a, b, days = edge["from"], edge["to"], edge["travel_days"]
                costs[a, b] = min(costs.get((a, b), float("inf")), days)
                if edge["bidirectional"]:
                    costs[b, a] = min(costs.get((b, a), float("inf")), days)
        for k in ids:
            for i in ids:
                for j in ids:
                    via = costs.get((i, k), float("inf")) + costs.get((k, j), float("inf"))
                    if via < costs.get((i, j), float("inf")):
                        costs[i, j] = via
        radius = max(society.marriage_radius, society.migration_radius)
        return {
            i: sorted(
                (days * 25, j)
                for (origin, j), days in costs.items()
                if origin == i and j != i and days * 25 <= radius
            )[: society.max_neighbors]
            for i in ids
        }
