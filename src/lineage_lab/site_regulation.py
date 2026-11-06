"""Exact residual controller, observing only the preceding publication (§28.4)."""

from fractions import Fraction
from itertools import pairwise

import numpy as np

from .registry_contract import half_up, ratio


def interpolate(markers, year, field):
    if year <= markers[0].year:
        return Fraction(getattr(markers[0], field))
    for left, right in pairwise(markers):
        if year <= right.year:
            return Fraction(
                getattr(left, field) * (right.year - year)
                + getattr(right, field) * (year - left.year),
                right.year - left.year,
            )
    return Fraction(getattr(markers[-1], field))


class SiteRegulator:
    def __init__(self, config, *, enabled=True):
        self.settings = config.historical
        self.enabled = enabled
        self.sites = [self.settings.site_ids[p.id] for p in config.settlements]
        self.markers = {
            s: sorted(
                [m for m in self.settings.baseline_markers if m.site_id == s], key=lambda m: m.year
            )
            for s in self.sites
        }
        self.states = [
            {
                "q": Fraction(1),
                "M": 0,
                "M_ref": 0,
                "zero": self.baseline(s, config.start_year) == 0,
                "freeze_until": None,
            }
            for s in self.sites
        ]
        self.rows, self.pending = [], []

    def baseline(self, site, year):
        return interpolate(self.markers[site], year, "population")

    def begin(self, year, core, background):
        self.pending = []
        multipliers = []
        settings = self.settings.regulation
        for i, site in enumerate(self.sites):
            old = self.states[i]
            freeze = old["freeze_until"]
            for shock in self.settings.shocks:
                if shock["valid"]["start_year"] == year and site in shock["site_ids"]:
                    rebase = shock["valid"]["end_year"] + shock["recovery_years"]
                    freeze = rebase if freeze is None else max(freeze, rebase)
            frozen = freeze is not None
            population = int(core[i] + background[i])
            raw = (
                Fraction(population)
                if old["zero"]
                else (old["q"] * self.baseline(site, year - 1) + old["M"] - old["M_ref"])
            )
            guide = max(Fraction(0), raw)
            error = guide - population
            scale = max(Fraction(1), guide)
            tolerance = interpolate(self.markers[site], year - 1, "tolerance_bps")
            ppm = 1000000
            saturated = False
            if (
                self.enabled
                and not frozen
                and not old["zero"]
                and 10000 * abs(error) > tolerance * scale
            ):
                desired = half_up(1000000 * (1 + error / (settings["response_years"] * scale)))
                ppm = max(
                    settings["min_fertility_multiplier_ppm"],
                    min(settings["max_fertility_multiplier_ppm"], desired),
                )
                saturated = ppm != desired
            self.pending.append(
                {
                    "site_id": site,
                    "year": year,
                    "observation_year": year - 1,
                    "guide_raw": ratio(raw),
                    "guide": ratio(guide),
                    "error": ratio(error),
                    "multiplier_ppm": ppm,
                    "frozen": frozen,
                    "freeze_until": freeze,
                    "saturated": saturated,
                    "core_above_guide": int(core[i]) > guide,
                }
            )
            multipliers.append(ppm)
        return np.array(multipliers, np.int64)

    def publish(self, year, core, background, incoming, outgoing, eligible):
        for i, row in enumerate(self.pending):
            old = self.states[i]
            population = int(core[i] + background[i])
            migration = old["M"] + int(incoming[i] - outgoing[i])
            baseline = self.baseline(self.sites[i], year)
            freeze = row["freeze_until"]
            complete = freeze == year
            new = {**old, "M": migration, "freeze_until": freeze}
            if baseline == 0:
                new.update(zero=True, q=None, M_ref=migration)
            elif complete or (old["zero"] and not row["frozen"]):
                new.update(q=Fraction(population) / baseline, M_ref=migration, zero=False)
            if complete:
                new["freeze_until"] = None
            self.states[i] = new
            self.rows.append(
                {
                    **row,
                    "q": ratio(new["q"]),
                    "M": migration,
                    "M_ref": new["M_ref"],
                    "zero": new["zero"],
                    "freeze_until": new["freeze_until"],
                    "rebased": complete or (old["zero"] and not row["frozen"] and baseline > 0),
                    "eligible_pairs": int(eligible[i]),
                    "no_eligible_pairs": bool(eligible[i] == 0),
                    "population": population,
                }
            )
