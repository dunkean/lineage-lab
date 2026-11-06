"""Programmed household moves, statistical ports and cultural-rate mobility."""

from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
from math import lcm

import numpy as np

from .registry_contract import address, in_span, stream


def capacities(config, year):
    settings = config.historical
    growth = (1 + Fraction(str(config.society.capacity_growth))) ** (year - config.start_year)
    result = []
    for place in config.settlements:
        episode = next(
            e for e in settings.occupation_episodes[place.id] if in_span(year, e["valid"])
        )
        value = (
            Fraction(episode["capacity"]) * growth if episode["state"] == "active" else Fraction(0)
        )
        for shock in settings.shocks:
            if settings.site_ids[place.id] in shock["site_ids"] and in_span(year, shock["valid"]):
                value *= Fraction(shock["effects"].get("capacity_multiplier_ppm", 1000000), 1000000)
        result.append(value)
    return result


def routes(config, year):
    ids = [p.id for p in config.settlements]
    indexes = {p: i for i, p in enumerate(ids)}
    costs = [[None for _ in ids] for _ in ids]
    for i in range(len(ids)):
        costs[i][i] = 0
    for edge in config.historical.mobility_edges:
        if not in_span(year, edge["valid"]):
            continue
        a, b = indexes[edge["from"]], indexes[edge["to"]]
        costs[a][b] = (
            edge["travel_days"] if costs[a][b] is None else min(costs[a][b], edge["travel_days"])
        )
        if edge["bidirectional"]:
            costs[b][a] = (
                edge["travel_days"]
                if costs[b][a] is None
                else min(costs[b][a], edge["travel_days"])
            )
    for k in range(len(ids)):
        for i in range(len(ids)):
            for j in range(len(ids)):
                if costs[i][k] is not None and costs[k][j] is not None:
                    via = costs[i][k] + costs[k][j]
                    if costs[i][j] is None or via < costs[i][j]:
                        costs[i][j] = via
    return costs


class RegistryMobility:
    def __init__(self, registry, demography):
        self.r, self.demography = registry, demography
        self.settings = registry.settings
        self.remainders = defaultdict(Fraction)
        self.diagnostics = []

    def opening(self):
        return np.array([len(units) for units in self.r.site_households], np.int64)

    def _demand(self, key, value):
        exact = self.remainders[key] + value
        demand = exact.numerator // exact.denominator
        self.remainders[key] = exact - demand
        return demand

    def _select(self, site, demand, reason, key):
        choices = self.r.eligible_households(site)
        rng = stream(self.r.config.seed, "mobility", reason, key, self.r.year)
        return [int(hh) for hh in rng.permutation(choices)[:demand]]

    def programmed(self, core_operations=()):
        r = self.r
        core = {
            o["op_id"]: o["actual_core_households"] for o in core_operations if o["year"] == r.year
        }
        for op in sorted(self.settings.historical_program, key=lambda op: op["op_id"]):
            if op["year"] != r.year:
                continue
            demand = max(0, op["target_households"] - core.get(op["op_id"], 0))
            origin = r.public_index[op["origin_site_id"]]
            destination = r.public_index.get(op["destination_site_id"])
            moved = sum(
                r.transfer(
                    hh,
                    destination,
                    reason="programmed",
                    operation_id=op["op_id"],
                    port_id=op["destination_port_id"],
                )
                for hh in self._select(origin, demand, "programmed", op["op_id"])
            )
            self.diagnostics.append(
                {
                    "kind": "operation",
                    "year": r.year,
                    "op_id": op["op_id"],
                    "requested_households": op["target_households"],
                    "actual_core_households": core.get(op["op_id"], 0),
                    "actual_projected_households": moved,
                    "unserved_households": demand - moved,
                }
            )

    def ports(self, opening):
        r = self.r
        for port in sorted(self.settings.boundary_ports, key=lambda p: p["port_id"]):
            if not in_span(r.year, port["valid"]):
                continue
            episode = next(e for e in port["rate_episodes"] if in_span(r.year, e["valid"]))
            site = r.public_index[port["entry_site_id"]]
            base = Fraction(int(opening[site]), 10000 * port["rate_period_years"])
            incoming = self._demand((port["port_id"], "in"), base * episode["incoming_rate_bps"])
            outgoing = self._demand((port["port_id"], "out"), base * episode["outgoing_rate_bps"])
            moved = sum(
                r.transfer(hh, None, reason="statistical_outgoing", port_id=port["port_id"])
                for hh in self._select(site, outgoing, "statistical_outgoing", port["port_id"])
            )
            for ordinal in range(incoming):
                self._incoming(port, site, ordinal)
            self.diagnostics.append(
                {
                    "kind": "statistical_port",
                    "year": r.year,
                    "port_id": port["port_id"],
                    "incoming_households": incoming,
                    "outgoing_households": moved,
                    "unserved_households": outgoing - moved,
                    "named_external_people": False,
                }
            )

    def _incoming(self, port, site, ordinal):
        r, d = self.r, self.r.data
        profile = next(
            p for p in self.settings.external_profiles if p["id"] == port["external_profile_id"]
        )
        rng = stream(r.config.seed, "incoming", port["port_id"], r.year, ordinal)
        sizes = sorted((int(k), v) for k, v in profile["household_size_weights"].items())
        weights = np.array([v for _, v in sizes], float)
        size = sizes[int(rng.choice(len(sizes), p=weights / weights.sum()))][0]
        ancestries = [a.name for a in r.config.races]
        weights = np.array([profile["ancestry_weights"].get(a, 0) for a in ancestries], float)
        race = int(rng.choice(len(ancestries), p=weights / weights.sum()))
        family = address("incoming_family", r.domain, port["port_id"], r.year, ordinal)
        people = []
        groups = defaultdict(list)
        for member in range(size):
            age = (
                int(rng.integers(profile["adult_age_min"], profile["adult_age_max"] + 1))
                if member < 2
                else int(rng.integers(0, r.config.society.dependent_age))
            )
            sex = member if member < 2 else int(rng.integers(0, 2))
            groups[r.year - age].append(sex)
        for birth, sexes in sorted(groups.items()):
            rows = r.add_cohort(
                "arrival",
                site,
                race,
                birth,
                sexes,
                token=[port["port_id"], r.year, ordinal],
                families=[family] * len(sexes),
            )
            self.demography.traits(rows)
            for row in rows:
                r.residences[r.residence_index[int(row)]]["start_phase"] = 40
            people.extend(map(int, rows))
        adults = [p for p in people if r.year - d["birth"][p] >= r.config.society.dependent_age]
        if len(adults) >= 2:
            a, b = adults[:2]
            pair = {
                "key": family,
                "a": a,
                "b": b,
                "state": "active",
                "formed": r.year,
                "last_birth": r.start - 10000,
                "union": len(r.unions),
            }
            r.pairs.append(pair)
            r.partner[a], r.partner[b] = b, a
            r.unions.append(
                {
                    "union_id": address(
                        "incoming_union", r.domain, port["port_id"], r.year, ordinal
                    ),
                    "partner_slots": [a, b],
                    "family_key": family,
                    "valid": {"start_year": r.year, "end_year": r.end},
                    "start_phase": 40,
                    "end_phase": 0,
                    "end_reason": "coverage_end",
                    "site_at_formation": r.site_ids[site],
                    "source": "incoming_formed",
                }
            )
        hh = r.new_household(
            ["incoming", port["port_id"], r.year, ordinal],
            site,
            "union" if len(adults) >= 2 else "single",
            "statistical_arrival",
        )
        for p in people:
            d["household"][p] = hh
            if p in adults:
                d["adult_initialized"][p] = 1
            if p not in adults:
                r.declared_guardians[p] = adults.copy()
                d["guardian"][p] = adults[0] if adults else -1
                for guardian in adults:
                    r.guardians.append(
                        {
                            "child_slot": p,
                            "guardian_slot": guardian,
                            "year": r.year,
                            "phase": 40,
                            "source": "incoming_declared",
                        }
                    )
        r._commit_household(hh, tuple(sorted(people)), site, 40)
        r.moved.add(hh)
        r.counters[site, 2] += len(people)
        r.transfers.append(
            {
                "transfer_id": address(
                    "arrival_transfer", r.domain, port["port_id"], r.year, ordinal
                ),
                "household_id": r.households[hh]["household_id"],
                "origin_slots": sorted(people),
                "from_site": None,
                "to_site": r.site_ids[site],
                "port_id": port["port_id"],
                "year": r.year,
                "phase": 40,
                "reason": "statistical_incoming",
                "operation_id": None,
                "source": "projected",
            }
        )

    def ordinary(self, opening, core_counts):
        r = self.r
        costs, capacity = routes(r.config, r.year), capacities(r.config, r.year)
        for site in range(len(r.places)):
            factor = Fraction(str(r.config.society.migration_rate))
            for shock in self.settings.shocks:
                if r.site_ids[site] in shock["site_ids"] and in_span(r.year, shock["valid"]):
                    factor *= Fraction(
                        shock["effects"].get("migration_multiplier_ppm", 1000000), 1000000
                    )
            demand = self._demand(("ordinary", site), int(opening[site]) * min(1, factor))
            moved = 0
            for hh in self._select(site, demand, "ordinary", r.site_ids[site]):
                population = r.counts() + core_counts
                options, weights = [], []
                for dest, days in enumerate(costs[site]):
                    if dest != site and days is not None:
                        weight = max(Fraction(0), capacity[dest] - int(population[dest])) / (
                            1 + days
                        )
                        if weight:
                            options.append(dest)
                            weights.append(weight)
                if not weights:
                    break
                denominator = lcm(*(w.denominator for w in weights))
                integer_weights = [int(w * denominator) for w in weights]
                total = sum(integer_weights)
                # Integer normalization; a 53-bit draw selects a point in the exact total.
                draw = int(
                    stream(
                        r.config.seed, "destination", r.households[hh]["household_id"], r.year
                    ).integers(0, 2**53)
                )
                point = draw * total // 2**53
                cumulative = 0
                for dest, weight in zip(options, integer_weights, strict=True):
                    cumulative += weight
                    if point < cumulative:
                        moved += r.transfer(hh, dest, reason="ordinary")
                        break
            self.diagnostics.append(
                {
                    "kind": "ordinary",
                    "site_id": r.site_ids[site],
                    "year": r.year,
                    "requested_households": demand,
                    "moved_households": moved,
                    "unserved_households": demand - moved,
                    "fractional_remainder": [
                        self.remainders["ordinary", site].numerator,
                        self.remainders["ordinary", site].denominator,
                    ],
                }
            )
