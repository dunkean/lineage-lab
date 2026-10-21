"""Core histories plus count cohorts; procedural pedigrees are calculated on demand."""

from __future__ import annotations

import copy
import hashlib
import json
import sqlite3
import threading
import time
import zlib
from collections import Counter, OrderedDict
from contextlib import closing
from pathlib import Path

import numpy as np

from .cohorts import Cohorts
from .config import Scenario, Settlement
from .memory import MemoryArchive
from .politics import PoliticalTimeline
from .projection import PROJECTION_VERSION, canonical, unpack_arrays


class HybridArchive:
    identity_kind = "opaque"
    population_model = "hybrid"

    def __init__(self, payload, native_core=None):
        if payload.get("version") != PROJECTION_VERSION:
            raise ValueError(
                "Unsupported hybrid projection version; the old replay format is obsolete"
            )
        self._payload = payload if native_core is not None else copy.deepcopy(payload)
        self.config = Scenario.model_validate(payload["config"])
        self.summary = self._payload["summary"]
        self.world_id = self.summary["world_id"]
        expected = hashlib.sha256(
            canonical([PROJECTION_VERSION, self.config.model_dump(mode="json")]).encode()
        ).hexdigest()
        if self.world_id != expected:
            raise ValueError("Hybrid world identity does not match its configuration")
        self.settlements = [Settlement.model_validate(p) for p in payload["settlements"]]
        self.activities = sorted({a for p in self.settlements for a in p.activities})
        self.politics = PoliticalTimeline(self.config, self.settlements)
        self.saved_years = {int(y) for y in payload["maps"]}
        self.has_distributions, self.start_year = True, self.config.start_year
        self.path, self.ephemeral, self.created = Path("memory.hybrid"), True, time.time()
        self._lock, self._cache, self._cache_bytes = threading.RLock(), OrderedDict(), 0
        self._budget = self.config.hybrid.projection_cache_mebibytes * 1024**2
        self.background = Cohorts(
            self.config, self.settlements, payload["records"], payload["factors"]
        )
        self.final_counts = unpack_arrays(payload["final_counts"])["counts"]
        self.core = native_core
        if self.core is None and payload["core_config"] is not None:
            local = Scenario.model_validate(payload["core_config"])
            self.core = MemoryArchive(None, local, local.settlements, self.activities)
            arrays = unpack_arrays(payload["core_data"])
            self.core.data = {
                k.removeprefix("person_"): v for k, v in arrays.items() if k.startswith("person_")
            }
            self.core.n = len(self.core.data["birth"])
            self.core.summary = payload["core_summary"]
            self.core.union_table.data, self.core.move_table.data = (
                arrays["unions"],
                arrays["moves"],
            )
            self.core.union_table.n, self.core.move_table.n = (
                len(arrays["unions"]),
                len(arrays["moves"]),
            )
            ids = np.flatnonzero(self.core.data["death"] == np.iinfo(np.int32).min)
            order = np.lexsort((ids, self.core.data["place"][ids]))
            self.core.resident_ids = ids[order]
            self.core.resident_ranges = {}
            for p in self.settlements:
                positions = np.flatnonzero(self.core.data["place"][self.core.resident_ids] == p.id)
                if len(positions):
                    self.core.resident_ranges[p.id] = (int(positions[0]), int(positions[-1]) + 1)
        self._families = None
        if self.core:
            self._families = np.zeros(self.core.n, np.uint8)
            np.maximum.at(self._families, self.core.data["family"], self.core.data["status"])

    @property
    def cache_bytes(self):
        return self._cache_bytes

    def clear_cache(self):
        with self._lock:
            self._cache.clear()
            self._cache_bytes = 0
            self.background._curves.clear()
            self.background._births.clear()

    def _id(self, segment, person):
        return f"h:{self.world_id}:{segment}:{person}"

    def _parse(self, identity):
        parts = str(identity).split(":")
        if len(parts) != 4 or parts[:2] != ["h", self.world_id]:
            raise ValueError("Use a hybrid person ID belonging to this world")
        try:
            segment, person = int(parts[2]), int(parts[3])
        except ValueError as exc:
            raise ValueError("Invalid hybrid identity") from exc
        if self._id(segment, person) != str(identity):
            raise ValueError("Noncanonical hybrid identity")
        if (
            not 0 <= segment < len(self._payload["plans"])
            or not 0 <= person < self._payload["plans"][segment]["people_ever"]
        ):
            raise KeyError(identity)
        return segment, person

    def _ref(self, reference, segment):
        if reference is None:
            return None
        return (
            self._id(*reference)
            if isinstance(reference, (tuple, list))
            else self._id(segment, reference)
        )

    def _core_info(self, person):
        explicit = self._payload["core"].get(f"0:{person}")
        if explicit:
            return explicit
        raw = self.core._person(person)
        score = int(self._families[raw["family"]])
        role, priority = (
            ("noble_house", 100)
            if score >= 3
            else ("bourgeois_lineage", 80)
            if score >= 2
            else ("local_notable", 70)
            if raw["status"] >= 1
            else ("support", 20)
        )
        rules = {r.role: r for r in self.config.hybrid.promotion_rules}
        if role in rules:
            priority = rules[role].priority
        else:
            role, priority = "support", 20
        year = max(self.start_year, raw["birth"])
        return {"role": role, "priority": priority, "promotion_year": year, "family_score": score}

    def _brief(self, segment, raw):
        result = copy.deepcopy(
            {k: v for k, v in raw.items() if k not in {"children", "unions", "migrations"}}
        )
        identity = raw["id"]
        if isinstance(identity, (tuple, list)):
            segment, identity = identity
        result["id"] = self._id(segment, identity)
        for key in ("father", "mother"):
            result[key] = self._ref(raw.get(key), segment)
        family = raw["family"]
        family_segment, family_slot = (
            family if isinstance(family, (tuple, list)) else (segment, family)
        )
        result["family"] = f"f:{self.world_id}:{family_segment}:{family_slot}"
        info = self._core_info(identity) if segment == 0 else None
        result["source"], result["importance"] = ("core", info) if info else ("projected", None)
        syllables = ("Al", "Bel", "Cor", "Dar", "El", "Fen", "Gal", "Har", "Is", "Lor")
        digest = hashlib.blake2b(
            canonical([self.world_id, "name", segment, identity]).encode(), digest_size=8
        ).digest()
        result["name"] = (
            syllables[digest[0] % len(syllables)]
            + ("a" if raw["sex"] else "on")
            + f" of {raw['birth_place']}"
        )
        return result

    def _ordinary(self, segment, slot):
        index = segment - 1
        record = self.background.records[index]
        sex, rank = self.background.sex_rank(index, slot)
        curve = self.background.curve(index)
        begin = max(self.start_year, record["birth"])
        dead = np.flatnonzero(curve[begin - self.start_year :, sex] <= rank)
        death = int(begin + dead[0]) if len(dead) else None
        father, mother = self.background.parents(index, slot)
        raw = {
            "id": (segment, slot),
            "sex": sex,
            "birth": record["birth"],
            "death": death,
            "father": father,
            "mother": mother,
            "birth_place": record["place"],
            "place": record["place"],
            "death_place": record["place"] if death is not None else None,
            "family": father if father else (segment, slot),
            "status": 0,
            "activity": self._activity(record),
            "race": record["race"],
            "founder": father is None,
            "history_known_from": begin,
            "annotations": {},
        }
        demo, society = self.background.policy(begin)
        own_inf = int(
            record["counts"][sex]
            * (
                demo[record["race"]].female_infertility_rate
                if sex
                else demo[record["race"]].male_infertility_rate
            )
        )
        lower = max(
            int(record["counts"][0] * demo[record["race"]].male_infertility_rate),
            int(record["counts"][1] * demo[record["race"]].female_infertility_rate),
        )
        raw["infertile"] = rank < own_inf
        raw["childfree"] = (
            lower <= rank < lower + int(min(record["counts"]) * society.childfree_rate)
        )
        return raw

    def _activity(self, record):
        place = next(p for p in self.settlements if p.id == record["place"])
        activity = max(sorted(place.activities), key=lambda k: place.activities[k])
        return self.activities.index(activity)

    def _raw(self, segment, person):
        return self.core._person(person) if segment == 0 else self._ordinary(segment, person)

    def _children(self, segment, slot):
        if segment == 0:
            return [(0, int(pid)) for pid in self.core._children(slot)]
        _, pair = self.background.sex_rank(segment - 1, slot)
        return self.background.children(segment - 1, pair)

    def person(self, identity):
        segment, person = self._parse(identity)
        with self._lock:
            if identity in self._cache:
                result, charge = self._cache.pop(identity)
                self._cache[identity] = result, charge
                return copy.deepcopy(result)
            if segment == 0:
                raw = self.core.person(person)
            else:
                raw = self._ordinary(segment, person)
                raw["children"] = [self._raw(*ref) for ref in self._children(segment, person)]
                raw["unions"], raw["migrations"] = [], []
                _sex, rank = self.background.sex_rank(segment - 1, person)
                record = self.background.records[segment - 1]
                if rank < min(record["counts"]):
                    curve = self.background.curve(segment - 1)
                    started = np.flatnonzero(curve[:, 2] > rank)
                    if len(started):
                        start = self.start_year + int(started[0])
                        ended = np.flatnonzero(
                            (curve[:, 3] > rank) | (curve[:, :2].min(axis=1) <= rank)
                        )
                        ended = ended[ended >= started[0]]
                        end = self.start_year + int(ended[0]) if len(ended) else None
                        raw["unions"].append(
                            {
                                "id": rank,
                                "male": (segment, self.background.slot(segment - 1, 0, rank)),
                                "female": (segment, self.background.slot(segment - 1, 1, rank)),
                                "start": start,
                                "end": end,
                                "place": record["place"],
                                "reason": "divorce"
                                if end is not None and curve[end - self.start_year, 3] > rank
                                else "death"
                                if end is not None
                                else None,
                            }
                        )
            result = self._brief(segment, raw)
            result["children"] = [self._brief(segment, row) for row in raw["children"]]
            result["unions"] = []
            for row in raw["unions"]:
                result["unions"].append(
                    {
                        **row,
                        "id": f"u:{self.world_id}:{segment}:{row['id']}",
                        "male": self._ref(row["male"], segment),
                        "female": self._ref(row["female"], segment),
                    }
                )
            result["migrations"] = [
                {**row, "id": f"m:{self.world_id}:{segment}:{row['id']}", "person": identity}
                for row in raw["migrations"]
            ]
            result.setdefault("annotations", {})
            charge = 2 * len(canonical(result).encode())
            while self._cache and self._cache_bytes + charge > self._budget:
                _, (_, removed) = self._cache.popitem(last=False)
                self._cache_bytes -= removed
            if charge <= self._budget:
                self._cache[identity] = result, charge
                self._cache_bytes += charge
            return copy.deepcopy(result)

    def lineage(self, identity, depth=4, direction="ancestors", limit=1000):
        if (
            not 0 <= depth <= 12
            or not 1 <= limit <= 5000
            or direction not in {"ancestors", "descendants"}
        ):
            raise ValueError("Invalid lineage bounds or direction")
        root = self._parse(identity)
        if root[0] != 0:
            depth = min(depth, 2 if direction == "ancestors" else 5)
        seen, frontier, people = {root}, [root], []
        with self._lock:
            for generation in range(1, depth + 1):
                next_frontier = []
                for segment, slot in frontier:
                    raw = self._raw(segment, slot)
                    refs = (
                        [raw["father"], raw["mother"]]
                        if direction == "ancestors"
                        else self._children(segment, slot)
                    )
                    for ref in refs:
                        if ref is None:
                            continue
                        ref = tuple(ref) if isinstance(ref, (tuple, list)) else (segment, ref)
                        if ref in seen:
                            continue
                        if len(people) >= limit:
                            return {"people": people, "truncated": True}
                        seen.add(ref)
                        next_frontier.append(ref)
                        people.append(
                            {**self._brief(ref[0], self._raw(*ref)), "generation": generation}
                        )
                frontier = next_frontier
        return {"people": people, "truncated": False}

    def important_people(self, limit=100, role=None):
        if not 1 <= limit <= 500:
            raise ValueError("limit must be 1..500")
        if not self.core:
            return []
        ids = np.arange(self.core.n)
        scores = self._families[self.core.data["family"]]
        roles = np.full(self.core.n, "support", dtype=object)
        priorities = np.full(self.core.n, 20, np.int32)
        rules = {r.role: r for r in self.config.hybrid.promotion_rules}
        masks = {
            "noble_house": scores >= 3,
            "bourgeois_lineage": (scores >= 2) & (scores < 3),
            "local_notable": (scores < 2) & (self.core.data["status"] >= 1),
        }
        for name, mask in masks.items():
            if name in rules:
                roles[mask], priorities[mask] = name, rules[name].priority
        for info in self._payload["core"].values():
            slot = info["person"]
            roles[slot], priorities[slot] = info["role"], info["priority"]
        if role is not None:
            ids = ids[roles == role]
        order = np.lexsort((ids, -priorities[ids]))[:limit]
        return [self._brief(0, self.core._person(int(pid))) for pid in ids[order]]

    def residence(self, identity, year):
        segment, slot = self._parse(identity)
        if year > self.summary["end_year"]:
            raise ValueError("Year is beyond published history")
        if segment == 0:
            return self.core.residence(slot, year)
        with self._lock:
            raw = self._ordinary(segment, slot)
        return (
            raw["place"]
            if year >= raw["history_known_from"] and (raw["death"] is None or year < raw["death"])
            else None
        )

    def residents(self, settlement, limit=100, after=-1):
        with self._lock:
            return self._residents(settlement, limit, after)

    def _residents(self, settlement, limit, after):
        if not 1 <= limit <= 500:
            raise ValueError("limit must be 1..500")
        first, offset = (0, -1) if str(after) == "-1" else self._parse(after)
        result = []
        if first == 0 and self.core:
            result = [self._brief(0, p) for p in self.core.residents(settlement, limit, offset)]
        if len(result) >= limit:
            return result
        for index, record in enumerate(self.background.records):
            segment = index + 1
            if segment < first or record["place"] != settlement:
                continue
            # Sort integer slots only; construct personal facts for this page alone.
            slots = []
            for sex in (0, 1):
                n = int(self.final_counts[index, sex])
                if n:
                    from .projection import permutation

                    size = record["counts"][sex]
                    a, b = permutation(self.config.seed, ["slot", index, sex], size)
                    values = (a * np.arange(n, dtype=np.int64) + b) % size
                    if sex:
                        values += record["counts"][0]
                    slots.extend(values.tolist())
            for slot in sorted(slots):
                if segment == first and slot <= offset:
                    continue
                result.append(self._brief(segment, self._ordinary(segment, slot)))
                if len(result) >= limit:
                    return result
        return sorted(result, key=lambda p: self._parse(p["id"]))

    def resident_atlas(
        self, settlement, min_age=0, max_age=2000, race=None, sex=None, limit=1200, source=None
    ):
        if (
            not 0 <= min_age <= max_age <= 2000
            or not 1 <= limit <= 2000
            or sex not in (None, 0, 1)
            or race is not None
            and race < 0
            or source not in (None, "core", "projected")
        ):
            raise ValueError("Invalid resident atlas bounds")
        year = self.summary["end_year"]
        groups, total, buckets = Counter(), 0, []
        core_ids = []
        if self.core:
            a, b = self.core.resident_ranges.get(settlement, (0, 0))
            ids = self.core.resident_ids[a:b]
            total += len(ids)
            for pid in ids:
                raw = self.core._person(int(pid))
                age = year - raw["birth"]
                if (
                    source != "projected"
                    and min_age <= age <= max_age
                    and (race is None or raw["race"] == race)
                    and (sex is None or raw["sex"] == sex)
                ):
                    core_ids.append(int(pid))
                    groups[age, raw["race"], raw["sex"]] += 1
            buckets.append((0, 0, len(core_ids)))
        for index, record in enumerate(self.background.records):
            if record["place"] != settlement:
                continue
            counts = self.final_counts[index]
            total += int(counts.sum())
            age = year - record["birth"]
            if (
                source != "core"
                and min_age <= age <= max_age
                and (race is None or record["race"] == race)
            ):
                for s in (0, 1):
                    if sex is None or sex == s:
                        n = int(counts[s])
                        groups[age, record["race"], s] += n
                        buckets.append((index + 1, s, n))
        matched = sum(n for _, _, n in buckets)
        positions = np.linspace(0, matched - 1, min(matched, limit), dtype=np.int64)
        people, prefix = [], 0
        for segment, s, n in buckets:
            selected = positions[(positions >= prefix) & (positions < prefix + n)] - prefix
            prefix += n
            for rank in selected:
                slot = (
                    core_ids[int(rank)]
                    if segment == 0
                    else self.background.slot(segment - 1, s, int(rank))
                )
                record = (
                    self.core._person(slot)
                    if segment == 0
                    else self.background.records[segment - 1]
                )
                people.append(
                    {
                        "id": self._id(segment, slot),
                        "birth": record["birth"],
                        "race": record["race"],
                        "sex": record["sex"] if segment == 0 else s,
                        "source": "core" if segment == 0 else "projected",
                    }
                )
        return {
            "year": year,
            "total": total,
            "matched": matched,
            "sampled": matched > limit,
            "groups": [
                {"age": a, "race": r, "sex": s, "count": n}
                for (a, r, s), n in sorted(groups.items())
                if n
            ],
            "people": people,
            "source": source,
        }

    def settlement_history(self, settlement):
        if settlement not in {p.id for p in self.settlements}:
            raise ValueError("Unknown settlement")
        return [
            {"year": year, "population": dict(self._payload["maps"][str(year)]).get(settlement, 0)}
            for year in sorted(self.saved_years)
        ]

    def map_at(self, year, race=None):
        if year not in self.saved_years:
            raise ValueError("No stored census at this year; choose an available snapshot")
        counts = dict(self._payload["maps"][str(year)])
        if race is not None:
            counts = {p: n for p, r, n in self._payload["racial_maps"][str(year)] if r == race}
        owners, _, _ = self.politics.at(year)
        return [
            {"settlement": p.id, "population": counts.get(p.id, 0), "nation": int(owners[i])}
            for i, p in enumerate(self.settlements)
        ]

    def distributions(self, year):
        if year not in self.saved_years:
            raise ValueError("No stored census at this year")
        return {
            "available": True,
            "groups": {
                kind: [
                    {"kind": kind, "category": i, "population": n}
                    for i, n in enumerate(values)
                    if n
                ]
                for kind, values in self._payload["groups"][str(year)].items()
            },
            "flows": copy.deepcopy(self._payload["flows"][str(year)]),
        }

    def overview(self):
        return copy.deepcopy(
            {
                "summary": self.summary,
                "history": self._payload["history"],
                "settlements": [
                    {**p.model_dump(mode="json"), "metadata": json.dumps(p.metadata)}
                    for p in self.settlements
                ],
                "activities": [{"id": i, "name": a} for i, a in enumerate(self.activities)],
                "races": [
                    {"id": i, "name": r.name, "metadata": json.dumps(r.metadata)}
                    for i, r in enumerate(self.config.races)
                ],
            }
        )

    def connect(self):
        db = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)
        db.row_factory = sqlite3.Row
        return db

    def save(self, path):
        path = Path(path).resolve()
        staging = path.with_suffix(path.suffix + ".partial")
        if path.exists() or staging.exists():
            raise FileExistsError(f"Choose a new output path: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        data = canonical(self._payload).encode("utf-8")
        try:
            with closing(sqlite3.connect(staging)) as db:
                db.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)")
                db.execute("CREATE TABLE hybrid_plan(data BLOB NOT NULL,digest TEXT NOT NULL)")
                db.executemany(
                    "INSERT INTO metadata VALUES (?,?)",
                    [
                        ("population_model", '"hybrid"'),
                        ("config", canonical(self._payload["config"])),
                        ("summary", canonical(self.summary)),
                        ("complete", "true"),
                    ],
                )
                db.execute(
                    "INSERT INTO hybrid_plan VALUES (?,?)",
                    (
                        zlib.compress(data),
                        hashlib.sha256(data).hexdigest(),
                    ),
                )
                db.commit()
                self.summary["disk_bytes_written"] = staging.stat().st_size
                data = canonical(self._payload).encode("utf-8")
                db.execute(
                    "UPDATE hybrid_plan SET data=?,digest=?",
                    (
                        zlib.compress(data),
                        hashlib.sha256(data).hexdigest(),
                    ),
                )
                db.execute(
                    "UPDATE metadata SET value=? WHERE key='summary'", (canonical(self.summary),)
                )
                db.commit()
            # Windows rename never replaces an existing archive.
            staging.rename(path)
        except Exception:
            staging.unlink(missing_ok=True)
            raise
        self.path, self.ephemeral = path, False
        self.summary["disk_bytes_written"] = path.stat().st_size

    @classmethod
    def open(cls, path):
        path = Path(path).resolve()
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
            complete = db.execute("SELECT value FROM metadata WHERE key='complete'").fetchone()
            if not complete or complete[0] != "true":
                raise ValueError("Archive is incomplete")
            row = db.execute("SELECT data,digest FROM hybrid_plan").fetchone()
            if row is None:
                raise ValueError("Missing hybrid plan")
            try:
                data = zlib.decompress(row[0])
            except zlib.error as exc:
                raise ValueError("Hybrid archive compressed plan is corrupt") from exc
            if hashlib.sha256(data).hexdigest() != row[1]:
                raise ValueError("Hybrid archive checksum mismatch")
            payload = json.loads(data)
        result = cls(payload)
        result.path, result.ephemeral = path, False
        result.created = path.stat().st_mtime
        result.summary["disk_bytes_written"] = path.stat().st_size
        return result

    def close(self):
        self.clear_cache()
