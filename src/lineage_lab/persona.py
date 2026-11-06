"""Descriptive personas derived from recorded parentage; nothing here affects a run.

Each trait is one standardized value per person. A child regresses towards the
midparent value by the trait's heritability, plus individual chance. Values come
from dedicated hash streams, never from engine random state, and are recomputed
on demand instead of being archived. Parameters are worldbuilding assumptions,
not a calibrated genetic model.

Projected hybrid people have no enumerable ancestry. Their values use bounded
hashes of the person, parents and grandparents, which keeps parent and child
consistent at a fixed lookup cost but only approximates inheritance.
"""

from __future__ import annotations

import math
import threading
from bisect import bisect_right
from typing import NamedTuple

import numpy as np
from numba import njit

from .projection import derive_seed

PERSONA_MODEL = "persona-v1"


class Trait(NamedTuple):
    name: str
    group: str
    heritability: float


TRAITS = (
    Trait("height", "physical", 0.8),
    Trait("build", "physical", 0.5),
    Trait("skin_pigment", "physical", 0.85),
    Trait("hair_pigment", "physical", 0.85),
    Trait("hair_red", "physical", 0.8),
    Trait("eye_pigment", "physical", 0.85),
    Trait("hair_curl", "physical", 0.8),
    Trait("hair_loss", "physical", 0.7),
    Trait("nose", "physical", 0.65),
    Trait("mouth", "physical", 0.65),
    Trait("jaw", "physical", 0.65),
    Trait("intelligence", "cognitive", 0.6),
    Trait("memory", "cognitive", 0.5),
    Trait("verbal", "cognitive", 0.5),
    Trait("spatial", "cognitive", 0.5),
    Trait("openness", "temperament", 0.45),
    Trait("conscientiousness", "temperament", 0.45),
    Trait("extraversion", "temperament", 0.45),
    Trait("agreeableness", "temperament", 0.45),
    Trait("neuroticism", "temperament", 0.45),
    Trait("robustness", "health", 0.35),
)
TRAIT_INDEX = {trait.name: index for index, trait in enumerate(TRAITS)}

ADULT_HEIGHT_CM = ((172.0, 7.0), (160.0, 6.5))  # male, female
NEWBORN_HEIGHT_SHARE = 0.45
STATUS_HEIGHT_CM = 1.0
MAX_STATUS_HEIGHT_CM = 3.0
GREY_HAIR_LIFESPAN_SHARE = 0.5
RED_HAIR_THRESHOLD = 1.75
HAIR_CUTS, HAIR_LABELS = (-0.85, 0.0, 1.0), ("blond", "light brown", "brown", "black")
# Shared pigmentation: fair hair tends to come with light eyes and skin.
EYE_HAIR_SHARE, SKIN_HAIR_SHARE = 0.6, 0.5
SCORE_CUTS = (-1.3, -0.5, 0.5, 1.3)
SCORE_LABELS = ("very low", "low", "average", "high", "very high")
ADJECTIVE_THRESHOLD = 0.8
MAX_ADJECTIVES = 5
# Bipolar readings: weighted standardized traits, then (high word, low word).
ADJECTIVES = (
    ({"openness": 1}, "curious", "incurious"),
    ({"extraversion": 0.7, "agreeableness": 0.5, "neuroticism": -0.5}, "jovial", "sombre"),
    ({"neuroticism": -0.8, "extraversion": 0.6}, "brave", "cowardly"),
    ({"conscientiousness": 1}, "diligent", "careless"),
    ({"agreeableness": 1}, "kind", "harsh"),
    ({"neuroticism": -1}, "calm", "anxious"),
    ({"extraversion": 1}, "sociable", "reserved"),
    ({"openness": 0.6, "intelligence": 0.4}, "inventive", "conventional"),
)
LABELS = {
    "height": ((-1.3, -0.5, 0.5, 1.3), ("very short", "short", "average", "tall", "very tall")),
    "build": ((-1.3, -0.5, 0.5, 1.3), ("thin", "slender", "average", "stocky", "heavy")),
    "skin_tone": (
        (-1.2, -0.4, 0.4, 1.2),
        ("pale", "fair", "olive", "brown", "dark"),
    ),
    "eye_color": ((-0.8, -0.2, 0.7), ("blue", "green", "hazel", "brown")),
    "hair_texture": ((0.2, 0.9, 1.6), ("straight", "wavy", "curly", "coily")),
    "nose": ((-1.0, 1.0), ("narrow", "medium", "broad")),
    "mouth": ((-1.0, 1.0), ("thin", "medium", "full")),
    "jaw": ((-1.0, 1.0), ("soft", "medium", "square")),
}


def bucket(value, kind):
    cuts, labels = LABELS[kind]
    return labels[bisect_right(cuts, value)]


def validate_race_persona(race_name, settings):
    """Reject scenario mistakes early; offsets are in standard deviations."""
    if settings is None:
        return
    if not isinstance(settings, dict):
        raise ValueError(f"Race {race_name}: metadata.persona must be a mapping")
    for trait, values in settings.items():
        if trait not in TRAIT_INDEX:
            raise ValueError(f"Unknown persona trait {trait!r} for race {race_name}")
        if not isinstance(values, dict) or set(values) - {"mean", "sd"}:
            raise ValueError(f"Race {race_name}: persona {trait} accepts only mean and sd")
        if "sd" in values and not values["sd"] > 0:
            raise ValueError(f"Race {race_name}: persona {trait} sd must be positive")


def hair_colors(values):
    """Natural hair color for a (people, traits) array."""
    pigment = np.asarray(values[:, TRAIT_INDEX["hair_pigment"]])
    labels = np.array(HAIR_LABELS, dtype=object)[np.searchsorted(HAIR_CUTS, pigment, side="right")]
    labels[np.asarray(values[:, TRAIT_INDEX["hair_red"]]) > RED_HAIR_THRESHOLD] = "red"
    return labels


@njit(cache=True, nogil=True, inline="always")
def _mix(value):
    value ^= value >> np.uint64(30)
    value *= np.uint64(0xBF58476D1CE4E5B9)
    value ^= value >> np.uint64(27)
    value *= np.uint64(0x94D049BB133111EB)
    value ^= value >> np.uint64(31)
    return value


@njit(cache=True, nogil=True, inline="always")
def _normal(key):
    first = (_mix(key) >> np.uint64(11)) / 2.0**53 + 2.0**-54
    second = (_mix(key + np.uint64(1)) >> np.uint64(11)) / 2.0**53
    return math.sqrt(-2 * math.log(first)) * math.cos(2 * math.pi * second)


@njit(cache=True, nogil=True)
def _inherited_values(father, mother, race, means, spreads, heritability, seed):
    # Parents always precede children, so one ascending pass resolves ancestry.
    people, traits = len(father), len(heritability)
    values = np.empty((people, traits), np.float32)
    golden = np.uint64(0x9E3779B97F4A7C15)
    for person in range(people):
        dad, mum, group = father[person], mother[person], race[person]
        for trait in range(traits):
            key = seed ^ (np.uint64(person * traits + trait) * golden)
            mean, spread = means[group, trait], spreads[group, trait]
            if dad < 0 or mum < 0:
                values[person, trait] = mean + spread * _normal(key)
            else:
                share = heritability[trait]
                middle = 0.5 * (values[dad, trait] + values[mum, trait])
                chance = spread * math.sqrt(1 - 0.5 * share * share) * _normal(key)
                values[person, trait] = mean + share * (middle - mean) + chance
    return values


class PersonaModel:
    def __init__(self, config):
        self.config = config
        self.seed = derive_seed(config.seed, "persona")
        self.heritability = np.array([trait.heritability for trait in TRAITS])
        self.means = np.zeros((len(config.races), len(TRAITS)))
        self.spreads = np.ones((len(config.races), len(TRAITS)))
        for row, race in enumerate(config.races):
            for trait, values in (race.metadata.get("persona") or {}).items():
                self.means[row, TRAIT_INDEX[trait]] = values.get("mean", 0)
                self.spreads[row, TRAIT_INDEX[trait]] = values.get("sd", 1)

    def values(self, father, mother, race):
        return _inherited_values(
            np.asarray(father, np.int32),
            np.asarray(mother, np.int32),
            np.asarray(race, np.int16),
            self.means,
            self.spreads,
            self.heritability,
            np.uint64(self.seed),
        )

    def _draw(self, kind, node):
        # Fixed-length entropy: SeedSequence must not see trailing-zero aliases.
        return np.random.default_rng([self.seed, kind, *node]).standard_normal(len(TRAITS))

    @staticmethod
    def _parent_nodes(node, parents):
        segment, slot, first, second = node
        if first == 0:
            father, mother = parents((segment, slot))
            if father is not None:
                return (*father, 0, 0), (*mother, 0, 0)
            return (segment, slot, 1, 0), (segment, slot, 2, 0)
        return (segment, slot, first, 1), (segment, slot, first, 2)

    def projected_values(self, reference, parents, race):
        """Bounded inheritance: weights keep unit variance and midparent slope h."""
        node = (*reference, 0, 0)
        lineage = 0.5 * self._draw(0, node)
        for parent in self._parent_nodes(node, parents):
            lineage += 0.5 * self._draw(0, parent)
            for grandparent in self._parent_nodes(parent, parents):
                lineage += 0.25 * self._draw(0, grandparent)
        standard = (
            np.sqrt(self.heritability) * lineage
            + np.sqrt(1 - self.heritability) * self._draw(1, node)
        )
        return self.means[race] + self.spreads[race] * standard

    def _race_ages(self, race):
        profile = self.config.races[race]
        adult = profile.dependent_age or self.config.society.dependent_age
        lifespan = profile.demography.max_age or self.config.demography.max_age
        return adult, lifespan

    def describe(self, values, sex, race, birth, year, status, source):
        def value(name):
            return float(values[TRAIT_INDEX[name]])

        age = max(0, year - birth)
        adult, lifespan = self._race_ages(race)
        physical = self._physical(value, sex, age, adult, lifespan, status)
        scores = {
            group: {
                trait.name: _score(value(trait.name)) for trait in TRAITS if trait.group == group
            }
            for group in ("cognitive", "temperament", "health")
        }
        readings = _readings(value)
        return {
            "model": PERSONA_MODEL,
            "source": source,
            "reference_year": int(year),
            "age": int(age),
            "physical": physical,
            **scores,
            "readings": readings,
            "summary": _summary(physical, readings, sex, age, adult),
        }

    @staticmethod
    def _physical(value, sex, age, adult, lifespan, status):
        mean, spread = ADULT_HEIGHT_CM[sex]
        grown = NEWBORN_HEIGHT_SHARE + (1 - NEWBORN_HEIGHT_SHARE) * min(1, age / adult)
        nutrition = min(MAX_STATUS_HEIGHT_CM, STATUS_HEIGHT_CM * status)
        height = (mean + spread * value("height") + nutrition) * grown
        body_mass_index = min(40.0, max(15.0, 22 + 3 * value("build")))
        natural = str(hair_colors(np.array([[value(trait.name) for trait in TRAITS]]))[0])
        loss = value("hair_loss") + (age - adult) / 25
        hair_pigment = value("hair_pigment")
        eye = EYE_HAIR_SHARE * hair_pigment + math.sqrt(1 - EYE_HAIR_SHARE**2) * value(
            "eye_pigment"
        )
        skin = SKIN_HAIR_SHARE * hair_pigment + math.sqrt(1 - SKIN_HAIR_SHARE**2) * value(
            "skin_pigment"
        )
        return {
            "height_cm": round(height),
            "height": bucket(value("height"), "height"),
            "weight_kg": round(body_mass_index * (height / 100) ** 2),
            "build": bucket(value("build"), "build"),
            "skin_tone": bucket(skin, "skin_tone"),
            "natural_hair_color": natural,
            "hair_color": "grey" if age >= GREY_HAIR_LIFESPAN_SHARE * lifespan else natural,
            "hair_texture": bucket(value("hair_curl"), "hair_texture"),
            "hair_loss": (
                "bald"
                if sex == 0 and loss > 2.2
                else "thinning"
                if sex == 0 and loss > 1.4
                else "none"
            ),
            "eye_color": bucket(eye, "eye_color"),
            "nose": bucket(value("nose"), "nose"),
            "mouth": bucket(value("mouth"), "mouth"),
            "jaw": bucket(value("jaw"), "jaw"),
        }


def _score(value):
    return {
        "percentile": round(50 * (1 + math.erf(value / math.sqrt(2)))),
        "label": SCORE_LABELS[bisect_right(SCORE_CUTS, value)],
    }


def _readings(value):
    extraversion, neuroticism = value("extraversion"), value("neuroticism")
    type16 = "".join(
        high if value(name) >= 0 else low
        for name, high, low in (
            ("extraversion", "E", "I"),
            ("openness", "N", "S"),
            ("agreeableness", "F", "T"),
            ("conscientiousness", "J", "P"),
        )
    )
    if extraversion >= 0:
        temperament = "choleric" if neuroticism >= 0 else "sanguine"
    else:
        temperament = "melancholic" if neuroticism >= 0 else "phlegmatic"
    ranked = []
    for weights, high, low in ADJECTIVES:
        norm = math.sqrt(sum(weight**2 for weight in weights.values()))
        score = sum(weight * value(name) for name, weight in weights.items()) / norm
        if abs(score) >= ADJECTIVE_THRESHOLD:
            ranked.append((abs(score), high if score > 0 else low))
    ranked.sort(key=lambda item: -item[0])
    return {
        "type16": type16,
        "classical_temperament": temperament,
        "adjectives": [word for _, word in ranked[:MAX_ADJECTIVES]],
    }


def _summary(physical, readings, sex, age, adult):
    noun = (("boy", "girl") if age < adult else ("man", "woman"))[sex]
    words = [word for word in (physical["height"], physical["build"]) if word != "average"]
    described = ", ".join(words) or "ordinary"
    article = "An" if described[0] in "aeiou" else "A"
    text = (
        f"{article} {described} {noun} with {physical['hair_color']} hair "
        f"and {physical['eye_color']} eyes"
    )
    adjectives = readings["adjectives"][:3]
    if adjectives:
        text += ", " + (
            adjectives[0]
            if len(adjectives) == 1
            else ", ".join(adjectives[:-1]) + " and " + adjectives[-1]
        )
    return text + "."


class PersonaReader:
    """Lazily derives every recorded person once; the array is a runtime cache."""

    def __init__(self, config, columns):
        self.model = PersonaModel(config)
        self._columns = columns
        self._values = None
        self._lock = threading.Lock()

    def describe(self, person, end_year):
        with self._lock:
            if self._values is None:
                self._values = self.model.values(*self._columns())
        return self.model.describe(
            self._values[person["id"]],
            person["sex"],
            person["race"],
            person["birth"],
            end_year if person["death"] is None else person["death"],
            person["status"],
            "founder" if person["father"] is None or person["mother"] is None else "inherited",
        )
