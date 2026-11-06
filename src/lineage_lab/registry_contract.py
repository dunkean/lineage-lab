"""Versioned S3 execution contracts. Standalone random streams are untouched."""

from __future__ import annotations

import hashlib
from fractions import Fraction

import numpy as np

from .chronos_plan import canonical_bytes

REGISTRY_VERSION = "chronos-resident-registry-1"
MORTALITY_VERSION = "chronos-annual-present-mortality-2"
FERTILITY_VERSION = "chronos-successful-birth-spacing-1"
REGULATOR_VERSION = "chronos-residual-rational-1"
PHASES = {"resolve": 10, "birth": 20, "death": 30, "union": 35, "move": 40, "publish": 80}
NO_YEAR = -(2**31)


def address(*parts):
    """Lossless, non-recursive semantic address, independent of table positions."""
    return canonical_bytes(list(parts)).decode("utf-8")


def stream(seed, *parts):
    digest = hashlib.sha256(canonical_bytes(["chronos-registry-rng-1", str(seed), *parts])).digest()
    return np.random.default_rng(int.from_bytes(digest[:16], "little"))


def half_up(value):
    value = Fraction(value)
    if value < 0:
        return -half_up(-value)
    return (2 * value.numerator + value.denominator) // (2 * value.denominator)


def ratio(value):
    return None if value is None else [value.numerator, value.denominator]


def in_span(year, span):
    return span["start_year"] <= year < span["end_year"]
