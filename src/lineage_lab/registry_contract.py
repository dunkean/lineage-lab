"""Versioned S3 execution contracts. Standalone random streams are untouched."""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction

import numpy as np

from .chronos_plan import canonical_bytes
from .registry_kernels import pcg64_opening

REGISTRY_VERSION = "chronos-resident-registry-1"
MORTALITY_VERSION = "chronos-annual-present-mortality-2"
FERTILITY_VERSION = "chronos-successful-birth-spacing-1"
REGULATOR_VERSION = "chronos-residual-rational-1"
REPAIRING_VERSION = "chronos-local-repairing-1"
PHASES = {"resolve": 10, "birth": 20, "death": 30, "union": 35, "move": 40, "publish": 80}
NO_YEAR = -(2**31)


def address(*parts):
    """Lossless, non-recursive semantic address, independent of table positions."""
    return canonical_bytes(list(parts)).decode("utf-8")


def stream(seed, *parts):
    digest = hashlib.sha256(canonical_bytes(["chronos-registry-rng-1", str(seed), *parts])).digest()
    return np.random.default_rng(int.from_bytes(digest[:16], "little"))


def encoded(text):
    """Canonical bytes of one string, to assemble `stream` keys without re-encoding it."""
    return json.dumps(text, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def key_prefix(seed):
    return b'["chronos-registry-rng-1",' + encoded(str(seed)) + b","


def opening_draws(keys):
    """First double and following uint64 of `stream` for each assembled key.

    `keys` are the canonical bytes `stream` would hash. A batch gives exactly what
    `rng.random()` then `rng.integers(0, 2**64, dtype=np.uint64)` give, without
    building a Generator per key. `test_opening_draws_match_numpy` pins this.
    """
    if not keys:
        return np.empty(0), np.empty(0, np.uint64)
    sha256 = hashlib.sha256
    digests = b"".join([sha256(key).digest() for key in keys])
    return pcg64_opening(np.frombuffer(digests, dtype="<u4").reshape(-1, 8))


def half_up(value):
    value = Fraction(value)
    if value < 0:
        return -half_up(-value)
    return (2 * value.numerator + value.denominator) // (2 * value.denominator)


def ratio(value):
    return None if value is None else [value.numerator, value.denominator]


def in_span(year, span):
    return span["start_year"] <= year < span["end_year"]
