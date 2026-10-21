"""Canonical identities and compressed numeric state; no background replay."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import zipfile

import numpy as np

PROJECTION_VERSION = "cohort-runtime-v2"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def derive_seed(seed, channel):
    payload = canonical([PROJECTION_VERSION, channel, seed]).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def pack_arrays(arrays):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for name, array in arrays.items():
            with archive.open(name + ".npy", "w", force_zip64=True) as entry:
                np.lib.format.write_array(entry, np.asarray(array), allow_pickle=False)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def unpack_arrays(data):
    with np.load(io.BytesIO(base64.b64decode(data)), allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def permutation(seed, channel, size):
    if size <= 1:
        return 1, 0
    key = derive_seed(seed, channel)
    factor = 1 + key % (size - 1)
    while math.gcd(factor, size) != 1:
        factor = 1 + factor % (size - 1)
    return factor, (key >> 32) % size


def permute(rank, size, parameters):
    if not size:
        raise ValueError("Empty procedural domain")
    a, b = parameters
    return (a * rank + b) % size


def inverse(slot, size, parameters):
    if size == 1:
        return 0
    a, b = parameters
    return ((slot - b) * pow(a, -1, size)) % size


def uniforms(seed, ids, channel):
    values = np.asarray(ids, dtype=np.uint64) ^ np.uint64(derive_seed(seed, channel))
    values ^= values >> np.uint64(30)
    values *= np.uint64(0xBF58476D1CE4E5B9)
    values ^= values >> np.uint64(27)
    values *= np.uint64(0x94D049BB133111EB)
    values ^= values >> np.uint64(31)
    return (values >> np.uint64(11)).astype(np.float64) / 2**53
