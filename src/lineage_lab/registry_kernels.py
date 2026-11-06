"""Serial, fixed-width linear scans for the bounded integrated registry."""

import numpy as np
from numba import njit


@njit(cache=True)
def radix_order(keys):
    order = np.arange(len(keys), dtype=np.int32)
    target = np.empty(len(keys), np.int32)
    for shift in range(0, 64, 8):
        counts = np.zeros(256, np.int32)
        for row in order:
            counts[int((keys[row] >> np.uint64(shift)) & np.uint64(255))] += 1
        offsets = np.zeros(256, np.int32)
        for i in range(1, 256):
            offsets[i] = offsets[i - 1] + counts[i - 1]
        for row in order:
            bucket = int((keys[row] >> np.uint64(shift)) & np.uint64(255))
            target[offsets[bucket]] = row
            offsets[bucket] += 1
        order, target = target, order
    return order


@njit(cache=True)
def annual_ranks(ordinals, salt):
    """A bijection of uint64, freshly keyed annually, over immutable ordinals."""
    result = np.empty(len(ordinals), np.uint64)
    for i in range(len(ordinals)):
        value = np.uint64(ordinals[i]) + salt + np.uint64(0x9E3779B97F4A7C15)
        value = (value ^ (value >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        value = (value ^ (value >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        result[i] = value ^ (value >> np.uint64(31))
    return result


@njit(cache=True)
def count_living_sites(rows, sites, size):
    result = np.zeros(size, np.int64)
    for row in rows:
        if sites[row] >= 0:
            result[sites[row]] += 1
    return result
