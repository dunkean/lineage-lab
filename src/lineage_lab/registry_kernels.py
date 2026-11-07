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


_M32 = np.uint64(0xFFFFFFFF)


@njit(cache=True, inline="always")
def _hashmix(value, const):
    value = (value ^ const) & _M32
    const = (const * np.uint64(0x931E8875)) & _M32
    value = (value * const) & _M32
    return value ^ (value >> np.uint64(16)), const


@njit(cache=True, inline="always")
def _mul_128(a_hi, a_lo, b_hi, b_lo):
    """Low 128 bits of a product of two (high, low) uint64 pairs."""
    x0, x1 = a_lo & _M32, a_lo >> np.uint64(32)
    y0, y1 = b_lo & _M32, b_lo >> np.uint64(32)
    low = x0 * y0
    middle = x1 * y0 + (low >> np.uint64(32))
    middle2 = x0 * y1 + (middle & _M32)
    high = x1 * y1 + (middle >> np.uint64(32)) + (middle2 >> np.uint64(32))
    return high + a_hi * b_lo + a_lo * b_hi, (middle2 << np.uint64(32)) | (low & _M32)


@njit(cache=True, inline="always")
def _add_128(a_hi, a_lo, b_hi, b_lo):
    low = a_lo + b_lo
    carry = np.uint64(1) if low < a_lo else np.uint64(0)
    return a_hi + b_hi + carry, low


@njit(cache=True, inline="always")
def _pcg_output(hi, lo):
    rotation = hi >> np.uint64(58)
    value = hi ^ lo
    left = (np.uint64(64) - rotation) & np.uint64(63)
    return (value >> rotation) | (value << left)


@njit(cache=True)
def pcg64_opening(words):
    """First two outputs of `default_rng` for rows of 128-bit little-endian entropy.

    SeedSequence (pool of four words) then PCG64 XSL-RR, as NumPy computes them.
    Column 0 to 3 of `words` hold the entropy; other columns are ignored.
    """
    n = words.shape[0]
    doubles = np.empty(n, np.float64)
    salts = np.empty(n, np.uint64)
    mult_hi = np.uint64(0x2360ED051FC65DA4)
    mult_lo = np.uint64(0x4385DF649FCCF645)
    pool = np.empty(4, np.uint64)
    state = np.empty(4, np.uint64)
    for row in range(n):
        const = np.uint64(0x43B0D7E5)
        for i in range(4):
            pool[i], const = _hashmix(np.uint64(words[row, i]), const)
        for source in range(4):
            for target in range(4):
                if source != target:
                    hashed, const = _hashmix(pool[source], const)
                    mixed = (
                        np.uint64(0xCA01F9DD) * pool[target] - np.uint64(0x4973F715) * hashed
                    ) & _M32
                    pool[target] = mixed ^ (mixed >> np.uint64(16))
        const = np.uint64(0x8B51F9DD)
        for i in range(8):
            value = (pool[i & 3] ^ const) & _M32
            const = (const * np.uint64(0x58F38DED)) & _M32
            value = (value * const) & _M32
            value ^= value >> np.uint64(16)
            if i & 1:
                state[i >> 1] |= value << np.uint64(32)
            else:
                state[i >> 1] = value
        inc_hi = (state[2] << np.uint64(1)) | (state[3] >> np.uint64(63))
        inc_lo = (state[3] << np.uint64(1)) | np.uint64(1)
        hi, lo = inc_hi, inc_lo  # Step from a zero state.
        hi, lo = _add_128(hi, lo, state[0], state[1])
        hi, lo = _mul_128(hi, lo, mult_hi, mult_lo)
        hi, lo = _add_128(hi, lo, inc_hi, inc_lo)
        hi, lo = _mul_128(hi, lo, mult_hi, mult_lo)
        hi, lo = _add_128(hi, lo, inc_hi, inc_lo)
        doubles[row] = (_pcg_output(hi, lo) >> np.uint64(11)) * (1.0 / 9007199254740992.0)
        hi, lo = _mul_128(hi, lo, mult_hi, mult_lo)
        hi, lo = _add_128(hi, lo, inc_hi, inc_lo)
        salts[row] = _pcg_output(hi, lo)
    return doubles, salts


@njit(cache=True)
def count_living_sites(rows, sites, size):
    result = np.zeros(size, np.int64)
    for row in rows:
        if sites[row] >= 0:
            result[sites[row]] += 1
    return result
