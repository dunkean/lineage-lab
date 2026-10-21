# Hybrid design notes

The implemented hybrid runtime is described in [the model contract](genealogy-hybrid.md). This document records its boundaries and remaining design work.

## Shared interfaces

`archive_protocol.py` defines a read interface for full, compact and hybrid exploration. `engine.py` dispatches from validated `Scenario` settings. `hybrid_engine.py` combines a single detailed core run with `cohorts.py`; `projection.py` supplies versioned deterministic permutations and `hybrid_archive.py` serves bounded on-demand records. The existing studio and server consume this archive interface.

World identity includes configuration and algorithm version. Hybrid IDs are opaque; callers must not cast them to integer person rows. Existing full-mode formats and algorithms remain distinct.

## Why cohort counts are not exhaustive genealogy

Counts alone cannot recover every historically simulated marriage, sibling group, birth interval or kinship exclusion. Independently hashing parents at read time is insufficient. The current background uses constrained, invertible family assignment, with documented simplifications and depth limits. It never claims equivalence to full histories.

The core is simulated with exact family decisions under the full engine's rules. Its annual living and historical budgets are enforced without truncating valid families. Ordinary background slots remain procedural; queries must not secretly run the full engine.

## Verified boundaries

- Exactly one core engine, receiving only the configured core founders.
- No ordinary person simulation while constructing background cohorts.
- Parent/child inversion, stable IDs, reopening, cache eviction, lookup order and concurrent reads.
- Reconciled census counts and atlas totals without materializing the whole population.
- Bounded personal and cohort caches, explicit algorithm versions and archive checksums.
- Explicit source labels and depth limits in both API and studio.

## Future work

Arbitrary retrospective promotion of an ordinary background person into a newly causal simulated life is not implemented. Such promotion would need a consistent policy for already exposed relatives, partners and events.

The core and background currently do not exchange partners. Background individual migration, crossbreeding, social transitions and detailed cousin checks would require new population accounting and family assignment rules rather than a display-only change.

An exhaustive world of tens of millions over a millennium remains a separate full-mode scaling problem. Candidate work includes separating living state from archived history, storing family-event blocks, addressable versioned randomness and carefully validated parallel matching. None of these are inferred from hybrid timing improvements.

Any GPU implementation should be checked against a CPU oracle before performance claims. Approximate background genealogy and exact full genealogy must remain visibly different product contracts.
