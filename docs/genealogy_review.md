# Historical measurements and review notes

These measurements came from the source project's September/October 2026 work. They were not rerun as large benchmarks during standalone extraction or English localization. Current release checks are in [validation.md](validation.md).

## Full-mode architecture

The engine uses dense IDs, contiguous NumPy columns, Numba matching kernels, bounded partner searches and reusable ancestry marks. Kinship checking is never disabled at large population sizes. Household movements, dated unions, deaths, births, politics and crises are recorded rather than invented during reading.

The memory backend avoids SQLite during simulation and retains binary histories for exploration. Explicit persistence writes SQLite; compact export uses packed columns and indexed family access. The common case was measured at 21 bytes per person, 15 per union and 11 per migration for compact data, excluding broader application overhead.

## Earlier full-mode samples

Windows 11, i7-10700K, 64 GB RAM, Python 3.12.13, NumPy 2.5.3 and Numba 0.67.0. No GPU was used. Topologies and versions differ; rows are not controlled head-to-head comparisons.

| Founders / settlements / years | Final living | Historical people | Simulation + SQLite | Peak RSS |
| --- | ---: | ---: | ---: | ---: |
| 1 M / 500 / 25, dispersed v3 | 1,000,018 | 1,722,266 | 44.0 s | 321.2 MiB |
| 100 k / 500 / 1,000, dispersed v3 | 348,040 | 5,817,116 | 336.0 s | 542.2 MiB |
| 2 M / 500 / 25, grid v2 | 1,977,802 | 3,435,980 | 84.8 s | 523.4 MiB |

Later memory-only runs reported 10.16 s for 1 M founders / 25 years and 71.62 s for 100 k founders / 1,000 years, with the same historical/final populations in those scenarios. First indexed person access added 0.31 s and 1.48 s respectively. These figures do not establish 50 M residents over 1,000 years.

The million-founder SQLite example occupied 156,319,744 bytes. Its compact export occupied 51,974,083 bytes for 1,722,266 people, 535,174 unions and 395,778 movements. This was a storage result, not an automatic reduction in earlier SQLite write cost.

## Audited invariants

Historical audits checked parent ordering and eligibility, birth dates and active unions, non-overlapping union histories, configured-depth kinship exclusions, residence chains and exact settlement census/flow balances. Founders' earlier ancestry remains unknown.

Corrections included founder age distributions, dependent household movement, period-specific age validation, stable capital assignment, closed-border marriage moves, compact ancestor traversal and crises shown at their actual dates between census snapshots.

Later work replaced calibration pilots with one-run population regulation. Biological/social reproductive eligibility is applied before fertility recovery; harmful crises suspend recovery; natural deaths are preserved when additional ceiling deaths are required. Lifelong infertility and childfree states are independent and versioned.

## Interpretation

Demographic inputs are worldbuilding choices, not historical calibration. The full engine still retains dead people in memory. Hybrid cohort timing has a different contract and cannot be cited as proof of an exhaustive full-mode world at the same scale.
