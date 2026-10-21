# Lineage Lab

Standalone Python population sandbox. Source is `src/lineage_lab/`; no dependency on the original RPG repository, its `src` namespace or any sibling project.
Run `python -m pytest` and `ruff check src tests` after behavior changes. Preserve seeded full-mode histories and explicitly distinguish projected hybrid people from simulated people.
Keep assets and presets in package data, and verify an installed wheel outside the checkout when changing packaging. Never commit output, caches, credentials or virtual environments.

Keep all user-facing text, accessibility labels, errors and documentation in English. Preserve original-language names in datasets and scenario proper names.
