# Cousinade

Standalone Python population sandbox. Source is `src/cousinade/`; no dependency on the original RPG repository, its `src` namespace or any sibling project.
Run `python -m pytest` and `ruff check src tests` after behavior changes. Preserve seeded full-mode histories and explicitly distinguish projected hybrid people from simulated people.
Keep assets and presets in package data, and verify an installed wheel outside the checkout when changing packaging. Never commit output, caches, credentials or virtual environments.
