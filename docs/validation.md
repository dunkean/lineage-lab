# Validation

The standalone application is checked with Python 3.12.13 on Windows. Runtime versions are pinned in `requirements.lock`.

- 108 Python tests cover full/hybrid genealogies, events, determinism, archives, concurrency, settings, bundled presets outside the checkout and packaged web assets.
- Ruff checks source and tests.
- Edge checks cover generation, 500 settlements, dated nations, exploration, JSON editing, mobile layout, opaque hybrid IDs, lineage depth and return to full mode.
- Wheel and source distributions are built, with the installed wheel exercised in a fresh environment outside the checkout.
- The large historical benchmarks were not rerun for extraction or localization.

Use the commands in README.md. GitHub Actions runs Python tests, Ruff and package builds. No original RPG repository or LLM service is required.
