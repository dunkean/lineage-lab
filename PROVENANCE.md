# Provenance

Lineage Lab was extracted from the genealogy module of `dunkean/AI_RPG_Generator` on October 21, 2026. Source baseline: `b6e9142`, plus the local working version, including the `cohort-runtime-v2` hybrid model and its tests.

Only population code, scenarios, browser assets, tests and documentation were extracted. The LLM pipeline, private settings, credentials, calculated archives and caches were excluded. The source RPG repository was left intact.

The standalone repository was renamed in place; its commit history is preserved. The Python package is `lineage_lab` and the CLI is `lineage-lab`. Existing archive-format markers remain unchanged for compatibility. Bundled JavaScript libraries retain their notices under `src/lineage_lab/vendor/`.
