# Lineage Lab

A small, standalone sandbox for population histories and genealogies. Configure a world, run generations, then explore settlements, life events and family trees.

![Local demo](docs/images/demo.png)

Python, NumPy and Numba. No LLM, account, API key, map engine or sibling repository is required.

## Run the demo

Python **3.12** is the validated version. On Windows, double-click **`demo.cmd`**, then open **http://127.0.0.1:8772**. The first launch creates a private virtual environment and installs the pinned dependencies. Keep the terminal open.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.lock
.venv\Scripts\python.exe -m pip install -e ".[dev]"
.venv\Scripts\python.exe -m lineage_lab studio --port 8772
```

On Linux/macOS, use `.venv/bin/python`. The installed command is `lineage-lab`.

The first Numba compilation can take a few seconds. The default demo starts with 2,000 founders, 36 settlements and 60 simulated years. Generate a world, explore its map, then open a person and their family. Medieval, fantasy, civilization and hybrid presets are bundled, including in an installed wheel.

## Save or replay a world

The studio keeps worlds in RAM; closing the server releases them. Export a configuration and its seed to replay a run. To persist the actual histories, explicitly request an archive:

```powershell
.venv\Scripts\python.exe -m lineage_lab generate config/genealogy/fantasy.yaml --output output/fantasy.sqlite
.venv\Scripts\python.exe -m lineage_lab explore output/fantasy.sqlite --port 8772
```

Existing archives are never overwritten.

## Two population models

| Model | Contract | Main limit |
| --- | --- | --- |
| `full` (default) | Every person and their family events are simulated and recorded | Historical records, including the dead, consume memory |
| `hybrid` | A simulated core plus a cohort background with deterministic families on demand | Background families use explicit approximations and limited exploration depth |

For a complete genealogy of every resident, use `full`. Hybrid background exploration is limited to **two ancestor generations and five descendant generations**. It does not reproduce an equivalent full simulation. Both modes stop at founders whose earlier parents are unknown.

Rules are worldbuilding parameters, not a historically calibrated demographic model. Fifty million residents across a thousand years is not a demonstrated full-mode performance claim. Display names are placeholders; the project does not depend on Namebench. Scenario proper names can retain their original spelling.

## Check the project

```powershell
.venv\Scripts\python.exe -m pytest
.venv\Scripts\python.exe -m ruff check src tests
npm ci
# Start the studio on 8772 first. Edge is the default browser on Windows.
npm run test:browser
$env:STUDIO_URL="http://127.0.0.1:8772"
npm run test:hybrid
```

`STUDIO_BROWSER` selects a Chromium/Edge executable. Screenshots and test outputs go to ignored `output/`.

## Documentation

- [User guide](docs/genealogy.md)
- [Hybrid model contract](docs/genealogy-hybrid.md)
- [Architecture](docs/genealogy-architecture.html)
- [Design notes](docs/genealogy-hybrid-design.md)
- [Historical measurements](docs/genealogy_review.md)
- [Validation](docs/validation.md) and [provenance](PROVENANCE.md)

Source: `src/lineage_lab/`. Example scenarios: `config/genealogy/`. Tests: `tests/`. Presets and browser assets are packaged with the application.
