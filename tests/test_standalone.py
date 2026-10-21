"""The studio carries its presets and assets with the installed Python package."""
from pathlib import Path

import lineage_lab
from lineage_lab.config import Scenario
from lineage_lab.engine import generate
from lineage_lab.studio import Studio


def test_bundled_presets_are_the_published_examples():
    package = Path(lineage_lab.__file__).parent
    for name in ("civilization", "medieval", "fantasy", "hybrid"):
        assert (package / "presets" / f"{name}.yaml").read_bytes() == (
            Path("config/genealogy") / f"{name}.yaml"
        ).read_bytes()


def test_studio_presets_work_outside_the_checkout(tmp_path, monkeypatch):
    archive = generate(Scenario(initial_population=50, years=2), None)
    monkeypatch.chdir(tmp_path)
    presets = Studio(archive).presets()
    assert {"blank", "current", "civilization", "medieval", "fantasy", "hybrid"} == set(presets)
    package = Path(lineage_lab.__file__).parent
    for asset in ("explorer.html", "studio.css", "studio.js", "vendor/d3-delaunay.min.js"):
        assert (package / asset).is_file()
