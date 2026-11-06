"""Descriptive personas: inheritance, readings, modes and history independence."""

import numpy as np
import pytest
from pydantic import ValidationError

from lineage_lab.config import Scenario
from lineage_lab.engine import generate
from lineage_lab.hybrid_archive import HybridArchive
from lineage_lab.persona import TRAIT_INDEX, TRAITS, PersonaModel, hair_colors
from lineage_lab.store import Archive


def world(**changes):
    return Scenario(**{"seed": 41, "initial_population": 1500, "years": 90, **changes})


def known_children(archive):
    father, mother = archive.data["father"], archive.data["mother"]
    return np.flatnonzero((father >= 0) & (mother >= 0)), father, mother


def values_of(archive):
    return PersonaModel(archive.config).values(
        archive.data["father"], archive.data["mother"], archive.data["race"]
    )


@pytest.mark.parametrize("trait", ["height", "extraversion"])
def test_children_regress_to_midparent_at_the_configured_heritability(trait):
    archive = generate(world(initial_population=4000), None)
    children, father, mother = known_children(archive)
    column = TRAIT_INDEX[trait]
    values = values_of(archive)[:, column]
    midparent = (values[father[children]] + values[mother[children]]) / 2
    slope = np.polyfit(midparent, values[children], 1)[0]
    assert slope == pytest.approx(TRAITS[column].heritability, abs=0.06)
    assert values.std() == pytest.approx(1, abs=0.06)


def test_two_blond_parents_mostly_have_blond_children():
    archive = generate(world(initial_population=4000), None)
    children, father, mother = known_children(archive)
    values = values_of(archive)
    colors = hair_colors(values)
    both_blond = (colors[father[children]] == "blond") & (colors[mother[children]] == "blond")
    neither_blond = (colors[father[children]] != "blond") & (colors[mother[children]] != "blond")
    assert both_blond.sum() > 30
    assert (colors[children][both_blond] == "blond").mean() > 0.55
    assert (colors[children][neither_blond] == "blond").mean() < 0.25


def test_persona_is_reproducible_and_leaves_recorded_people_untouched():
    first, second = generate(world(), None), generate(world(), None)
    before = {name: column.copy() for name, column in first.data.items()}
    person = first.n - 1
    assert first.person(person)["persona"] == second.person(person)["persona"]
    for name, column in before.items():
        assert np.array_equal(column, first.data[name])


def test_sqlite_archive_reports_the_memory_persona(tmp_path):
    config = world(initial_population=300, years=40)
    memory = generate(config, None)
    path = tmp_path / "world.sqlite"
    generate(config, path)
    person = memory.n - 1
    assert Archive(path).person(person)["persona"] == memory.person(person)["persona"]


def test_full_persona_exposes_every_catalogue_group_and_a_text_summary():
    archive = generate(world(initial_population=300, years=40), None)
    persona = archive.person(archive.n - 1)["persona"]
    assert persona["source"] == "inherited"
    assert persona["physical"]["height_cm"] > 0
    assert persona["physical"]["hair_color"]
    assert set(persona["temperament"]) == {
        "openness",
        "conscientiousness",
        "extraversion",
        "agreeableness",
        "neuroticism",
    }
    assert 0 <= persona["cognitive"]["intelligence"]["percentile"] <= 100
    assert "robustness" in persona["health"]
    assert persona["summary"].endswith(".")


def describe(values, sex=0, birth=1000, year=1030, status=0):
    model = PersonaModel(world())
    return model.describe(values, sex, 0, birth, year, status, "inherited")


def test_readings_are_derived_from_ocean_signs():
    values = np.zeros(len(TRAITS), np.float32)
    for name, value in {
        "extraversion": 1.5,
        "openness": 1.5,
        "agreeableness": -1.5,
        "conscientiousness": -1.5,
        "neuroticism": 1.5,
    }.items():
        values[TRAIT_INDEX[name]] = value
    readings = describe(values)["readings"]
    assert readings["type16"] == "ENTP"
    assert readings["classical_temperament"] == "choleric"
    assert "curious" in readings["adjectives"]


def test_courage_is_a_temperament_reading_not_a_stored_trait():
    values = np.zeros(len(TRAITS), np.float32)
    values[TRAIT_INDEX["neuroticism"]] = 2.5
    values[TRAIT_INDEX["extraversion"]] = -2.5
    assert "cowardly" in describe(values)["readings"]["adjectives"]
    assert "courage" not in TRAIT_INDEX


def test_age_changes_the_body_but_not_the_mind():
    values = np.ones(len(TRAITS), np.float32)
    child, adult = describe(values, year=1005), describe(values, year=1030)
    assert child["physical"]["height_cm"] < adult["physical"]["height_cm"]
    assert child["temperament"] == adult["temperament"]
    assert child["cognitive"] == adult["cognitive"]


def test_old_people_turn_grey_and_keep_their_natural_color():
    values = np.zeros(len(TRAITS), np.float32)
    values[TRAIT_INDEX["hair_pigment"]] = -2
    old = describe(values, year=1085)["physical"]
    assert old["hair_color"] == "grey"
    assert old["natural_hair_color"] == "blond"


def test_only_body_readings_depend_on_sex():
    values = np.zeros(len(TRAITS), np.float32)
    male, female = describe(values, sex=0), describe(values, sex=1)
    assert male["physical"]["height_cm"] > female["physical"]["height_cm"]
    for group in ("cognitive", "temperament", "health"):
        assert male[group] == female[group]


def test_race_offsets_persist_across_generations():
    config = world(
        initial_population=2000,
        races=[{"name": "dwarf", "metadata": {"persona": {"height": {"mean": -3}}}}],
    )
    archive = generate(config, None)
    values = values_of(archive)[:, TRAIT_INDEX["height"]]
    children, _, _ = known_children(archive)
    assert values[children].mean() == pytest.approx(-3, abs=0.1)


def test_mixed_parents_pull_a_child_towards_both_races():
    races = [
        {"name": "tall", "metadata": {"persona": {"height": {"mean": 3}}}},
        {"name": "short", "metadata": {"persona": {"height": {"mean": -3}}}},
    ]
    model = PersonaModel(Scenario(races=races))
    size = 3000
    # Even founders are tall, odd founders short; each pair has two tall-race children.
    father = np.r_[np.full(size, -1), np.arange(0, size, 2).repeat(2)].astype(np.int32)
    mother = np.r_[np.full(size, -1), np.arange(1, size, 2).repeat(2)].astype(np.int32)
    race = np.r_[np.arange(size) % 2, np.zeros(size)].astype(np.int16)
    height = model.values(father, mother, race)[:, TRAIT_INDEX["height"]]
    assert height[size:].mean() == pytest.approx(3 * (1 - TRAITS[0].heritability), abs=0.1)


def test_unknown_race_persona_trait_is_rejected():
    with pytest.raises(ValidationError, match="Unknown persona trait"):
        Scenario(races=[{"name": "elf", "metadata": {"persona": {"courage": {"mean": 1}}}}])


def test_nonpositive_race_persona_spread_is_rejected():
    with pytest.raises(ValidationError, match="sd"):
        Scenario(races=[{"name": "elf", "metadata": {"persona": {"height": {"sd": 0}}}}])


def test_default_scenario_dump_is_unchanged_by_personas():
    assert "persona" not in Scenario().model_dump(mode="json")


def hybrid_world():
    return world(population_model="hybrid", initial_population=2000, years=75)


def background_person(archive):
    segment = max(
        range(len(archive.background.records)),
        key=lambda i: (archive.background.records[i]["birth"], sum(archive.background.records[i]["counts"])),
    )
    return archive._id(segment + 1, 0)


def test_projected_persona_is_marked_as_an_approximation():
    archive = generate(hybrid_world(), None)
    person = archive.person(background_person(archive))
    assert person["source"] == "projected"
    assert person["persona"]["source"] == "approximated"


def test_core_persona_is_inherited():
    archive = generate(hybrid_world(), None)
    person = archive.person(archive._id(0, archive.core.n - 1))
    assert person["persona"]["source"] == "inherited"


def test_projected_persona_is_independent_of_lookup_order_and_cache():
    archive = generate(hybrid_world(), None)
    identity = background_person(archive)
    first = archive.person(identity)["persona"]
    archive.clear_cache()
    father = archive.person(identity)["father"]
    if father:
        archive.person(father)
    archive.clear_cache()
    assert archive.person(identity)["persona"] == first


def test_projected_persona_survives_a_saved_archive(tmp_path):
    config = hybrid_world()
    memory = generate(config, None)
    path = tmp_path / "hybrid.sqlite"
    generate(config, path)
    identity = background_person(memory)
    assert HybridArchive.open(path).person(identity)["persona"] == memory.person(identity)["persona"]


def test_projected_children_resemble_their_parents():
    archive = generate(hybrid_world(), None)
    background, column = archive.background, TRAIT_INDEX["height"]
    rows = []
    for index, record in enumerate(background.records):
        for slot in range(min(sum(record["counts"]), 8)):
            father, mother = background.parents(index, slot)
            if father is not None:
                rows.append(((index + 1, slot), father, mother))
        if len(rows) >= 300:
            break
    assert len(rows) >= 300
    values = {}

    def value(reference):
        if reference not in values:
            race = background.records[reference[0] - 1]["race"]
            values[reference] = archive.projected_values(reference, race)[column]
        return values[reference]

    child = np.array([value(row[0]) for row in rows])
    midparent = np.array([(value(row[1]) + value(row[2])) / 2 for row in rows])
    slope = np.polyfit(midparent, child, 1)[0]
    assert slope == pytest.approx(TRAITS[column].heritability, abs=0.15)


def test_founders_are_not_reported_as_inheriting_from_parents():
    archive = generate(world(initial_population=300, years=40), None)
    assert archive.person(0)["father"] is None
    assert archive.person(0)["persona"]["source"] == "founder"
