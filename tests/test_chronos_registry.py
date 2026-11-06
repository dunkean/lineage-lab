"""Executable S3 registry contracts and independent exported-table checks."""

import copy
from fractions import Fraction

import numpy as np
import pytest
from test_chronos_bridge import example, seal, society_parameters

from lineage_lab.chronos_bridge import complete_result
from lineage_lab.chronos_plan import translate_plan
from lineage_lab.cohort_residence import ResidentRegistry
from lineage_lab.config import Scenario
from lineage_lab.historical_program import ProviderError
from lineage_lab.integrated_core import precompute_core
from lineage_lab.integrated_population import IntegratedPopulation
from lineage_lab.population_export import export_tables, validate_population_result
from lineage_lab.registry_contract import PHASES, address, half_up, stream
from lineage_lab.registry_demography import RegistryDemography
from lineage_lab.registry_mobility import RegistryMobility, routes
from lineage_lab.site_regulation import SiteRegulator


def test_phase_and_random_reference_vectors():
    assert list(PHASES.values()) == [10, 20, 30, 35, 40, 80]
    assert half_up(Fraction(5, 2)) == 3
    assert half_up(Fraction(1, 2)) == 1
    assert half_up(Fraction(-5, 2)) == -3
    assert address("domain/a", "a/b", 7) != address("domain/a/a", "b", 7)
    assert stream(42, "mortality", 1300).integers(0, 2**32, 4).tolist() == [
        1172437434,
        3481787920,
        495763068,
        2443831700,
    ]


def test_integrated_settings_preserve_plan_fields():
    plan = example("runtime")
    config = translate_plan(plan)
    restored = Scenario.model_validate_json(config.model_dump_json())
    settings = restored.historical
    for field in ("historical_program", "boundary_ports", "shocks", "regulation"):
        assert getattr(settings, field) == plan[field]
    assert settings.external_profiles == plan["resolved_profiles"]["external"]
    assert settings.events[0].end_year == plan["shocks"][0]["valid"]["end_year"] - 1
    assert settings.mobility_edges[0]["mode"] == plan["mobility_edges"][0]["mode"]


def registry():
    return ResidentRegistry(translate_plan(example("runtime")), "projected")


def test_sealed_origins_and_birth_year_deaths_preserve_identity():
    r = registry()
    parents = r.add_cohort("founder", 0, 0, 1270, [0, 1])
    r.partition(founder=True)
    original = r.ids.copy()
    r.open_year(1301)
    child = r.add_cohort("birth", 0, 0, 1301, [1], parents=[(1, 0)])[0]
    r.die([child], 1301)
    assert r.ids[:2] == original
    assert r.children[1] == r.children[0] == [child]
    episode = r.residences[r.residence_index[child]]
    assert episode["valid"] == {"start_year": 1301, "end_year": 1301}
    assert (episode["start_phase"], episode["end_phase"]) == (20, 30)
    assert r.data["death_site"][child] == 0
    assert r.cohorts[0]["sex_counts"] == [1, 1]
    assert len(parents) == 2


def test_households_widowhood_majority_and_guardians():
    r = registry()
    adults = r.add_cohort("founder", 0, 0, 1270, [0, 1])
    children = r.add_cohort("founder", 0, 0, 1295, [0, 1])
    r.partner = {0: 1, 1: 0}
    r.partition(founder=True)
    hh = r.data["household"][adults[0]]
    assert all(r.data["guardian"][children] >= 0)
    assert all(r.data["household"][children] == hh)
    r.open_year(1301)
    r.die([0], 1301)
    r.partner.clear()
    r.partition()
    assert r.data["household"][1] == hh
    assert all(r.data["guardian"][children] == 1)
    r.open_year(1302)
    r.die([1], 1302)
    r.partition()
    care = r.households[r.data["household"][children[0]]]
    assert care["kind"] == "synthetic_care"
    assert not r.transfer(r.data["household"][children[0]], 1, reason="test")
    r.open_year(1313)
    r.partition()
    assert r.data["household"][children[0]] != r.data["household"][children[1]]


@pytest.mark.parametrize("widowed", [False, True])
def test_majority_leaves_parent_household_and_preserves_survivor_identity(widowed):
    r = registry()
    r.add_cohort("founder", 0, 0, 1260, [0, 1])
    child = r.add_cohort("founder", 0, 0, 1283, [1])[0]
    r.partner = {0: 1, 1: 0}
    r.partition(founder=True)
    parental_household = int(r.data["household"][0])
    assert r.data["household"][child] == parental_household
    r.open_year(1301)
    if widowed:
        r.die([0], 1301)
    r.partition()
    assert r.data["household"][1] == parental_household
    assert r.data["household"][child] != parental_household
    assert r.households[r.data["household"][child]]["kind"] == "single"


def test_background_migrant_keeps_id_and_closes_balance():
    r = registry()
    rows = r.add_cohort("founder", 0, 0, 1270, [0, 1])
    r.partition(founder=True)
    ids = r.ids.copy()
    before = r.counts()
    r.open_year(1301)
    assert r.transfer(r.data["household"][rows[0]], 1, reason="programmed")
    assert r.ids == ids
    assert np.array_equal(r.counts(), before + r.counters[:, 2] - r.counters[:, 3])
    assert r.transfers[0]["origin_slots"] == [0]


def test_split_cohort_deaths_stay_in_exposed_site():
    r = registry()
    rows = r.add_cohort("founder", 0, 0, 1270, [0] * 100)
    r.partition(founder=True)
    r.open_year(1303)
    for row in rows[:50]:
        r.transfer(r.data["household"][row], 1, reason="test")
    r.config.historical.events[0].extra_mortality = 1
    # Natural mortality off: only the shock may kill, and only at the exposed site.
    r.config.races[0].demography.adult_mortality = 0
    r.config.races[0].demography.aging_coefficient = 0
    RegistryDemography(r).deaths()
    assert len(r.deaths) == 50
    assert {e["site_id"] for e in r.deaths} == {r.site_ids[0]}
    assert r.counts().tolist() == [0, 50]


def test_separated_potential_pairs_do_not_reproduce():
    r = registry()
    rows = r.add_cohort("founder", 0, 0, 1273, [0, 1])
    r.seal_pairs(rows)
    r.partition(founder=True)
    r.open_year(1301)
    r.transfer(r.data["household"][0], 1, reason="test")
    r.activate_pairs()
    r.open_year(1302)
    demography = RegistryDemography(r)
    demography.births()
    assert not r.births and not r.unions
    assert demography.exposure["separated_potential_pair_years"] == 1


def test_local_repairing_replaces_a_dead_sealed_partner():
    r = registry()
    rows = r.add_cohort("founder", 0, 0, 1270, [0, 1, 0, 1])
    r.seal_pairs(rows)
    r.partition(founder=True)
    r.config.society.marriage_rate = 1
    r.open_year(1301)
    pairs = list(r.pairs)
    assert len(pairs) == 2
    r.die([pairs[0]["a"], pairs[1]["b"]], 1301)
    r.activate_pairs()
    r.repair_local()
    assert len(r.unions) == 1
    union = r.unions[0]
    assert set(union["partner_slots"]) == {pairs[0]["b"], pairs[1]["a"]}
    assert "local_repair" in union["family_key"]
    r.partition()
    a, b = union["partner_slots"]
    assert r.data["household"][a] == r.data["household"][b]


def test_local_repairing_excludes_kin_former_partners_and_other_sites():
    r = registry()
    r.config.society.marriage_rate = 1
    parents = r.add_cohort("founder", 0, 0, 1250, [0, 1])
    r.add_cohort("founder", 0, 0, 1275, [0, 1], parents=[(parents[1], parents[0])] * 2)
    r.partition(founder=True)
    r.open_year(1301)
    r.die(parents, 1301)
    r.repair_local()
    assert not r.unions, "siblings cannot re-pair"

    r = registry()
    r.config.society.marriage_rate = 1
    rows = r.add_cohort("founder", 0, 0, 1270, [0, 1])
    r.partition(founder=True)
    r.open_year(1301)
    r.former_partners.add(frozenset(map(int, rows)))
    r.repair_local()
    assert not r.unions, "a closed pair never forms again"
    r.former_partners.clear()
    r.transfer(r.data["household"][rows[0]], 1, reason="test")
    r.repair_local()
    assert not r.unions, "re-pairing stays within one site"


def test_core_event_tape_and_founder_guardians():
    config = translate_plan(example("runtime"))
    core, size = precompute_core(config)
    r = core.registry
    assert size == 18
    assert len(core.annual) == 8
    assert all(e["phase"] in (10, 20, 30, 35, 40) for e in core.tape)
    assert {e["transfer_id"] for e in core.tape if e["kind"] == "transfer"} == {
        e["transfer_id"] for e in r.transfers
    }
    assert all(e["source"] == "founder_assignment" for e in r.guardians if e["phase"] == 10)
    assert len(r.ids) == len(set(r.ids))


def result_plan():
    p = example("runtime")
    p["limits"]["core_target_share_bps"] = 0
    p["boundary_ports"] = []
    society_parameters(p, migration_rate_bps=0)
    return p


def materialize(p):
    seal(p)
    return complete_result(p, translate_plan(p))


def test_every_resident_has_one_household():
    p = result_plan()
    result = materialize(p)
    assert validate_population_result(result, p["limits"])["valid"]
    broken = copy.deepcopy(result)
    broken["household_episodes"].pop()
    with pytest.raises(ProviderError, match="household"):
        validate_population_result(broken, p["limits"])


@pytest.mark.parametrize("field", ["population", "births", "deaths", "incoming", "outgoing"])
def test_balance_is_recomputed_from_tables_independently_of_counters(field):
    p = result_plan()
    result = materialize(p)
    result["annual_censuses"][0][field] += 1
    with pytest.raises(ProviderError, match="independently recomputed"):
        validate_population_result(result, p["limits"])


def test_ordinary_projected_migration_uses_cultural_rate():
    r = registry()
    r.config.society.migration_rate = 0.06
    r.add_cohort("founder", 0, 0, 1270, [0] * 100)
    r.partition(founder=True)
    r.open_year(1301)
    m = RegistryMobility(r, RegistryDemography(r))
    opening = m.opening()
    m.ordinary(opening, np.zeros(2, np.int64))
    assert len(r.transfers) == 6
    assert all(t["reason"] == "ordinary" for t in r.transfers)
    assert len({p for t in r.transfers for p in t["origin_slots"]}) == 6


def test_mobility_plan_is_self_contained():
    r = registry()
    r.settings.mobility_edges.clear()
    r.config.society.migration_rate = 1
    r.add_cohort("founder", 0, 0, 1270, [0] * 10)
    r.partition(founder=True)
    r.open_year(1301)
    m = RegistryMobility(r, RegistryDemography(r))
    assert routes(r.config, 1301) == [[0, None], [None, 0]]
    m.ordinary(m.opening(), np.zeros(2, np.int64))
    assert not r.transfers
    assert sum(d["unserved_households"] for d in m.diagnostics) == 10


def test_statistical_ports_are_not_named_transfers():
    r = registry()
    r.add_cohort("founder", 0, 0, 1270, [0] * 100)
    r.partition(founder=True)
    port = r.settings.boundary_ports[0]
    port["rate_episodes"][0].update(incoming_rate_bps=100, outgoing_rate_bps=100)
    r.open_year(1301)
    m = RegistryMobility(r, RegistryDemography(r))
    m.ports(m.opening())
    arrivals = [t for t in r.transfers if t["from_site"] is None]
    assert len(arrivals) == 1
    assert all(t["port_id"] == port["port_id"] and t["operation_id"] is None for t in r.transfers)
    for person in arrivals[0]["origin_slots"]:
        assert r.data["mother"][person] == r.data["father"][person] == -1
        assert r.cohorts[r.data["cohort"][person]]["origin_type"] == "arrival"
        assert r.data["household"][person] in r.moved
    departed = next(t for t in r.transfers if t["to_site"] is None)["origin_slots"][0]
    assert r.data["death"][departed] == -(2**31)
    assert r.data["coverage_end"][departed] == 1301


def test_founder_dependents_follow_declared_guardians():
    r = registry()
    r.add_cohort("founder", 0, 0, 1270, [0, 1])
    children = r.add_cohort("founder", 0, 0, 1290, [0, 1])
    r.partner = {0: 1, 1: 0}
    r.partition(founder=True)
    r.open_year(1301)
    guardian = int(r.data["guardian"][children[0]])
    r.transfer(int(r.data["household"][guardian]), 1, reason="programmed")
    assert all(r.data["site"][children] == 1)
    assert all(r.data["mother"][children] == -1)
    assert set(children).issubset(r.transfers[-1]["origin_slots"])


def test_family_links_survive_projection_order():
    r = registry()
    r.add_cohort("founder", 0, 0, 1270, [0, 1])
    r.partner = {0: 1, 1: 0}
    r.partition(founder=True)
    r.open_year(1301)
    r.transfer(int(r.data["household"][0]), 1, reason="test")
    family = r.family[1]
    r.add_cohort("birth", 1, 0, 1301, [1], parents=[(1, 0)], families=[family])
    other = registry()
    other.source = "simulated"
    other.add_cohort("founder", 1, 0, 1260, [1])
    assert export_tables([r, other]) == export_tables([other, r])
    table = export_tables([r])
    child = table["birth_allocations"][0]["child_slot"]
    mother = table["candidate_table"]["mother_slot"][child]
    assert child in table["children_index"][mother]
    assert table["candidate_table"]["family_key"][child] == family


def test_marker_does_not_anticipate_shock_on_real_materialize_path():
    p = result_plan()
    a = materialize(p)
    p["shocks"] = []
    b = materialize(p)
    assert [r for r in a["annual_censuses"] if r["year"] < 1303] == [
        r for r in b["annual_censuses"] if r["year"] < 1303
    ]
    assert [r for r in a["diagnostics"]["regulation"] if r["year"] < 1303] == [
        r for r in b["diagnostics"]["regulation"] if r["year"] < 1303
    ]


def test_regulator_does_not_cancel_crisis_and_spared_village_keeps_baseline_target():
    p = result_plan()
    p["shocks"][0]["effects"] = {"additional_mortality_bps": 5000}
    a = materialize(p)
    site_a, site_b = [s["site_id"] for s in p["settlements"]]
    rows_a = [r for r in a["diagnostics"]["regulation"] if r["site_id"] == site_a]
    rows_b = [r for r in a["diagnostics"]["regulation"] if r["site_id"] == site_b]
    assert all(r["multiplier_ppm"] == 1000000 and r["frozen"] for r in rows_a if r["year"] >= 1303)
    assert all(r["q"] == [1, 1] and not r["frozen"] for r in rows_b)
    counts = {r["year"]: r["population"] for r in a["annual_censuses"] if r["site_id"] == site_a}
    assert counts[1304] < counts[1302] // 2


def test_recovery_rebase_includes_same_year_transfer():
    p = result_plan()
    p["shocks"][0]["recovery_years"] = 1
    result = materialize(p)
    site = p["settlements"][0]["site_id"]
    states = {r["year"]: r for r in result["diagnostics"]["regulation"] if r["site_id"] == site}
    census = {r["year"]: r for r in result["annual_censuses"] if r["site_id"] == site}
    row = states[1306]
    assert row["rebased"] and row["frozen"] and row["multiplier_ppm"] == 1000000
    assert census[1306]["outgoing"] > 0
    assert Fraction(*row["q"]) == Fraction(census[1306]["population"], 240)
    assert row["M_ref"] == row["M"] == -census[1306]["outgoing"]
    assert states[1307]["multiplier_ppm"] == 1000000 and not states[1307]["frozen"]


def test_displacement_is_not_cancelled_by_regulator():
    p = result_plan()
    p["shocks"] = []
    result = materialize(p)
    site = p["settlements"][0]["site_id"]
    row = next(
        r for r in result["diagnostics"]["regulation"] if r["site_id"] == site and r["year"] == 1307
    )
    moved = sum(
        len(t["origin_slots"]) for t in result["residence_transfers"] if t["from_site"] == site
    )
    assert moved > 0
    assert row["M"] == -moved
    assert Fraction(*row["guide_raw"]) == 240 - moved


def test_controller_overlap_zero_baseline_and_exact_quantization():
    p = result_plan()
    site = p["settlements"][0]["site_id"]
    p["shocks"][0].update(valid={"start_year": 1301, "end_year": 1302}, recovery_years=1)
    second = copy.deepcopy(p["shocks"][0])
    second.update(
        shock_id="episode/overlap",
        source_event_id="overlap",
        valid={"start_year": 1303, "end_year": 1304},
    )
    p["shocks"].append(second)
    c = SiteRegulator(translate_plan(p))
    for year in range(1300, 1307):
        c.begin(year, [0, 0], [120, 120])
        c.publish(year, [0, 0], [120, 120], np.zeros(2, int), np.zeros(2, int), [0, 0])
    rows = {r["year"]: r for r in c.rows if r["site_id"] == site}
    assert not rows[1303]["rebased"]
    assert rows[1305]["rebased"] and rows[1305]["multiplier_ppm"] == 1000000
    assert not rows[1306]["frozen"]
    for marker in p["baseline_markers"]:
        if marker["site_id"] == site:
            marker["population"] = 0 if marker["year"] == 1300 else 70
    p["shocks"] = []
    c = SiteRegulator(translate_plan(p))
    assert c.begin(1300, [0, 0], [0, 120])[0] == 1000000
    c.publish(1300, [0, 0], [10, 120], np.array([10, 0]), np.zeros(2, int), [0, 0])
    assert c.states[0]["zero"] and c.states[0]["q"] is None
    assert c.begin(1301, [0, 0], [10, 120])[0] == 1000000
    c.publish(1301, [0, 0], [10, 120], np.zeros(2, int), np.zeros(2, int), [0, 0])
    assert c.states[0]["q"] == 1 and not c.states[0]["zero"]


def test_crisis_is_applied_once_on_registry_path():
    r = registry()
    r.add_cohort("founder", 0, 0, 1270, [0] * 10000)
    r.partition(founder=True)
    r.open_year(1303)
    demo = RegistryDemography(r)
    natural = r.config.races[0].demography
    natural.adult_mortality = natural.aging_coefficient = 0
    r.settings.events[0].extra_mortality = 0.08
    demo.deaths()
    assert len(r.deaths) == 800


def test_newcomers_have_no_excess_mortality():
    totals = np.zeros(4, np.int64)  # established deaths/exposed, newcomer deaths/exposed
    for seed in range(24):
        r = registry()
        r.config.seed = seed
        event = r.settings.events[0]
        event.start_year, event.end_year = 1300, 1304
        event.extra_mortality, event.mortality_factor = 0, 0
        natural = r.config.races[0].demography
        natural.adult_mortality, natural.aging_coefficient = 0.2, 0
        natural.male_mortality_factor = 1
        rows = r.add_cohort("founder", 0, 0, 1270, [0] * 1000)
        r.partition(founder=True)
        r.open_year(1300)
        for row in rows[:500]:
            r.transfer(int(r.data["household"][row]), 1, reason="setup")
        demo = RegistryDemography(r)
        # Established residents accumulate prior exposure; prospective newcomers
        # are absent from that site's annual ordering until the final transition.
        for year in range(1301, 1305):
            r.open_year(year)
            demo.deaths()
        r.open_year(1305)
        established = r.living()[r.data["site"][r.living()] == 1]
        for row in rows[500:]:
            assert r.transfer(int(r.data["household"][row]), 1, reason="arrival")
        assert np.all(r.data["site"][500:1000] == 1)
        demo.deaths()
        totals += [
            sum(r.data["death"][established] == 1305),
            len(established),
            sum(r.data["death"][500:1000] == 1305),
            500,
        ]
    assert abs(totals[0] / totals[1] - totals[2] / totals[3]) < 0.025


def test_successful_birth_spacing_is_per_ancestry_with_temporary_interruption():
    r = registry()
    r.config.society.founder_match_participation = 1
    first = r.config.races[0]
    first.demography.birth_spacing = 1
    first.demography.fertility_peak = 1
    second = first.model_copy(deep=True)
    second.name, second.demography.birth_spacing = "second", 4
    r.config.races.append(second)
    original = []
    for race in (0, 1):
        rows = r.add_cohort("founder", 0, race, 1273, [0, 1] * 20, flags=np.zeros(40, np.int32))
        r.seal_pairs(rows)
        original.extend(rows)
    r.activate_pairs(initial=True)
    r.partition(founder=True)
    event = r.settings.events[0]
    event.start_year = event.end_year = 1303
    event.fertility_factor = 0
    demo = RegistryDemography(r)
    counts = [0, 0]
    previous = {}
    for year in range(1301, 1307):
        r.open_year(year)
        r.data["birth"][original] = year - 27
        before = r.n
        mothers, _ = demo.births()
        if year == 1303:
            assert not len(mothers)
        for mother in mothers:
            race = int(r.data["race"][mother])
            counts[race] += 1
            if mother in previous:
                assert year - previous[mother] >= (1, 4)[race]
            previous[mother] = year
        r.data["site"][before : r.n] = -1
        del r.pairs[40:]
        del r.open_pairs[40:]
    assert counts == [100, 40]


def test_core_tape_replay_and_residual_household_requests():
    p = example("runtime")
    p["historical_program"][0].update(year=1300, target_households=20)
    p["boundary_ports"] = []
    p["shocks"] = []
    society_parameters(p, migration_rate_bps=0)
    world = IntegratedPopulation(translate_plan(p)).run()
    core, registry = world.core, world.core.registry
    operation = core.operations[0]
    background = next(r for r in world.mobility.diagnostics if r["kind"] == "operation")
    assert operation["allocated_core_households"] == 1
    assert background["actual_core_households"] == operation["actual_core_households"]
    assert background["actual_projected_households"] == 20 - operation["actual_core_households"]
    previous = np.zeros(2, np.int64)
    for row in registry.residences:
        if row["valid"]["start_year"] == 1300 and row["start_phase"] == 10:
            previous[registry.public_index[row["site_id"]]] += 1
    for year in range(1300, 1308):
        closing = previous.copy()
        for event in core.tape:
            if event["year"] != year:
                continue
            if event["kind"] in {"birth", "death"}:
                closing[registry.public_index[event["site_id"]]] += (
                    1 if event["kind"] == "birth" else -1
                )
            elif event["kind"] == "transfer":
                closing[registry.public_index[event["from_site"]]] -= len(event["origin_slots"])
                closing[registry.public_index[event["to_site"]]] += len(event["origin_slots"])
        assert np.array_equal(closing, core.annual[year]["counts"])
        previous = closing


def test_coresident_marriage_is_not_a_household_move():
    config = translate_plan(example("runtime"))
    core, _ = precompute_core(config)
    r = core.registry
    candidates = [p for p in core.alive if core.data["partner"][p] >= 0]
    if not candidates:
        # Use the bootstrap source with deterministic full founder participation.
        config.society.founder_match_participation = 1
        core, _ = precompute_core(config)
        r = core.registry
        candidates = [p for p in core.alive if core.data["partner"][p] >= 0]
    assert candidates
    person = int(candidates[0])
    row = core.slot_map[person]
    hh = int(r.data["household"][row])
    r.moved.clear()
    core.pending_marriages = [(person, int(r.data["site"][row]))]
    transfers = len(r.transfers)
    core._commit_marriages()
    assert len(r.transfers) == transfers and hh not in r.moved


def test_regulator_saturation_signed_guide_and_unavailable_fertility():
    r = registry()
    c = SiteRegulator(r.config)
    c.states[0]["M"] = -300
    multiplier = c.begin(1301, [10, 0], [1000, 120])
    assert c.pending[0]["guide_raw"] == [-60, 1]
    assert c.pending[0]["guide"] == [0, 1]
    assert c.pending[0]["saturated"] and c.pending[0]["core_above_guide"]
    assert multiplier[0] == r.settings.regulation["min_fertility_multiplier_ppm"]
    r.config.society.founder_match_participation = 1
    r.config.races[0].demography.fertility_peak = 0
    rows = r.add_cohort("founder", 0, 0, 1273, [0, 1], flags=[0, 0])
    r.seal_pairs(rows)
    r.activate_pairs(initial=True)
    r.partition(founder=True)
    r.open_year(1301)
    mothers, eligible = RegistryDemography(r).births([4000000, 4000000])
    assert not len(mothers) and eligible.tolist() == [0, 0]


def test_maternal_mortality_targets_actual_mothers():
    r = registry()
    r.config.society.founder_match_participation = 1
    demo = r.config.races[0].demography
    demo.fertility_peak = demo.maternal_mortality = 1
    demo.adult_mortality = demo.aging_coefficient = demo.infant_mortality = 0
    demo.child_mortality = 0
    rows = r.add_cohort("founder", 0, 0, 1274, [0] * 10 + [1] * 30, flags=[0] * 40)
    r.seal_pairs(rows)
    r.activate_pairs(initial=True)
    r.partition(founder=True)
    r.open_year(1301)
    demography = RegistryDemography(r)
    mothers, _ = demography.births()
    assert len(mothers) == 10
    demography.deaths(mothers)
    assert {row["person_slot"] for row in r.deaths} == set(mothers)


def test_projected_divorce_permanently_closes_union():
    r = registry()
    r.config.society.founder_match_participation = 1
    rows = r.add_cohort("founder", 0, 0, 1273, [0, 1], flags=[0, 0])
    r.seal_pairs(rows)
    r.activate_pairs(initial=True)
    r.partition(founder=True)
    r.open_year(1301)
    r.config.society.divorce_rate = 1
    r.partition(force_single=r.activate_pairs())
    assert r.data["household"][0] != r.data["household"][1]
    r.open_year(1302)
    r.config.society.divorce_rate = 0
    r.activate_pairs()
    assert len(r.unions) == 1 and r.pairs[0]["state"] == "closed"


def test_simultaneous_founder_guardian_deaths_use_care_without_inventing_parents():
    r = registry()
    r.add_cohort("founder", 0, 0, 1270, [0, 1])
    child = r.add_cohort("founder", 0, 0, 1295, [0])[0]
    r.partner = {0: 1, 1: 0}
    r.partition(founder=True)
    r.open_year(1301)
    r.die([0, 1], 1301)
    r.partition()
    assert r.households[r.data["household"][child]]["kind"] == "synthetic_care"
    assert r.data["mother"][child] == r.data["father"][child] == -1


def test_opening_draws_match_numpy():
    from lineage_lab.chronos_plan import canonical_bytes
    from lineage_lab.registry_contract import opening_draws, stream

    parts = [["death", index, "é\"", None, index % 2 == 0] for index in range(2000)]
    keys = [canonical_bytes(["chronos-registry-rng-1", "42", *p]) for p in parts]
    doubles, salts = opening_draws(keys)
    for part, double, salt in zip(parts, doubles, salts, strict=True):
        rng = stream(42, *part)
        assert double == rng.random()
        assert salt == rng.integers(0, 2**64, dtype=np.uint64)
