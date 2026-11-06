"""Chronos worker contracts, isolation, bootstrap and soft historical effects."""

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from lineage_lab.chronos_bridge import (
    PARTIAL_FORMAT,
    PROTOCOL,
    Worker,
    partial_result,
    result_path,
    select_core_variant,
)
from lineage_lab.chronos_plan import (
    CONTRACT,
    DEMOGRAPHY_FIELDS,
    SCHEMA,
    SOCIETY_FIELDS,
    canonical_bytes,
    fingerprint,
    plan_fingerprint,
    translate_parameters,
    translate_plan,
    validate_schema,
)
from lineage_lab.cohorts import Cohorts
from lineage_lab.config import Scenario
from lineage_lab.engine import Engine
from lineage_lab.historical_program import (
    CoreBudgetExceeded,
    HistoricalProgram,
    ProviderError,
    apply_initial_program,
    mortality_probability,
)


@pytest.fixture(scope="module")
def worker(tmp_path_factory):
    return Worker(tmp_path_factory.mktemp("chronos"))


def example(runtime, *, small=True):
    plan = json.loads(
        CONTRACT.joinpath("village-population-plan.example.json").read_text(encoding="utf-8")
    )
    plan["provider"]["runtime_fingerprint"] = runtime
    if small:
        start, end = 1300, 1308
        plan["timeline"]["end_year"] = end
        for i, site in enumerate(plan["settlements"]):
            site["initial_population"] = (240, 120)[i]
            site["occupation_episodes"][0]["valid"]["end_year"] = end
        plan["baseline_markers"] = [
            {
                "site_id": s["site_id"],
                "year": y,
                "population": s["initial_population"],
                "tolerance_bps": 2000,
            }
            for s in plan["settlements"]
            for y in (start, end - 1)
        ]
        plan["shocks"][0]["valid"] = {"start_year": 1303, "end_year": 1305}
        plan["historical_program"][0]["year"] = 1306
        for role in plan["role_demands"]:
            role["valid"]["end_year"] = end
        for port in plan["boundary_ports"]:
            port["valid"]["end_year"] = end
            port["rate_episodes"][0]["valid"]["end_year"] = end
        for edge in plan["mobility_edges"]:
            edge["valid"]["end_year"] = end
    return seal(plan)


def seal(plan):
    plan["plan_fingerprint"] = plan_fingerprint(plan)
    return plan


def request(worker, method, **kwargs):
    return worker.handle({"protocol": PROTOCOL, "request_id": "test", "method": method, **kwargs})


def read_result(worker, response):
    assert response["ok"], response
    root = worker.job_root / response["job_directory"]
    path = result_path(root, response["result_ref"]["relative_path"])
    raw = path.read_bytes()
    assert len(raw) == response["result_ref"]["bytes"]
    assert hashlib.sha256(raw).hexdigest() == response["result_ref"]["file_sha256"]
    result = json.loads(raw)
    assert raw == canonical_bytes(result)
    assert result["result_fingerprint"] == fingerprint(
        {k: v for k, v in result.items() if k != "result_fingerprint"}
    )
    assert response["result_fingerprint"] == result["result_fingerprint"]
    assert not list(root.glob("*.partial"))
    return result


def test_bundled_contract_bytes_are_pinned():
    expected = {
        "population-plan.schema.json": "0495fbfd2267405d594c39c0c9b4c649c8fe2aa74239f1bdb4391496d3f2d738",
        "village-population-plan.example.json": "47c0e1e5e39267537fb70a3474794c457b248f695a2255465b791478f358ef1f",
        "requirements.lock": "87274b79ba36271b52388f9acbe4dbe7eeabed43dd87e85f998372bbd2d22223",
    }
    for name, digest in expected.items():
        assert hashlib.sha256(CONTRACT.joinpath(name).read_bytes()).hexdigest() == digest
    assert (
        CONTRACT.joinpath("requirements.lock").read_bytes()
        == Path("requirements.lock").read_bytes()
    )


def test_canonical_reference_vector_and_integer_only():
    value = {"é": '"\\\n\x01/é', "a": [-12, 0, True, False, None]}
    assert (
        canonical_bytes(value)
        == b'{"a":[-12,0,true,false,null],"\xc3\xa9":"\\"\\\\\\n\\u0001/\xc3\xa9"}'
    )
    assert canonical_bytes({"z": 1, "a": 2}) == canonical_bytes({"a": 2, "z": 1})
    for invalid in (0.5, float("nan"), (1, 2), {1: "key"}):
        with pytest.raises(ProviderError):
            canonical_bytes(invalid)


def test_units_and_complete_validated_configuration(worker):
    plan = example(worker.runtime_id)
    config = translate_plan(plan)
    assert config.society.migration_rate == 0.006  # No second cultural mobility multiplier.
    assert config.society.distance_scale == 25
    assert config.society.remarriage_delay == 1
    assert config.races[0].demography.fertility_width == 10
    assert config.races[0].demography.aging_exponent == 0.095
    assert config.settlements[1].x == 15
    assert config.settlements[1].y == 5
    event = config.historical.events[0]
    assert event.extra_mortality == 0.08 and event.fertility_factor == 0.7
    assert event.end_year == 1304
    assert config.target_population is None and config.target_mode == "report"
    with pytest.raises(ValueError, match="no standalone"):
        Scenario.model_validate({**config.model_dump(), "target_population": 500})
    assert (
        set(
            translate_parameters(plan["resolved_profiles"]["society"]["parameters"], SOCIETY_FIELDS)
        )
        <= type(Scenario().society).model_fields.keys()
    )
    assert (
        set(
            translate_parameters(
                plan["resolved_profiles"]["ancestries"][0]["parameters"], DEMOGRAPHY_FIELDS
            )
        )
        <= type(Scenario().demography).model_fields.keys()
    )


def test_worker_receives_foster_limit(worker):
    plan = example(worker.runtime_id)
    plan["resolved_profiles"]["kinship"][0]["founder_foster_children_limit"] = 3
    seal(plan)
    assert request(worker, "prepare", plan=plan)["ok"]
    _, config = worker.prepared[plan["plan_id"]]
    assert config.historical.kinship_profiles[0]["founder_foster_children_limit"] == 3
    result = partial_result(plan, config)
    applied = result["provider_manifest"]["applied_configuration"]
    assert applied["historical"]["kinship_profiles"][0]["founder_foster_children_limit"] == 3
    assert "founder_assignments" in result["diagnostics"]["deferred"]


def test_unknown_parameter_is_refused(worker):
    plan = example(worker.runtime_id)
    plan["resolved_profiles"]["society"]["parameters"]["migration_magic_bps"] = 1
    response = request(worker, "prepare", plan=seal(plan))
    assert not response["ok"]
    assert "Unknown profile parameter" in response["error"]["message"]
    assert response["error"]["code"] == "PopulationConstraintsUnsatisfied"
    assert response["error"]["plan_id"] == plan["plan_id"]
    assert {"owner", "year", "constraint"} <= response["error"].keys()


@pytest.mark.parametrize(
    "change",
    [
        "fixed_people",
        "exact_constraints",
        "protect_person",
        "bridge",
        "procedural",
        "capability",
        "dynamic",
        "custom",
        "exact_move",
    ],
)
def test_undelivered_capability_is_refused(worker, change):
    plan = example(worker.runtime_id)
    if change == "fixed_people":
        plan["fixed_people"] = [
            {
                "person_id": "p",
                "birth_year": 1280,
                "death_year": None,
                "ancestry_profile_id": "human-example",
                "sex": "F",
                "parent_ids": [],
                "initial_site_id": "site/demo/a",
                "protected_spans": [],
            }
        ]
    elif change == "exact_constraints":
        plan["exact_constraints"] = [
            {
                "constraint_id": "c",
                "source_event_id": "e",
                "kind": "population_at_date",
                "year": 1301,
                "site_ids": ["site/demo/a"],
                "count": 100,
                "phase": 80,
                "effect_mode": "assert",
                "count_basis": "state_at_date",
            }
        ]
    elif change == "protect_person":
        plan["historical_program"] = [
            {
                "op_id": "o",
                "source_event_id": "e",
                "kind": "protect_person",
                "person_id": "p",
                "valid": plan["timeline"],
            }
        ]
    elif change == "bridge":
        plan["boundary_ports"][0].update(mode="committed_bridge", bridge_plan_id="bridge-1")
    elif change == "procedural":
        plan["provider"]["mode"] = "procedural"
    elif change == "capability":
        plan["provider"]["capabilities"].append("synchronized_exact_v1")
    elif change == "dynamic":
        plan["settlements"][0]["foundation_year"] = 1301
    elif change == "custom":
        plan["resolved_profiles"]["ancestries"][0]["reproduction_mode"] = "custom"
    else:
        plan["historical_program"][0]["precision"] = "exact"
    response = request(worker, "prepare", plan=seal(plan))
    assert not response["ok"]
    assert response["error"]["code"] == "UnsupportedCapability", response


@pytest.mark.parametrize(
    "change",
    [
        "unknown",
        "float",
        "bool",
        "rank",
        "reference",
        "weights",
        "interval",
        "overlap",
        "seed",
        "fingerprint",
        "society_id",
        "attempts",
        "marker",
    ],
)
def test_invalid_plans_are_refused(worker, change):
    plan = example(worker.runtime_id)
    if change == "unknown":
        plan["settlements"][0]["unknown"] = 1
    elif change == "float":
        plan["settlements"][0]["x_mm"] = 1.5
    elif change == "bool":
        plan["settlements"][0]["initial_population"] = True
    elif change == "rank":
        plan["resolved_profiles"]["social_classes"][1]["provider_rank"] = 0
    elif change == "reference":
        plan["shocks"][0]["site_ids"] = ["unknown"]
    elif change == "weights":
        plan["settlements"][0]["ancestry_weights"] = {"human-example": 0}
    elif change == "interval":
        plan["shocks"][0]["valid"]["end_year"] = 1302
    elif change == "overlap":
        plan["settlements"][0]["occupation_episodes"] *= 2
    elif change == "seed":
        plan["seed"] = str(2**64)
    elif change == "society_id":
        plan["demography_society_id"] = "society/wrong"
    elif change == "attempts":
        plan["limits"]["max_attempts"] = 2
    elif change == "marker":
        plan["baseline_markers"][0]["source_event_id"] = "future-plague"
    if change != "float":
        seal(plan)
    if change == "fingerprint":
        plan["seed"] = "1"
    response = request(worker, "prepare", plan=plan)
    assert not response["ok"]
    assert response["error"]["code"] == "PopulationConstraintsUnsatisfied", response


def test_runtime_and_payload_reference_are_checked(worker):
    plan = example(worker.runtime_id)
    plan["provider"]["runtime_fingerprint"] = "another-runtime"
    response = request(worker, "prepare", plan=seal(plan))
    assert response["error"]["code"] == "ProviderVersionMismatch"
    plan = example(worker.runtime_id)
    assert request(worker, "prepare", plan=plan)["ok"]
    response = request(worker, "materialize", plan_id=plan["plan_id"], payload_ref="wrong")
    assert response["error"]["constraint"] == "payload_ref"
    assert request(worker, "query")["error"]["code"] == "UnsupportedCapability"


def test_full_example_publishes_atomic_partial_result(worker):
    plan = example(worker.runtime_id, small=False)
    assert request(worker, "prepare", plan=plan)["ok"]
    result = read_result(
        worker,
        request(
            worker, "materialize", plan_id=plan["plan_id"], payload_ref=plan["plan_fingerprint"]
        ),
    )
    assert result["format"] == PARTIAL_FORMAT
    assert result["result_type"] != "PopulationResult" and result["complete"] is False
    assert len(result["annual_censuses"]) == 200
    assert all(
        row["simulated"] + row["projected"] == row["population"]
        for row in result["annual_censuses"]
    )
    assert result["diagnostics"]["unexecuted_operation_ids"] == ["op/demo/move-1365"]
    assert "candidate_tables" in result["diagnostics"]["deferred"]
    previous = {s["site_id"]: s["initial_population"] for s in plan["settlements"]}
    for row in result["annual_censuses"]:
        assert (
            row["population"]
            == previous[row["site_id"]]
            + row["births"]
            - row["deaths"]
            + row["net_internal_migration"]
        )
        previous[row["site_id"]] = row["population"]
    for year in range(1300, 1400):
        assert (
            sum(
                row["net_internal_migration"]
                for row in result["annual_censuses"]
                if row["year"] == year
            )
            == 0
        )
    assert "seconds" not in json.dumps(result)
    assert str(worker.job_root) not in json.dumps(result)
    assert set(result["provider_manifest"]["capabilities"]) == {"historical_soft_v1"}


@pytest.mark.parametrize(
    "relative",
    [
        "../escape.json",
        "sub/../../escape.json",
        "..\\escape.json",
        "/escape.json",
        "C:\\escape.json",
        "C:escape.json",
        "data:stream",
    ],
)
def test_result_path_outside_job_directory_is_impossible(tmp_path, relative):
    with pytest.raises(ProviderError, match="inside|escapes"):
        result_path(tmp_path, relative)
    assert result_path(tmp_path, "result.json") == tmp_path / "result.json"


def test_publication_failure_has_no_canonical_result(worker, monkeypatch):
    plan = example(worker.runtime_id)
    assert request(worker, "prepare", plan=plan)["ok"]

    def fail_replace(*_args):
        raise OSError("atomic publication unavailable")

    monkeypatch.setattr(os, "replace", fail_replace)
    previous = set(worker.job_root.rglob("population-partial.json"))
    response = request(
        worker, "materialize", plan_id=plan["plan_id"], payload_ref=plan["plan_fingerprint"]
    )
    assert not response["ok"]
    assert set(worker.job_root.rglob("population-partial.json")) == previous
    assert not list(worker.job_root.rglob("*.partial"))


def test_one_job_at_a_time(worker):
    assert worker._lock.acquire(blocking=False)
    try:
        assert (
            request(worker, "prepare", plan=example(worker.runtime_id))["error"]["code"]
            == "ProviderUnavailable"
        )
    finally:
        worker._lock.release()


def test_crisis_is_applied_once(worker):
    plan = example(worker.runtime_id)
    config = translate_plan(plan)
    full = config.model_dump(mode="json")
    full.update(population_model="full", hybrid=None, initial_population=100)
    full["historical"]["limits"].update(max_core_living=1000, max_core_historical=10000)
    engine = Engine(Scenario.model_validate(full))
    engine.year = 1303
    factors = engine._events(engine.alive)
    exposed = engine.data["place"][engine.alive] == 0
    assert np.allclose(factors[1][exposed], 0.08)
    assert np.allclose(factors[1][~exposed], 0)
    assert np.allclose(factors[2][exposed], 0.7)
    assert config.events == []  # The program is not also copied into standalone events.
    program = HistoricalProgram(config)
    base = np.array([0.1, 0.1])
    m, e, _f, _, _ = program.factors(1303, [0, 1])
    assert np.allclose(mortality_probability(base, m, e), [1 - 0.9 * 0.92, 0.1])
    # Cohorts apply exactly that same combined probability, once.
    cohorts = Cohorts(config, config.settlements)
    cohorts.seed_founders(config.initial_population)
    indices = np.arange(len(cohorts.records))
    state = cohorts.initial(indices)
    survival = np.ones((len(indices), 2))
    cohorts.transition(indices, 1303, survival, state, np.zeros(len(indices)))
    columns = cohorts.columns()
    ages = 1303 - columns["birth"]
    demo = cohorts.policy(1303)[0][0]
    base = cohorts._mortality(demo, ages)
    base = np.column_stack((np.minimum(1, base * demo.male_mortality_factor), base))
    m, e, *_ = program.factors(1303, columns["place"])
    assert np.allclose(survival, 1 - mortality_probability(base, m[:, None], e[:, None]))


def test_independent_survival_composes_factors_before_additional_mortality(worker):
    plan = example(worker.runtime_id)
    second = copy.deepcopy(plan["shocks"][0])
    second["source_event_id"] = "event/z"
    second["effects"] = {"additional_mortality_bps": 2000, "mortality_multiplier_ppm": 2000000}
    plan["shocks"].append(second)
    program = HistoricalProgram(translate_plan(plan))
    m, e, *_ = program.factors(1303, [0])
    assert e[0] == pytest.approx(1 - 0.92 * 0.8)
    assert mortality_probability(np.array([0.1]), m, e)[0] == pytest.approx(1 - 0.9**2 * 0.92 * 0.8)


def test_marker_does_not_anticipate_shock(worker):
    plan = example(worker.runtime_id)
    with_shock = partial_result(plan, translate_plan(plan))
    plan["shocks"] = []
    without = partial_result(seal(plan), translate_plan(plan))
    before = lambda result: [r for r in result["annual_censuses"] if r["year"] < 1303]
    assert before(with_shock) == before(without)
    assert (
        with_shock["provider_manifest"]["core_selection"]
        == without["provider_manifest"]["core_selection"]
    )
    program = HistoricalProgram(translate_plan(plan))
    assert program.baseline("site/demo/a", 1302) == (240 * 7, 7)


def test_initial_shock_is_applied_before_first_census(worker):
    plan = example(worker.runtime_id)
    plan["shocks"][0].update(
        valid={"start_year": 1300, "end_year": 1301}, effects={"additional_mortality_bps": 10000}
    )
    plan["shocks"][0]["site_ids"] = [s["site_id"] for s in plan["settlements"]]
    result = partial_result(seal(plan), translate_plan(plan))
    first = [r for r in result["annual_censuses"] if r["year"] == 1300]
    assert sum(r["population"] for r in first) == 0
    assert sum(r["deaths"] for r in first) == 360
    assert result["diagnostics"]["initial_delta"]["deaths"] == 360
    assert result["provider_manifest"]["core_selection"]["selected_attempt"] == 0
    assert all(r["population"] == 0 for r in result["annual_censuses"])
    cohorts = Cohorts(translate_plan(plan), translate_plan(plan).settlements)
    cohorts.seed_founders(360)
    indices = np.arange(len(cohorts.records))
    survival = np.ones((len(indices), 2))
    state = cohorts.initial_program(indices, cohorts.initial(indices), survival)
    assert not state[:, :2].any()
    with pytest.raises(RuntimeError, match="already"):
        cohorts.initial_program(indices, state, survival)
    assert not cohorts.curve(0)[:, :2].any()  # Query reconstruction includes bootstrap.
    owner = type("Owner", (), {})()
    assert apply_initial_program(owner, lambda: 7) == 7
    with pytest.raises(RuntimeError):
        apply_initial_program(owner, lambda: 8)


def test_initial_program_is_not_reapplied_in_following_year(worker):
    plan = example(worker.runtime_id)
    plan["shocks"][0].update(
        valid={"start_year": 1300, "end_year": 1301}, effects={"additional_mortality_bps": 5000}
    )
    config = translate_plan(plan)
    cohorts = Cohorts(config, config.settlements)
    cohorts.seed_founders(360)
    indices = np.arange(len(cohorts.records))
    state, survival = cohorts.initial(indices), np.ones((len(indices), 2))
    state = cohorts.initial_program(indices, state, survival)
    after_initial = survival.copy()
    cohorts.transition(indices, 1301, survival, state, np.zeros(len(indices)))
    columns = cohorts.columns()
    demo = cohorts.policy(1301)[0][0]
    q = cohorts._mortality(demo, 1301 - columns["birth"])
    rates = np.column_stack((np.minimum(1, q * demo.male_mortality_factor), q))
    assert np.allclose(survival, after_initial * (1 - rates))


def test_free_core_policy_is_query_independent(worker):
    plan = example(worker.runtime_id)
    config = translate_plan(plan)
    calls = []

    def run(candidate):
        calls.append((candidate.seed, candidate.hybrid.core_target_share))
        if candidate.hybrid.core_target_share:
            raise CoreBudgetExceeded("test core ceiling", year=1301)
        return "projected"

    _, _, first = select_core_variant(plan, config, run)
    first_calls = calls.copy()
    calls.clear()
    _, _, second = select_core_variant(plan, config, run)
    assert first == second and calls == first_calls
    assert len({seed for seed, _ in calls}) == 4
    assert [share for _, share in calls] == [0.05, 0.05, 0.05, 0]
    assert first["selected_attempt"] == 3 and first["variant"] == "projected_only"
    # Real budget failures follow the same policy, and never delete core relatives.
    plan["limits"]["max_core_living"] = 0
    result = partial_result(seal(plan), translate_plan(plan))
    assert result["provider_manifest"]["core_selection"]["variant"] == "projected_only"
    assert all(row["simulated"] == 0 for row in result["annual_censuses"])


@pytest.mark.parametrize(
    "error",
    [
        MemoryError(),
        ProviderError("BudgetExceeded", "slots"),
        ProviderError("UnsupportedCapability", "biology"),
        ValueError("provider failure"),
    ],
)
def test_only_core_budget_errors_trigger_retries(worker, error):
    plan = example(worker.runtime_id)
    calls = []

    def run(candidate):
        calls.append(candidate)
        raise error

    with pytest.raises(type(error)):
        select_core_variant(plan, translate_plan(plan), run)
    assert len(calls) == 1


def test_extinct_core_is_a_valid_result_and_required_core_never_projects(worker):
    plan = example(worker.runtime_id)
    calls = []

    def success(candidate):
        calls.append(candidate)
        return "extinct"

    assert select_core_variant(plan, translate_plan(plan), success)[0] == "extinct"
    assert len(calls) == 1
    plan["core_policy"] = "required_core"
    plan["limits"]["max_attempts"] = 8
    calls.clear()

    def fail(candidate):
        calls.append(candidate)
        raise CoreBudgetExceeded("ceiling")

    with pytest.raises(CoreBudgetExceeded):
        select_core_variant(plan, translate_plan(plan), fail)
    assert len(calls) == 8
    assert all(c.hybrid.core_target_share > 0 for c in calls)


def test_mobility_graph_has_no_euclidean_fallback(worker):
    plan = example(worker.runtime_id)
    config = translate_plan(plan)
    assert HistoricalProgram(config).neighbors(1300, config.settlements, config.society)[0] == [
        (25, 1)
    ]
    plan["mobility_edges"] = []
    config = translate_plan(plan)
    assert HistoricalProgram(config).neighbors(1300, config.settlements, config.society) == {
        0: [],
        1: [],
    }


def test_schema_conditional_port_and_exact_constraint_rules():
    port_schema = SCHEMA["$defs"]["port"]
    port = json.loads(CONTRACT.joinpath("village-population-plan.example.json").read_text())[
        "boundary_ports"
    ][0]
    port["bridge_plan_id"] = "not-null"
    with pytest.raises(ProviderError):
        validate_schema(port, port_schema)
    port["mode"] = "committed_bridge"
    validate_schema(port, port_schema)


def run_worker_process(root, plans):
    requests = []
    for plan in plans:
        requests.extend(
            [
                {
                    "protocol": PROTOCOL,
                    "request_id": "prepare-" + plan["plan_id"],
                    "method": "prepare",
                    "plan": plan,
                },
                {
                    "protocol": PROTOCOL,
                    "request_id": "materialize-" + plan["plan_id"],
                    "method": "materialize",
                    "plan_id": plan["plan_id"],
                    "payload_ref": plan["plan_fingerprint"],
                },
            ]
        )
    result = subprocess.run(
        [sys.executable, "-m", "lineage_lab.chronos_bridge", "--job-root", str(root)],
        input="\n".join(json.dumps(r) for r in requests) + "\n",
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=True,
        timeout=120,
    )
    responses = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(responses) == len(requests), result.stdout
    facts = {}
    for response in responses:
        assert response["ok"], (response, result.stderr)
        if "result_ref" in response:
            raw = result_path(
                root / response["job_directory"], response["result_ref"]["relative_path"]
            ).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == response["result_ref"]["file_sha256"]
            facts[response["plan_id"]] = raw
    return facts


def test_worker_job_order_is_irrelevant(worker, tmp_path):
    a = example(worker.runtime_id)
    a["plan_id"] = "A"
    b = copy.deepcopy(a)
    b.update(plan_id="B", seed="1234567")
    seal(b)
    ab = run_worker_process(tmp_path / "ab", [a, b])
    ba = run_worker_process(tmp_path / "ba", [b, a])
    fresh_a = run_worker_process(tmp_path / "fresh-a", [a])
    fresh_b = run_worker_process(tmp_path / "fresh-b", [b])
    assert ab == ba == {**fresh_a, **fresh_b}


def test_malformed_json_does_not_kill_worker(tmp_path):
    input_lines = 'bad json\n{"protocol":"chronos-lineage-1","request_id":"a","request_id":"b"}\n'
    result = subprocess.run(
        [sys.executable, "-m", "lineage_lab.chronos_bridge", "--job-root", str(tmp_path)],
        input=input_lines,
        text=True,
        capture_output=True,
        check=True,
        timeout=60,
    )
    responses = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(responses) == 2
    assert all(r["error"]["code"] == "PopulationConstraintsUnsatisfied" for r in responses)


def test_capacity_history_reconstructs_without_query_dependent_births(worker):
    from lineage_lab.hybrid_engine import HybridEngine

    plan = example(worker.runtime_id)
    plan["limits"]["core_target_share_bps"] = 0
    plan["shocks"][0]["effects"] = {
        "capacity_multiplier_ppm": 10000,
        "fertility_multiplier_ppm": 700000,
    }
    archive = HybridEngine(translate_plan(plan)).run()
    background = archive.background
    # Replay each cohort separately, in reverse order, using the full-site context.
    counts = np.zeros(archive.config.duration + 1, np.int64)
    for index in reversed(range(len(background.records))):
        counts += background.curve(index)[:, :2].sum(axis=1)
        if background.records[index]["birth"] > archive.config.start_year:
            background.birth_plan(index)  # Reconstructed allocations must close exactly.
    expected = [
        sum(
            row["projected"]
            for row in archive._payload["integrated_censuses"]
            if row["year"] == year
        )
        for year in range(1300, 1308)
    ]
    assert counts.tolist() == expected


def test_mortality_multiplier_cannot_override_hard_lifespan(worker):
    plan = example(worker.runtime_id)
    plan["shocks"][0].update(
        valid={"start_year": 1301, "end_year": 1302}, effects={"mortality_multiplier_ppm": 0}
    )
    config = translate_plan(plan)
    cohorts = Cohorts(config, config.settlements)
    cohorts.append(0, 0, 1201, [1, 1])
    state = cohorts.initial([0])
    result = cohorts.transition([0], 1301, np.ones((1, 2)), state, np.zeros(1))
    assert not result[:, :2].any()


def test_last_public_year_and_one_year_horizon(worker):
    plan = example(worker.runtime_id)
    plan["timeline"]["end_year"] = 1301
    for site in plan["settlements"]:
        site["occupation_episodes"][0]["valid"]["end_year"] = 1301
    plan["shocks"] = []
    plan["historical_program"] = []
    plan["boundary_ports"] = []
    plan["role_demands"] = []
    plan["mobility_edges"] = []
    plan["baseline_markers"] = [m for m in plan["baseline_markers"] if m["year"] == 1300]
    assert request(worker, "prepare", plan=seal(plan))["ok"]
    result = partial_result(plan, translate_plan(plan))
    assert {row["year"] for row in result["annual_censuses"]} == {1300}
    assert sum(row["population"] for row in result["annual_censuses"]) == 360


def test_cohort_and_result_budgets_do_not_trigger_core_fallback(worker):
    plan = example(worker.runtime_id)
    plan["limits"]["max_cohorts"] = 1
    assert request(worker, "prepare", plan=seal(plan))["ok"]
    response = request(
        worker, "materialize", plan_id=plan["plan_id"], payload_ref=plan["plan_fingerprint"]
    )
    assert response["error"]["code"] == "BudgetExceeded"
    assert response["error"]["constraint"] == "max_cohorts"
    plan = example(worker.runtime_id)
    plan["limits"]["max_result_bytes"] = 1
    assert request(worker, "prepare", plan=seal(plan))["ok"]
    previous = set(worker.job_root.rglob("population-partial.json"))
    response = request(
        worker, "materialize", plan_id=plan["plan_id"], payload_ref=plan["plan_fingerprint"]
    )
    assert response["error"]["code"] == "BudgetExceeded"
    assert response["error"]["constraint"] == "max_result_bytes"
    assert set(worker.job_root.rglob("population-partial.json")) == previous
