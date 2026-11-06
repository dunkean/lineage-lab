"""Pinned PopulationPlan validation and integer-only canonical serialization."""

from __future__ import annotations

import hashlib
import json
import re
from importlib.resources import files

from .historical_program import ProviderError

CONTRACT = files("lineage_lab").joinpath("chronos_contract")
SCHEMA = json.loads(CONTRACT.joinpath("population-plan.schema.json").read_text(encoding="utf-8"))
SOCIETY_FIELDS = frozenset(
    [
        "childfree_rate_bps",
        "marriage_rate_bps",
        "founder_match_participation_bps",
        "marriage_min_age",
        "marriage_max_age",
        "max_age_gap",
        "divorce_rate_bps",
        "remarriage_delay_years",
        "kinship_depth",
        "matching_attempts",
        "status_affinity_bps",
        "status_inheritance_bps",
        "activity_inheritance_bps",
        "migration_rate_bps",
        "dependent_age",
        "residence",
        "distance_scale_mm",
        "marriage_radius_mm",
        "migration_radius_mm",
        "max_neighbors",
        "local_marriage_weight",
        "capacity_growth_ppm",
    ]
)
DEMOGRAPHY_FIELDS = frozenset(
    [
        "max_age",
        "infant_mortality_bps",
        "child_mortality_bps",
        "child_max_age",
        "adult_mortality_bps",
        "aging_coefficient_ppm",
        "aging_exponent_ppm",
        "male_mortality_factor_ppm",
        "maternal_mortality_bps",
        "male_birth_probability_bps",
        "female_infertility_rate_bps",
        "male_infertility_rate_bps",
        "fertility_peak_bps",
        "fertility_peak_age",
        "fertility_width_years",
        "fertility_min_age",
        "fertility_max_age",
        "birth_spacing_years",
    ]
)
EFFECT_FIELDS = {
    "additional_mortality_bps": ("extra_mortality", 10000),
    "mortality_multiplier_ppm": ("mortality_factor", 1000000),
    "fertility_multiplier_ppm": ("fertility_factor", 1000000),
    "migration_multiplier_ppm": ("migration_factor", 1000000),
    "capacity_multiplier_ppm": ("capacity_factor", 1000000),
}
PLAN_CAPABILITIES = {"historical_soft_v1", "candidate_slots_v1", "statistical_reservoir_v1"}


def invalid(message, path="plan", code="PopulationConstraintsUnsatisfied"):
    raise ProviderError(code, message, constraint=path)


def canonical_bytes(value):
    """§4.2: no floats, BOM, whitespace, ASCII escaping or non-JSON values."""

    def check(node):
        if node is None or type(node) in (bool, int):
            return
        if type(node) is str:
            node.encode("utf-8")  # Reject unpaired surrogates.
        elif type(node) is list:
            for item in node:
                check(item)
        elif type(node) is dict and all(type(k) is str for k in node):
            for key, item in node.items():
                check(key)
                check(item)
        else:
            invalid(
                "Canonical JSON only accepts integers, strings, booleans, null, lists and objects"
            )

    check(value)
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def fingerprint(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def plan_fingerprint(plan):
    # Lists are part of the contract: the host supplies ID-sorted lists and ordered registers.
    return fingerprint({k: v for k, v in plan.items() if k not in {"plan_id", "plan_fingerprint"}})


def validate_schema(value, schema=None, path="plan"):
    """Validate the exact keywords used by the bundled schema, without a dependency."""
    schema = SCHEMA if schema is None else schema
    if "$ref" in schema:
        target = SCHEMA
        for key in schema["$ref"].split("/")[1:]:
            target = target[key]
        validate_schema(value, target, path)

    def matches(branch):
        try:
            validate_schema(value, branch, path)
            return True
        except ProviderError:
            return False

    for key in ("anyOf", "oneOf"):
        if key in schema:
            count = sum(matches(branch) for branch in schema[key])
            if (key == "oneOf" and count != 1) or (key == "anyOf" and not count):
                invalid("Value does not match a permitted schema alternative", path)
    for branch in schema.get("allOf", []):
        validate_schema(value, branch, path)
    if "if" in schema:
        branch = schema.get("then" if matches(schema["if"]) else "else", {})
        validate_schema(value, branch, path)
    kind = schema.get("type")
    kinds = {
        "object": dict,
        "array": list,
        "integer": int,
        "string": str,
        "boolean": bool,
        "null": type(None),
    }
    if kind and type(value) is not kinds[kind]:
        invalid(f"Expected {kind}", path)
    if "const" in schema and (type(value) is not type(schema["const"]) or value != schema["const"]):
        invalid("Value differs from the schema constant", path)
    if "enum" in schema and not any(type(value) is type(v) and value == v for v in schema["enum"]):
        invalid("Value is outside the schema enumeration", path)
    if type(value) is dict:
        if len(value) < schema.get("minProperties", 0):
            invalid("Object has too few properties", path)
        for key in schema.get("required", []):
            if key not in value:
                invalid("Required field is missing", f"{path}.{key}")
        for key, item in value.items():
            if key in schema.get("properties", {}):
                validate_schema(item, schema["properties"][key], f"{path}.{key}")
            else:
                extra = schema.get("additionalProperties", True)
                if extra is False:
                    invalid("Unknown field", f"{path}.{key}")
                elif isinstance(extra, dict):
                    validate_schema(item, extra, f"{path}.{key}")
    if type(value) is list:
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get(
            "maxItems", float("inf")
        ):
            invalid("Array length is outside schema bounds", path)
        if schema.get("uniqueItems") and len({canonical_bytes(v) for v in value}) != len(value):
            invalid("Array items must be unique", path)
        if "items" in schema:
            for i, item in enumerate(value):
                validate_schema(item, schema["items"], f"{path}[{i}]")
    if type(value) is int and (
        value < schema.get("minimum", value) or value > schema.get("maximum", value)
    ):
        invalid("Integer is outside schema bounds", path)
    if type(value) is str:
        if len(value) < schema.get("minLength", 0) or len(value) > schema.get(
            "maxLength", len(value)
        ):
            invalid("String length is outside schema bounds", path)
        if "pattern" in schema and re.search(schema["pattern"], value) is None:
            invalid("String does not match the schema pattern", path)


def translate_parameters(parameters, allowed):
    from .config import Demography, Society

    unknown = parameters.keys() - allowed
    if unknown:
        invalid(f"Unknown profile parameter: {min(unknown)}", "resolved_profiles.parameters")
    result = {}
    for key, value in parameters.items():
        name, divisor = key, None
        for suffix, scale in (("_bps", 10000), ("_ppm", 1000000), ("_mm", 1000000), ("_years", 1)):
            if key.endswith(suffix):
                name, divisor = key[: -len(suffix)], scale
                break
        if key != "residence" and type(value) is not int:
            invalid("Profile numeric parameters must be integers", key)
        if key.endswith("_bps") and not 0 <= value <= 10000:
            invalid("Basis points must be between zero and 10000", key)
        if divisor is not None:
            value = value if divisor == 1 else value / divisor
        result[name] = value
    model = Society if allowed == SOCIETY_FIELDS else Demography
    model.model_validate(result)
    return result


def validate_plan(plan, runtime):
    canonical_bytes(plan)
    validate_schema(plan)
    # Never rely on object insertion order for chronology.
    start, end = plan["timeline"]["start_year"], plan["timeline"]["end_year"]
    if not start < end or end - start > 10000:
        invalid("Timeline must have positive duration of at most 10000 years", "timeline")
    for key in ("seed", "world_seed"):
        if int(plan[key]) >= 2**64:
            invalid("Seed exceeds u64", key)
    provider = plan["provider"]
    if provider["mode"] != "lineage_hybrid":
        invalid("Procedural demography requires S8", "provider.mode", "UnsupportedCapability")
    if (
        provider["algorithm_id"] != "chronos-cohort-1"
        or provider["projection_version"] != "chronos-social-projection-1"
    ):
        invalid(
            "Provider algorithm or projection version differs",
            "provider",
            "ProviderVersionMismatch",
        )
    if provider["runtime_fingerprint"] != runtime:
        invalid(
            "Provider runtime differs from the pinned plan",
            "provider.runtime_fingerprint",
            "ProviderVersionMismatch",
        )
    if set(provider["capabilities"]) - PLAN_CAPABILITIES:
        invalid(
            "Plan requires an undelivered capability",
            "provider.capabilities",
            "UnsupportedCapability",
        )
    if plan["fixed_people"] or plan["exact_constraints"]:
        invalid(
            "Fixed people and exact constraints require S6",
            "fixed_people/exact_constraints",
            "UnsupportedCapability",
        )
    for op in plan["historical_program"]:
        if op["kind"] != "move_households" or op.get("precision") == "exact":
            invalid(
                "Conditional people and exact operations require S6",
                op["op_id"],
                "UnsupportedCapability",
            )
    for port in plan["boundary_ports"]:
        if port["mode"] == "committed_bridge" or port["bridge_plan_id"] is not None:
            invalid("Committed bridges require S7", port["port_id"], "UnsupportedCapability")
    profiles = plan["resolved_profiles"]

    def indexed(rows, key):
        mapping = {row[key]: row for row in rows}
        if len(mapping) != len(rows):
            invalid("Duplicate identifier", key)
        return mapping

    sites = indexed(plan["settlements"], "site_id")
    indexed(plan["settlements"], "provider_local_id")
    ancestries = indexed(profiles["ancestries"], "id")
    kinship = indexed(profiles["kinship"], "id")
    classes = indexed(profiles["social_classes"], "id")
    external = indexed(profiles["external"], "id")
    indexed(plan["shocks"], "source_event_id")
    indexed(plan["historical_program"], "op_id")
    indexed(plan["role_demands"], "slot_id")
    ports = indexed(plan["boundary_ports"], "port_id")
    indexed(plan["mobility_edges"], "edge_id")
    if sorted(c["provider_rank"] for c in classes.values()) != list(range(len(classes))):
        invalid("Social class ranks must be distinct and contiguous from zero", "social_classes")

    def weights(values, refs=None):
        if sum(values.values()) <= 0 or (refs is not None and values.keys() - refs.keys()):
            invalid("Weights need a positive sum and known references", "weights")

    society = profiles["society"]
    translate_parameters(society["parameters"], SOCIETY_FIELDS)
    for key in ("status_weights", "core_status_weights"):
        weights(society[key], classes)
        if society[key].keys() != classes.keys():
            invalid("Status weights must cover every social class", key)
    society_id = "society/" + fingerprint(
        {
            k: society[k]
            for k in ("schema_id", "parameters", "status_weights", "core_status_weights")
        }
    )
    if society["id"] != society_id or plan["demography_society_id"] != society_id:
        invalid("Society identifier differs from resolved parameters", "demography_society_id")
    for ancestry in ancestries.values():
        if ancestry["reproduction_mode"] != "biparental":
            invalid("Reproduction mode is unsupported", ancestry["id"], "UnsupportedCapability")
        translate_parameters(ancestry["parameters"], DEMOGRAPHY_FIELDS)

    def span(value, lower=start, upper=end):
        a, b = value["start_year"], value["end_year"]
        if not lower <= a < b <= upper:
            invalid("Interval must be positive and within its owner coverage", "valid")

    initial = sum(s["initial_population"] for s in sites.values())
    if not 2 <= initial <= 20000:
        invalid("Domain initial population must be between 2 and 20000", "settlements")
    frames = {s["coordinate_frame"] for s in sites.values()}
    if len(frames) != 1:
        invalid("Domain sites must use one coordinate frame", "coordinate_frame")
    for site in sites.values():
        if site["foundation_year"] > start or any(
            e["state"] == "abandoned" for e in site["occupation_episodes"]
        ):
            invalid(
                "Dynamic settlement lifetimes require S8", site["site_id"], "UnsupportedCapability"
            )
        if site["kinship_profile_id"] not in kinship:
            invalid("Unknown kinship profile", site["site_id"])
        profile = kinship[site["kinship_profile_id"]]
        params = society["parameters"]
        if profile["dependent_age"] != params.get("dependent_age", 18) or profile[
            "min_parent_age"
        ] != params.get("marriage_min_age", 18):
            invalid(
                "Kinship profile is incompatible with the society",
                profile["id"],
                "UnsupportedCapability",
            )
        if len(site["history_culture_id"].split("/")) != 2:
            invalid("History culture must be family/variant", site["site_id"])
        weights(site["ancestry_weights"], ancestries)
        cursor = start
        for episode in site["occupation_episodes"]:
            span(episode["valid"])
            if episode["valid"]["start_year"] != cursor:
                invalid(
                    "Occupation episodes must cover the timeline without gaps or overlaps",
                    site["site_id"],
                )
            cursor = episode["valid"]["end_year"]
            weights(episode["culture_weights"])
            weights(episode["activity_weights"])
            if episode["capacity"] > 20000:
                invalid("Large sites require S8", site["site_id"], "UnsupportedCapability")
        if cursor != end:
            invalid("Occupation episodes must cover the timeline", site["site_id"])
    if sum(max(e["capacity"] for e in s["occupation_episodes"]) for s in sites.values()) > 20000:
        invalid("Domain expected capacity exceeds prototype budget", "settlements")
    markers = set()
    for marker in plan["baseline_markers"]:
        key = marker["site_id"], marker["year"]
        if key in markers or key[0] not in sites or not start <= key[1] < end:
            invalid("Invalid or duplicate baseline marker", "baseline_markers")
        markers.add(key)
    for site_id in sites:
        if (site_id, start) not in markers or (site_id, end - 1) not in markers:
            invalid("Baseline must cover first and last public year", site_id)
    for shock in plan["shocks"]:
        span(shock["valid"])
        if set(shock["site_ids"]) - sites.keys():
            invalid("Shock references an unknown site", shock["source_event_id"])
    for edge in plan["mobility_edges"]:
        span(edge["valid"])
        if edge["from_site_id"] not in sites or edge["to_site_id"] not in sites:
            invalid("Mobility edge references an unknown site", edge["edge_id"])
    for profile in external.values():
        weights(profile["ancestry_weights"], ancestries)
        weights(profile["household_size_weights"])
        if profile["adult_age_min"] > profile["adult_age_max"] or any(
            not k.isdecimal() or int(k) < 1 for k in profile["household_size_weights"]
        ):
            invalid("Invalid external household profile", profile["id"])
    for port in ports.values():
        span(port["valid"])
        if port["entry_site_id"] not in sites or port["external_profile_id"] not in external:
            invalid("Port references an unknown site or profile", port["port_id"])
        cursor = port["valid"]["start_year"]
        for episode in port["rate_episodes"]:
            span(episode["valid"], port["valid"]["start_year"], port["valid"]["end_year"])
            if episode["valid"]["start_year"] != cursor:
                invalid("Port rate episodes must be contiguous", port["port_id"])
            cursor = episode["valid"]["end_year"]
        if cursor != port["valid"]["end_year"]:
            invalid("Port rate episodes must cover the port interval", port["port_id"])
    for op in plan["historical_program"]:
        if not start <= op["year"] < end or op["origin_site_id"] not in sites:
            invalid("Operation date or origin is invalid", op["op_id"])
        if op["destination_site_id"] is not None and op["destination_site_id"] not in sites:
            invalid("Unknown operation destination site", op["op_id"])
        if op["destination_port_id"] is not None and op["destination_port_id"] not in ports:
            invalid("Unknown operation destination port", op["op_id"])
    activities = {
        k for s in sites.values() for e in s["occupation_episodes"] for k in e["activity_weights"]
    }
    for role in plan["role_demands"]:
        if role["policy"] == "required":
            invalid(
                "Required roles need S6 conditional lineages",
                role["slot_id"],
                "UnsupportedCapability",
            )
        span(role["valid"])
        if (
            role["site_id"] not in sites
            or role["min_age"] > role["max_age"]
            or set(role["status_ids"]) - classes.keys()
            or set(role["activity_ids"]) - activities
        ):
            invalid("Role has invalid bounds or references", role["slot_id"])
    limits = plan["limits"]
    if limits["core_target_share_bps"] > limits["core_max_share_bps"]:
        invalid("Core target exceeds its share ceiling", "limits")
    if plan["core_policy"] == "free_try3_then_projected" and limits["max_attempts"] != 4:
        invalid("Free core policy requires exactly four declared attempts", "limits.max_attempts")
    if len(ports) > limits["max_halo_contacts"] or initial > limits["max_candidate_slots"]:
        invalid(
            "Initial population or contacts exceed the declared budget", "limits", "BudgetExceeded"
        )
    if plan_fingerprint(plan) != plan["plan_fingerprint"]:
        invalid("Plan fingerprint differs from canonical content", "plan_fingerprint")


def translate_plan(plan):
    from .config import Scenario

    profiles, limits = plan["resolved_profiles"], plan["limits"]
    society_profile = profiles["society"]
    classes = sorted(profiles["social_classes"], key=lambda c: c["provider_rank"])
    society = translate_parameters(society_profile["parameters"], SOCIETY_FIELDS)
    society["status_weights"] = [society_profile["status_weights"][c["id"]] for c in classes]
    sites = sorted(plan["settlements"], key=lambda s: s["site_id"].encode("utf-8"))
    local_ids = {s["site_id"]: s["provider_local_id"] for s in sites}
    initial = sum(s["initial_population"] for s in sites)
    events = []
    for shock in sorted(plan["shocks"], key=lambda s: s["source_event_id"].encode("utf-8")):
        events.append(
            {
                "name": shock["source_event_id"],
                "start_year": shock["valid"]["start_year"],
                "end_year": shock["valid"]["end_year"] - 1,
                "settlements": [local_ids[s] for s in shock["site_ids"]],
                **{
                    EFFECT_FIELDS[k][0]: v / EFFECT_FIELDS[k][1]
                    for k, v in shock["effects"].items()
                },
            }
        )
    source = {
        "population_model": "hybrid",
        "seed": int(plan["seed"]),
        "compute_threads": 1,
        "start_year": plan["timeline"]["start_year"],
        "years": plan["timeline"]["end_year"] - plan["timeline"]["start_year"] - 1,
        "initial_population": initial,
        "target_population": None,
        "target_mode": "report",
        "snapshot_interval": 1,
        "capacity_mode": "fixed",
        "society": society,
        "races": [
            {
                "name": a["id"],
                "demography": translate_parameters(a["parameters"], DEMOGRAPHY_FIELDS),
            }
            for a in sorted(profiles["ancestries"], key=lambda a: a["id"].encode("utf-8"))
        ],
        "settlements": [
            {
                "id": s["provider_local_id"],
                "name": s["site_id"],
                "x": s["x_mm"] / 1000000,
                "y": s["y_mm"] / 1000000,
                "initial_weight": max(1, s["initial_population"]),
                "capacity": s["occupation_episodes"][0]["capacity"],
                "activities": s["occupation_episodes"][0]["activity_weights"],
                "races": s["ancestry_weights"],
                "metadata": {"chronos_initial_population": s["initial_population"]},
            }
            for s in sites
        ],
        "hybrid": {
            "core_target_share": limits["core_target_share_bps"] / 10000,
            "core_max_share": limits["core_max_share_bps"] / 10000,
            "core_historical_max_share": limits["core_max_share_bps"] / 10000,
            "core_status_weights": [
                society_profile["core_status_weights"][c["id"]] for c in classes
            ],
            "promotion_rules": [],
        },
        "historical": {
            "events": events,
            "baseline_markers": plan["baseline_markers"],
            "kinship_profiles": profiles["kinship"],
            "limits": limits,
            "occupation_episodes": {
                s["provider_local_id"]: s["occupation_episodes"] for s in sites
            },
            "mobility_edges": [
                {
                    "from": local_ids[e["from_site_id"]],
                    "to": local_ids[e["to_site_id"]],
                    **{k: e[k] for k in ("valid", "travel_days", "bidirectional")},
                }
                for e in plan["mobility_edges"]
            ],
        },
    }
    return Scenario.model_validate(source)
