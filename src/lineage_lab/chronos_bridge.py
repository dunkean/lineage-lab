"""Local Chronos JSONL worker: python -m lineage_lab.chronos_bridge --job-root DIR.

Materialize publishes the complete, independently validated S3 PopulationResult.
stdout is reserved for protocol envelopes; provider output and logs use stderr.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import sys
import threading
import time
from pathlib import Path

# Set before NumPy/Numba/provider imports, including in a new reference worker.
for _name, _value in {
    "NUMBA_CPU_NAME": "generic",
    "NUMBA_CPU_FEATURES": "",
    "NUMBA_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}.items():
    os.environ[_name] = _value

from .chronos_plan import (
    CONTRACT,
    canonical_bytes,
    fingerprint,
    translate_plan,
    validate_plan,
)
from .historical_program import CoreBudgetExceeded, ProviderError

PROTOCOL = "chronos-lineage-1"
PARTIAL_FORMAT = "chronos-lineage-partial-json-1"
ERROR_CODES = frozenset(
    {
        "ProviderUnavailable",
        "ProviderVersionMismatch",
        "UnsupportedCapability",
        "PopulationConstraintsUnsatisfied",
        "ExactImpactInfeasible",
        "BoundaryInfeasible",
        "InvalidPopulationResult",
        "CoreBudgetExceeded",
        "BudgetExceeded",
        "CanonicalDependencyCycle",
    }
)


def runtime_manifest():
    import numba
    import numpy as np

    numba.config.CPU_NAME = "generic"
    numba.config.CPU_FEATURES = ""
    numba.set_num_threads(1)
    package = Path(__file__).parent
    sources = {}
    for path in sorted(package.rglob("*")):
        if path.is_file() and path.suffix in {".py", ".json", ".lock", ".yaml"}:
            sources[path.relative_to(package).as_posix()] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy": np.__version__,
        "numba": numba.__version__,
        "llvmlite": importlib.metadata.version("llvmlite"),
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in (
                "pydantic",
                "pydantic_core",
                "PyYAML",
                "annotated-types",
                "typing-inspection",
                "typing_extensions",
            )
        },
        "provider_code_sha256": fingerprint(sources),
        "requirements_lock_sha256": hashlib.sha256(
            CONTRACT.joinpath("requirements.lock").read_bytes()
        ).hexdigest(),
        "backend": "compiled",
        "system": platform.system(),
        "system_release": platform.release(),
        "architecture": platform.machine(),
        "cpu_target": "generic",
        "compute_threads": 1,
    }


def runtime_fingerprint():
    return fingerprint(runtime_manifest())


def _attempt_seed(seed, index):
    # Versioned, indexed attempts; no request IDs, paths, wall time or mutable RNG state.
    return int.from_bytes(
        hashlib.sha256(
            canonical_bytes(["chronos-lineage-core-attempt-1", str(seed), index])
        ).digest()[:8],
        "little",
    )


def select_core_variant(plan, config, run=None):
    from .hybrid_engine import HybridEngine

    run = run or (lambda scenario: HybridEngine(scenario).run())
    free = plan["core_policy"] == "free_try3_then_projected"
    attempts = []
    for index in range(4 if free else plan["limits"]["max_attempts"]):
        projected = free and index == 3
        data = config.model_dump(mode="json")
        data["seed"] = _attempt_seed(plan["seed"], index)
        if projected:
            data["hybrid"]["core_target_share"] = 0
            data["historical"]["limits"]["core_target_share_bps"] = 0
        candidate = type(config).model_validate(data)
        try:
            archive = run(candidate)
        except CoreBudgetExceeded as exc:
            attempts.append(
                {
                    "index": index,
                    "outcome": exc.code,
                    "year": exc.year,
                    "constraint": exc.constraint,
                }
            )
            if projected:
                raise
        else:
            has_core = archive.summary["core_founders_simulated"] > 0
            if not free and not has_core:
                raise ProviderError(
                    "PopulationConstraintsUnsatisfied",
                    "Required core has no founders",
                    constraint="required_core",
                    year=plan["timeline"]["start_year"],
                )
            attempts.append({"index": index, "outcome": "selected", "seed": str(candidate.seed)})
            return (
                archive,
                candidate,
                {
                    "policy": plan["core_policy"],
                    "selected_attempt": index,
                    "variant": "target_core" if has_core else "projected_only",
                    "attempts": attempts,
                },
            )
    raise CoreBudgetExceeded(
        "No core variant satisfies the declared budget",
        constraint="core_policy",
        plan_id=plan["plan_id"],
    )


def initial_effects_without_consequence(plan):
    """Report only ineffective composed bootstrap effects, per exposed site."""
    from fractions import Fraction

    year = plan["timeline"]["start_year"]
    active = [s for s in plan["shocks"]
              if s["valid"]["start_year"] <= year < s["valid"]["end_year"]]
    output = []
    for shock in active:
        if shock["valid"]["end_year"] != year + 1:
            continue
        for site in sorted(shock["site_ids"]):
            effects = {}
            for key, value in shock["effects"].items():
                if not key.endswith("_multiplier_ppm"):
                    continue
                composed = Fraction(1)
                for other in active:
                    if site in other["site_ids"]:
                        composed *= Fraction(other["effects"].get(key, 1000000), 1000000)
                ineffective = composed < 1 if key == "mortality_multiplier_ppm" else composed != 1
                if ineffective and value != 1000000:
                    effects[key] = value
            if effects:
                output.append({"source_event_id": shock["source_event_id"], "site_ids": [site],
                               "effects": effects, "reason": "initial_only; no_annual_transition"})
    return sorted(output, key=lambda r: (r["source_event_id"].encode(), r["site_ids"]))


def partial_result(plan, config):
    archive, selected, policy = select_core_variant(plan, config)
    censuses = []
    sites = {s["provider_local_id"]: s for s in plan["settlements"]}
    for source in archive._payload["integrated_censuses"]:
        row = dict(source)
        row["site_id"] = sites[row.pop("local_id")]["site_id"]
        row.update(precision="realized", coverage="partial")
        censuses.append(row)
    censuses.sort(key=lambda r: (r["site_id"].encode("utf-8"), r["year"]))
    coverage = {r["year"]: r for r in archive.summary["core_coverage"]}

    # Config has floats only after validated unit conversion. Preserve its exact
    # binary64 values as hex strings inside integer-only canonical JSON.
    def applied(node):
        if type(node) is float:
            return {"binary64": node.hex()}
        if type(node) is dict:
            return {str(k): applied(v) for k, v in node.items()}
        if type(node) is list:
            return [applied(v) for v in node]
        return node

    runtime = runtime_manifest()
    first_year = plan["timeline"]["start_year"]
    first = next(r for r in archive._payload["history"] if r["year"] == first_year)
    without_consequence = initial_effects_without_consequence(plan)
    lineage_scope = (
        "projected_only; no_simulated_lineages; background_parentage_is_projected; "
        "bounded_by_provider_coverage_and_projection_depth; no_long_hereditary_lineages"
        if policy["variant"] == "projected_only"
        else "simulated_core_only; background_parentage_is_projected"
    )
    result = {
        "format": PARTIAL_FORMAT,
        "result_type": "PartialPopulationDiagnostics",
        "coverage": "partial",
        "complete": False,
        "stage": "S3-step-1",
        "plan_id": plan["plan_id"],
        "plan_fingerprint": plan["plan_fingerprint"],
        "timeline": plan["timeline"],
        "provider_manifest": {
            "algorithm_id": plan["provider"]["algorithm_id"],
            "projection_version": plan["provider"]["projection_version"],
            "runtime": runtime,
            "runtime_fingerprint": fingerprint(runtime),
            "capabilities": ["historical_soft_v1"],
            "core_selection": policy,
            "applied_configuration": applied(selected.model_dump(mode="json")),
            "input_profiles": plan["resolved_profiles"],
            "catalogue_fingerprints": plan["catalogue_fingerprints"],
            "external_reference_validation": "host_owned",
        },
        "annual_censuses": censuses,
        "diagnostics": {
            "initial_delta": {
                "year": first_year,
                "births": 0,
                "deaths": first["deaths"],
                "pre_shock_population": config.initial_population,
                "population": first["population"],
                "effects_without_consequence": without_consequence,
            },
            "core_coverage": [coverage[y] for y in sorted(coverage)],
            "lineage_scope": lineage_scope,
            "regulation": "disabled_pending_site_regulator",
            "deferred": [
                "candidate_tables",
                "residence_registry",
                "households",
                "founder_assignments",
                "statistical_ports",
                "move_households",
                "site_regulator",
                "complete_export",
            ],
            "unexecuted_operation_ids": [op["op_id"] for op in plan["historical_program"]],
            "unexecuted_port_ids": [p["port_id"] for p in plan["boundary_ports"]],
            "role_demands": "planning_only; no_office_holder_selected",
        },
    }
    result["result_fingerprint"] = fingerprint(result)
    return result


def result_path(job_directory, relative_path):
    root = Path(job_directory).resolve()
    # Also reject Windows path syntax on a POSIX host and vice versa.
    from pathlib import PureWindowsPath

    relative = Path(relative_path)
    if (
        relative.is_absolute()
        or PureWindowsPath(relative_path).drive
        or ":" in relative_path
        or ".." in relative.parts
        or ".." in PureWindowsPath(relative_path).parts
    ):
        raise ProviderError(
            "InvalidPopulationResult",
            "Result path must stay inside the job directory",
            constraint="result_ref.relative_path",
        )
    target = (root / relative).resolve()
    if target == root or not target.is_relative_to(root):
        raise ProviderError(
            "InvalidPopulationResult",
            "Result path escapes the job directory",
            constraint="result_ref.relative_path",
        )
    return target


def complete_result(plan, config):
    from .integrated_population import IntegratedPopulation
    from .population_export import RESULT_FORMAT, validate_population_result
    from .registry_contract import (
        FERTILITY_VERSION,
        MORTALITY_VERSION,
        REGISTRY_VERSION,
        REGULATOR_VERSION,
        REPAIRING_VERSION,
    )

    archive, selected, policy = select_core_variant(
        plan, config, run=lambda candidate: IntegratedPopulation(candidate).run()
    )
    runtime = runtime_manifest()
    first_year = plan["timeline"]["start_year"]
    first = [r for r in archive.censuses if r["year"] == first_year]
    result = {"format": RESULT_FORMAT, "result_type": "PopulationResult", "complete": True,
              "coverage": "complete", "stage": "S3-step-2", "plan_id": plan["plan_id"],
              "plan_fingerprint": plan["plan_fingerprint"], "timeline": plan["timeline"],
              "provider_manifest": {
                  "algorithm_id": plan["provider"]["algorithm_id"],
                  "projection_version": plan["provider"]["projection_version"],
                  "registry_version": REGISTRY_VERSION, "mortality_rule_version": MORTALITY_VERSION,
                  "fertility_rule_version": FERTILITY_VERSION, "regulator_version": REGULATOR_VERSION,
                  "repairing_rule_version": REPAIRING_VERSION,
                  "runtime": runtime, "runtime_fingerprint": fingerprint(runtime),
                  "core_selection": policy, "capabilities": sorted(plan["provider"]["capabilities"]),
                  "input_profiles": plan["resolved_profiles"],
                  "catalogue_fingerprints": plan["catalogue_fingerprints"],
                  "external_reference_validation": "host_owned",
                  "selected_seed": str(selected.seed), "core_stream": "independent_precomputed"},
              **archive.tables(), "annual_censuses": archive.censuses,
              "cohort_residence_counts": archive.cohort_counts,
              "diagnostics": {**archive.diagnostics(), "lineage_scope":
                              "simulated_core_only; background_parentage_is_projected" if
                              policy["variant"] == "target_core" else
                              "projected_only; no_simulated_lineages; no_long_hereditary_lineages",
                              "initial_delta": {"year": first_year, "births": 0,
                                                "deaths": sum(r["deaths"] for r in first),
                                                "pre_shock_population": config.initial_population,
                                                "population": sum(r["population"] for r in first),
                                                "effects_without_consequence": initial_effects_without_consequence(plan)}}}
    result["validation_proof"] = validate_population_result(result, plan["limits"])
    return result


class Worker:
    def __init__(self, job_root):
        self.job_root = Path(job_root).resolve()
        self.runtime = runtime_manifest()
        self.runtime_id = fingerprint(self.runtime)
        self.prepared = {}
        self._lock = threading.Lock()

    def prepare(self, plan):
        from .population_export import RESULT_FORMAT
        # Defensive copy: callers cannot mutate a prepared job behind its fingerprint.
        plan = json.loads(canonical_bytes(plan))
        validate_plan(plan, self.runtime_id)
        config = translate_plan(plan)
        self.prepared[plan["plan_id"]] = plan, config
        return {
            "plan_id": plan["plan_id"],
            "payload_ref": plan["plan_fingerprint"],
            "runtime_fingerprint": self.runtime_id,
            "result_coverage": "complete",
            "result_format": RESULT_FORMAT,
        }

    def materialize(self, plan_id, payload_ref):
        from .population_export import RESULT_FORMAT, compact_result
        if type(plan_id) is not str or not plan_id:
            raise ProviderError(
                "PopulationConstraintsUnsatisfied", "A nonempty plan_id is required",
                constraint="plan_id",
            )
        if plan_id not in self.prepared:
            raise ProviderError(
                "PopulationConstraintsUnsatisfied",
                "Plan has not been prepared",
                constraint="plan_id",
            )
        plan, config = self.prepared[plan_id]
        if payload_ref != plan["plan_fingerprint"]:
            raise ProviderError(
                "PopulationConstraintsUnsatisfied",
                "Materialize must reference the prepared fingerprint",
                constraint="payload_ref",
            )
        started = time.perf_counter()
        # Each job uses local generators; reset module RNGs as an extra isolation boundary.
        import numba
        import numpy as np

        random.seed(int(plan["seed"]))
        np.random.seed(int(plan["seed"]) % 2**32)
        numba.set_num_threads(1)
        result = complete_result(plan, config)
        wire = compact_result(result, plan["domain_id"])
        wire["result_fingerprint"] = fingerprint(wire)
        content = canonical_bytes(wire)
        if len(content) > plan["limits"]["max_result_bytes"]:
            raise ProviderError(
                "BudgetExceeded", "Result exceeds max_result_bytes", constraint="max_result_bytes"
            )
        # No caller-controlled output filename; one private directory per successful job.
        import tempfile

        self.job_root.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix="job-", dir=self.job_root))
        target = result_path(directory, "population-result.json")
        staging = result_path(directory, "population-result.json.partial")
        try:
            with staging.open("xb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(staging, target)
        except BaseException:
            staging.unlink(missing_ok=True)
            directory.rmdir()
            raise
        return {
            "plan_id": plan_id,
            "result_fingerprint": wire["result_fingerprint"],
            "job_directory": directory.relative_to(self.job_root).as_posix(),
            "result_ref": {
                "relative_path": target.relative_to(directory).as_posix(),
                "file_sha256": hashlib.sha256(content).hexdigest(),
                "bytes": len(content),
                "format": RESULT_FORMAT,
            },
            "metrics": {"materialize_seconds": time.perf_counter() - started,
                        "candidate_slots": len(result["candidate_table"]["origin_id"]),
                        "core_selection": result["provider_manifest"]["core_selection"]},
        }

    def handle(self, request):
        request_id = request.get("request_id") if isinstance(request, dict) else None
        plan_id = request.get("plan_id") if isinstance(request, dict) else None
        response = {
            "protocol": PROTOCOL,
            "request_id": request_id,
            "plan_id": plan_id,
            "result_fingerprint": None,
        }
        try:
            if not isinstance(request, dict) or request.get("protocol") != PROTOCOL:
                raise ProviderError(
                    "ProviderVersionMismatch", "Expected chronos-lineage-1 protocol"
                )
            if not isinstance(request_id, str) or not request_id:
                raise ProviderError(
                    "PopulationConstraintsUnsatisfied", "A nonempty request_id is required"
                )
            try:
                request_id.encode("utf-8")
            except UnicodeError:
                response["request_id"] = None
                raise ProviderError(
                    "PopulationConstraintsUnsatisfied", "request_id must be valid UTF-8",
                    constraint="request_id",
                ) from None
            if not self._lock.acquire(blocking=False):
                raise ProviderError("ProviderUnavailable", "Worker is already running a job")
            try:
                if request.get("method") == "prepare":
                    plan = request.get("plan")
                    if not isinstance(plan, dict):
                        raise ProviderError(
                            "PopulationConstraintsUnsatisfied", "prepare requires a complete plan"
                        )
                    plan_id = plan.get("plan_id")
                    response["plan_id"] = plan_id
                    if request.get("plan_id", plan_id) != plan_id:
                        raise ProviderError(
                            "PopulationConstraintsUnsatisfied", "Envelope and plan IDs differ"
                        )
                    response["result"] = self.prepare(plan)
                elif request.get("method") == "materialize":
                    response.update(self.materialize(plan_id, request.get("payload_ref")))
                else:
                    raise ProviderError(
                        "UnsupportedCapability", "Only prepare and materialize are implemented"
                    )
            finally:
                self._lock.release()
        except Exception as exc:  # noqa: BLE001 - protocol boundary must return typed failures
            if not isinstance(exc, ProviderError):
                from pydantic import ValidationError

                if (
                    isinstance(request, dict)
                    and request.get("method") == "prepare"
                    and isinstance(
                        exc, (ValidationError, ValueError, TypeError, KeyError, UnicodeError)
                    )
                ):
                    exc = ProviderError(
                        "PopulationConstraintsUnsatisfied",
                        "Plan or request validation failed",
                        constraint="validation",
                    )
                elif isinstance(exc, MemoryError):
                    exc = ProviderError(
                        "BudgetExceeded", "Provider memory allocation failed", constraint="memory"
                    )
                else:
                    print(f"Worker failure: {type(exc).__name__}", file=sys.stderr)
                    exc = ProviderError("ProviderUnavailable", "Provider execution failed")
            exc.plan_id = plan_id
            exc.owner = exc.owner or plan_id
            response.update(ok=False, error=exc.as_dict())
        else:
            response["ok"] = True
        return response


def main(argv=None):
    # The wire format is strict UTF-8, including on Windows with a legacy locale
    # or PYTHONIOENCODING override. Parent-process pipe settings do not control it.
    sys.stdout.reconfigure(encoding="utf-8", errors="strict")
    parser = argparse.ArgumentParser(
        description="Chronos local population worker (complete S3 registry)"
    )
    parser.add_argument("--job-root", type=Path, help="Private output root owned by the host")
    parser.add_argument(
        "--runtime", action="store_true", help="Print the runtime manifest and fingerprint"
    )
    args = parser.parse_args(argv)
    if args.runtime:
        manifest = runtime_manifest()
        print(
            canonical_bytes(
                {"runtime": manifest, "runtime_fingerprint": fingerprint(manifest)}
            ).decode()
        )
        return 0
    if args.job_root is None:
        parser.error("--job-root is required when serving JSONL")
    # Provider chatter cannot corrupt the protocol, including during imports/JIT.
    from contextlib import redirect_stdout

    with redirect_stdout(sys.stderr):
        worker = Worker(args.job_root)
    for raw_line in sys.stdin.buffer:
        try:
            line = raw_line.decode("utf-8", errors="strict")

            def unique_pairs(pairs):
                result = {}
                for key, value in pairs:
                    if key in result:
                        raise ValueError("Duplicate JSON key")
                    result[key] = value
                return result

            request = json.loads(line, object_pairs_hook=unique_pairs)
        except (ValueError, UnicodeError):
            response = {
                "protocol": PROTOCOL,
                "request_id": None,
                "plan_id": None,
                "result_fingerprint": None,
                "ok": False,
                "error": ProviderError(
                    "PopulationConstraintsUnsatisfied", "Invalid JSON request"
                ).as_dict(),
            }
        else:
            with redirect_stdout(sys.stderr):
                response = worker.handle(request)
        # ASCII escaping here also protects error envelopes echoing malformed IDs.
        print(json.dumps(response, ensure_ascii=True, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
