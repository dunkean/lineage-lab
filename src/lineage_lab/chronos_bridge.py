"""Local Chronos JSONL worker: python -m lineage_lab.chronos_bridge --job-root DIR.

The stage-1 artifact is explicitly partial, never a complete PopulationResult.
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
            attempts.append({"index": index, "outcome": "selected", "seed": str(candidate.seed)})
            return (
                archive,
                candidate,
                {
                    "policy": plan["core_policy"],
                    "selected_attempt": index,
                    "variant": "projected_only" if projected else "target_core",
                    "attempts": attempts,
                },
            )
    raise CoreBudgetExceeded(
        "No core variant satisfies the declared budget",
        constraint="core_policy",
        plan_id=plan["plan_id"],
    )


def partial_result(plan, config):
    archive, selected, policy = select_core_variant(plan, config)
    censuses = []
    sites = {s["provider_local_id"]: s for s in plan["settlements"]}
    previous = {s["site_id"]: s["initial_population"] for s in sites.values()}
    for source in archive._payload["integrated_censuses"]:
        row = dict(source)
        row["site_id"] = sites[row.pop("local_id")]["site_id"]
        row["net_internal_migration"] = (
            row["population"] - previous[row["site_id"]] - row["births"] + row["deaths"]
        )
        previous[row["site_id"]] = row["population"]
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
            },
            "core_coverage": [coverage[y] for y in sorted(coverage)],
            "lineage_scope": "simulated_core_only; background_parentage_is_projected",
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


class Worker:
    def __init__(self, job_root):
        self.job_root = Path(job_root).resolve()
        self.runtime = runtime_manifest()
        self.runtime_id = fingerprint(self.runtime)
        self.prepared = {}
        self._lock = threading.Lock()

    def prepare(self, plan):
        # Defensive copy: callers cannot mutate a prepared job behind its fingerprint.
        plan = json.loads(canonical_bytes(plan))
        validate_plan(plan, self.runtime_id)
        config = translate_plan(plan)
        self.prepared[plan["plan_id"]] = plan, config
        return {
            "plan_id": plan["plan_id"],
            "payload_ref": plan["plan_fingerprint"],
            "runtime_fingerprint": self.runtime_id,
            "result_coverage": "partial",
            "result_format": PARTIAL_FORMAT,
        }

    def materialize(self, plan_id, payload_ref):
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
        result = partial_result(plan, config)
        content = canonical_bytes(result)
        if len(content) > plan["limits"]["max_result_bytes"]:
            raise ProviderError(
                "BudgetExceeded", "Result exceeds max_result_bytes", constraint="max_result_bytes"
            )
        # No caller-controlled output filename; one private directory per successful job.
        import tempfile

        self.job_root.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix="job-", dir=self.job_root))
        target = result_path(directory, "population-partial.json")
        staging = result_path(directory, "population-partial.json.partial")
        try:
            with staging.open("xb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(staging, target)
        finally:
            staging.unlink(missing_ok=True)
        return {
            "plan_id": plan_id,
            "result_fingerprint": result["result_fingerprint"],
            "job_directory": directory.relative_to(self.job_root).as_posix(),
            "result_ref": {
                "relative_path": target.relative_to(directory).as_posix(),
                "file_sha256": hashlib.sha256(content).hexdigest(),
                "bytes": len(content),
                "format": PARTIAL_FORMAT,
            },
            "metrics": {"materialize_seconds": time.perf_counter() - started},
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

                if isinstance(
                    exc, (ValidationError, ValueError, TypeError, KeyError, UnicodeError)
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
    parser = argparse.ArgumentParser(
        description="Chronos local population worker (partial S3 step 1)"
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
    for line in sys.stdin:
        try:

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
        print(json.dumps(response, ensure_ascii=False, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
