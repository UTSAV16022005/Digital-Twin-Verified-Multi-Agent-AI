"""Run a short, dataset-aware demo using the repository's evaluated artifacts."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.multidataset_agent_demo import (
    DEFAULT_TRACE_ROOT,
    EVALUATION_REPORTS,
    QUALITY_ROOT,
    print_demo,
    run_demo,
)


def _trace_name(args: argparse.Namespace, trace: dict) -> str:
    if trace["roles_skipped"] and not trace["agent_results"]:
        role = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(args.role or "unsupported"))
        return f"{args.dataset}_not_supported_{role}.json"
    if args.dataset == "cmapss":
        unit = trace["metadata"]["unit_id"]
        cycle = trace["metadata"]["cycle"]
        return f"cmapss_{args.subset.lower()}_unit{unit:03d}_cycle{cycle:03d}.json"
    if args.dataset == "nasa_turbofan_ds2":
        unit = trace["metadata"]["unit_id"]
        cycle = trace["metadata"]["cycle"]
        return f"nasa_turbofan_ds2_unit{unit:03d}_cycle{cycle:03d}.json"
    sample = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(trace["metadata"]["id"]))
    return f"phm2024_sample_{sample}.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        choices=("cmapss", "nasa_turbofan_ds2", "phm2024"),
    )
    parser.add_argument("--subset", choices=("FD001", "FD002", "FD003", "FD004"), default="FD001")
    parser.add_argument("--unit-id", type=int, help="stable official dataset unit selection")
    parser.add_argument("--cycle", type=int, default=30, help="stable observed flight-cycle selection")
    parser.add_argument("--sample-id", help="stable PHM sample ID; defaults to the lowest labeled training ID")
    parser.add_argument(
        "--role",
        choices=(
            "prognostics", "digital_twin", "recovery_planner", "faulty_classifier",
            "torque_margin_regressor", "sequence_agent", "monitoring", "diagnosis", "safety_verifier",
        ),
        help="request one role; unsupported dataset/role pairs return not_supported",
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument(
        "--evaluation-model-root",
        type=Path,
        default=QUALITY_ROOT,
        help="read-only root for evaluated model artifacts",
    )
    parser.add_argument(
        "--evaluation-report-root",
        type=Path,
        default=EVALUATION_REPORTS,
        help="read-only root for evaluated reports used by the demo",
    )
    parser.add_argument(
        "--nasa-audit-path",
        type=Path,
        help="NASA audit JSON path; defaults to this clone's processed-data sidecar",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_TRACE_ROOT)
    args = parser.parse_args()

    try:
        trace = run_demo(
            dataset=args.dataset,
            data_root=args.data_root,
            subset=args.subset,
            unit_id=args.unit_id,
            cycle=args.cycle,
            sample_id=args.sample_id,
            requested_role=args.role,
            evaluation_model_root=args.evaluation_model_root,
            evaluation_report_root=args.evaluation_report_root,
            nasa_audit_path=args.nasa_audit_path,
        )
        trace_path = args.output_dir / "traces" / _trace_name(args, trace)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        import json

        trace_path.write_text(
            json.dumps(trace, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print_demo(trace, trace_path)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print(f"Multi-dataset agent demo failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
