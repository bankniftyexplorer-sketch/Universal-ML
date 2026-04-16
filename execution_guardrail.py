#!/usr/bin/env python3
"""
Execution guardrail for saved-artifact replay metrics.

Purpose:
  - capture a read-only baseline of replay metrics for base/policy execution
  - compare the current repo state against that baseline before trusting
    runtime, policy, or exit changes
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from inference_bridge import InferenceBridge
from instrument_registry import resolve_instrument_identity
from sleeve_registry import _build_1d_bundle, _build_1h_bundle, _evaluate_variant
from universal_ml_engine import get_artifact_paths

DEFAULT_BASELINE_DIR = ".execution_baselines"
TRACKED_ARTIFACT_KEYS = (
    "model",
    "features",
    "oos_proba",
    "calibrator",
    "trade_plan_models",
    "exit_surface",
    "policy_artifact",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _baseline_path(base_dir: str, name: str) -> Path:
    safe_name = name.replace("/", "_").replace(" ", "_")
    return Path(base_dir) / f"{safe_name}.json"


def _lane_list(selected: str) -> list[str]:
    return ["1H", "1D"] if selected == "all" else [selected]


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _artifact_snapshot(
    symbol_dir: str, file_prefix: str, lane: str
) -> dict[str, dict[str, Any]]:
    current_paths = get_artifact_paths(symbol_dir, file_prefix, lane)
    out: dict[str, dict[str, Any]] = {}
    for key in TRACKED_ARTIFACT_KEYS:
        path = current_paths.get(key)
        exists = bool(path and os.path.exists(path))
        out[key] = {
            "path": path,
            "exists": exists,
            "size_bytes": os.path.getsize(path) if exists else None,
            "mtime_utc": (
                datetime.fromtimestamp(os.path.getmtime(path), timezone.utc)
                .replace(microsecond=0)
                .isoformat()
                if exists
                else None
            ),
            "sha256": _sha256(path) if exists else None,
        }
    return out


def _safe_float(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _capture_symbol_lane_snapshot(
    bridge: InferenceBridge,
    outdir: str,
    symbol: str,
    lane: str,
) -> dict[str, Any]:
    lane_key = lane.upper()
    canonical_symbol = resolve_instrument_identity(
        symbol, outdir=outdir
    ).canonical_symbol
    builder = _build_1h_bundle if lane_key == "1H" else _build_1d_bundle
    bundle = builder(
        bridge,
        outdir,
        canonical_symbol,
        refresh_policy_artifact=False,
        persist_policy_artifact=False,
    )
    variants: dict[str, Any] = {}
    for variant in ("base", "policy"):
        variants[variant] = _evaluate_variant(bundle, variant=variant)

    file_prefix = canonical_symbol.lower().replace(" ", "_")
    return {
        "symbol": canonical_symbol,
        "lane": lane_key,
        "captured_at_utc": _utc_now(),
        "artifacts": _artifact_snapshot(bundle["symbol_dir"], file_prefix, lane_key),
        "variants": variants,
    }


def _metric_regressed(
    current: float | None,
    baseline: float | None,
    tolerance: float,
) -> bool:
    if current is None or baseline is None:
        return False
    return current + tolerance < baseline


def _artifact_identity_status(
    baseline_artifacts: dict[str, dict[str, Any]],
    current_artifacts: dict[str, dict[str, Any]],
) -> tuple[str, list[str]]:
    changed: list[str] = []
    for key, base_meta in baseline_artifacts.items():
        if base_meta.get("exists") and (
            base_meta.get("sha256") != current_artifacts.get(key, {}).get("sha256")
        ):
            changed.append(key)
    return ("DIFFERENT RUN", changed) if changed else ("SAME RUN", changed)


def capture_baseline(args: argparse.Namespace) -> int:
    Path(args.base_dir).mkdir(parents=True, exist_ok=True)
    baseline_path = (
        Path(args.baseline)
        if args.baseline
        else _baseline_path(
            args.base_dir,
            args.name,
        )
    )
    bridge = InferenceBridge(
        db_path=os.path.join(args.outdir, "data_vault", "ohlcv.db")
    )
    payload = {
        "captured_at_utc": _utc_now(),
        "symbols": [
            resolve_instrument_identity(symbol, outdir=args.outdir).canonical_symbol
            for symbol in args.symbols
        ],
        "lanes": {},
    }
    for symbol in payload["symbols"]:
        for lane in _lane_list(args.lane):
            key = f"{symbol}_{lane}"
            print(f"[capture] replaying {key}...")
            payload["lanes"][key] = _capture_symbol_lane_snapshot(
                bridge,
                args.outdir,
                symbol,
                lane,
            )
    baseline_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[capture] baseline saved to {baseline_path}")
    return 0


def compare_baseline(args: argparse.Namespace) -> int:
    baseline_path = (
        Path(args.baseline)
        if args.baseline
        else _baseline_path(
            args.base_dir,
            args.name,
        )
    )
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline file not found: {baseline_path}")

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    bridge = InferenceBridge(
        db_path=os.path.join(args.outdir, "data_vault", "ohlcv.db")
    )
    failures: list[str] = []
    for symbol in [
        resolve_instrument_identity(item, outdir=args.outdir).canonical_symbol
        for item in args.symbols
    ]:
        for lane in _lane_list(args.lane):
            key = f"{symbol}_{lane}"
            if key not in baseline.get("lanes", {}):
                failures.append(f"{key}: missing from baseline")
                continue
            print(f"[compare] replaying {key}...")
            current = _capture_symbol_lane_snapshot(bridge, args.outdir, symbol, lane)
            base_lane = baseline["lanes"][key]
            run_status, changed = _artifact_identity_status(
                base_lane.get("artifacts", {}),
                current.get("artifacts", {}),
            )
            if changed:
                print(
                    f"[compare] {key} artifact identity: {run_status} "
                    f"(changed: {', '.join(changed)})"
                )
            else:
                print(f"[compare] {key} artifact identity: {run_status}")

            for variant in ("base", "policy"):
                current_metrics = current["variants"].get(variant)
                base_metrics = base_lane.get("variants", {}).get(variant)
                if base_metrics is None:
                    continue
                if current_metrics is None:
                    failures.append(f"{key}/{variant}: missing current metrics")
                    continue

                print(
                    f"[compare] {key}/{variant}: "
                    f"eq={_safe_float(current_metrics.get('final_equity'))} "
                    f"pf={_safe_float(current_metrics.get('profit_factor'))} "
                    f"sh={_safe_float(current_metrics.get('sharpe'))} "
                    f"mdd={_safe_float(current_metrics.get('max_drawdown'))} "
                    f"trades={current_metrics.get('total_trades')}"
                )

                if _metric_regressed(
                    _safe_float(current_metrics.get("final_equity")),
                    _safe_float(base_metrics.get("final_equity")),
                    args.metric_tolerance,
                ):
                    failures.append(f"{key}/{variant}: final_equity regressed")
                if _metric_regressed(
                    _safe_float(current_metrics.get("profit_factor")),
                    _safe_float(base_metrics.get("profit_factor")),
                    args.metric_tolerance,
                ):
                    failures.append(f"{key}/{variant}: profit_factor regressed")
                if _metric_regressed(
                    _safe_float(current_metrics.get("sharpe")),
                    _safe_float(base_metrics.get("sharpe")),
                    args.metric_tolerance,
                ):
                    failures.append(f"{key}/{variant}: sharpe regressed")

                current_mdd = _safe_float(current_metrics.get("max_drawdown"))
                base_mdd = _safe_float(base_metrics.get("max_drawdown"))
                if (
                    current_mdd is not None
                    and base_mdd is not None
                    and current_mdd > (base_mdd + args.drawdown_tolerance)
                ):
                    failures.append(f"{key}/{variant}: max_drawdown worsened")

    if failures:
        print("[compare] FAIL")
        for item in failures:
            print(f"  - {item}")
        return 1

    print("[compare] PASS")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Saved-artifact execution guardrail")
    sub = parser.add_subparsers(dest="command", required=True)

    capture = sub.add_parser("capture", help="Capture a replay baseline")
    capture.add_argument("--symbols", nargs="+", required=True, help="Target symbols")
    capture.add_argument(
        "--outdir",
        default="/home/km/Universal-ML/",
        help="Project root / artifact root",
    )
    capture.add_argument(
        "--lane",
        choices=("1H", "1D", "all"),
        default="all",
        help="Lane to capture",
    )
    capture.add_argument(
        "--name",
        default="core_execution",
        help="Baseline file stem",
    )
    capture.add_argument(
        "--base-dir",
        default=DEFAULT_BASELINE_DIR,
        help="Directory for baseline files",
    )
    capture.add_argument(
        "--baseline",
        default=None,
        help="Optional explicit baseline output path",
    )
    capture.set_defaults(func=capture_baseline)

    compare = sub.add_parser("compare", help="Compare current replay state to baseline")
    compare.add_argument("--symbols", nargs="+", required=True, help="Target symbols")
    compare.add_argument(
        "--outdir",
        default="/home/km/Universal-ML/",
        help="Project root / artifact root",
    )
    compare.add_argument(
        "--lane",
        choices=("1H", "1D", "all"),
        default="all",
        help="Lane to compare",
    )
    compare.add_argument(
        "--name",
        default="core_execution",
        help="Baseline file stem",
    )
    compare.add_argument(
        "--base-dir",
        default=DEFAULT_BASELINE_DIR,
        help="Directory for baseline files",
    )
    compare.add_argument(
        "--baseline",
        default=None,
        help="Optional explicit baseline path",
    )
    compare.add_argument(
        "--metric-tolerance",
        type=float,
        default=1e-9,
        help="Allowed negative drift for equity/PF/Sharpe before failing",
    )
    compare.add_argument(
        "--drawdown-tolerance",
        type=float,
        default=1e-9,
        help="Allowed positive max drawdown drift before failing",
    )
    compare.set_defaults(func=compare_baseline)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
