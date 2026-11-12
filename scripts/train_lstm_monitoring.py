#!/usr/bin/env python3
"""
End-to-end training + evaluation of the FD001 LSTM Autoencoder Monitoring Agent.

Pipeline:
  1. parse train/test/RUL files (26 columns)
  2. split training engines into fit, calibration and evaluation groups by unit
  3. drop constant features (training engines only)
  4. fit StandardScaler on healthy training rows only
  5. build 30-cycle stride-1 windows; train AE on healthy windows only
  6. early-stop/calibrate on calibration engines; evaluate on separate engines
  7. persistent alerts + per-feature attribution
  8. evaluate validation engines (run-to-failure) and official test engines
  9. save model checkpoint, JSON report and per-window event CSVs
 10. compare directly with PCA reconstruction baseline under the same protocol

This is anomaly MONITORING, not fault diagnosis.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import io
import json
import logging
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.lstm_monitoring import (
    CANDIDATE_FEATURES, CYCLE_COL, UNIT_COL, TORCH_AVAILABLE, TORCH_IMPORT_ERROR, LSTMAutoencoder,
    LSTMMonitoringAgent, MonitoringConfig, anomaly_scores, compute_threshold,
    evaluate_fleet, find_constant_features, fit_scaler, healthy_window_mask,
    load_cmapss_file, load_rul_file, make_windows, reconstruction_errors,
    scale_frame, select_device, set_seed, split_units, train_autoencoder,
)
from src.monitoring import PCAMonitoringAgent, evaluate_alert_events

DATA_URL = "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
LOGGER = logging.getLogger("train_lstm_monitoring")


def ensure_dataset(data_dir: Path, subset: str) -> None:
    """Download and extract the required files if they are not present."""
    needed = [f"train_{subset}.txt", f"test_{subset}.txt", f"RUL_{subset}.txt"]
    if all((data_dir / n).exists() for n in needed):
        return
    data_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("downloading %s", DATA_URL)
    try:
        with urllib.request.urlopen(DATA_URL, timeout=120) as resp:
            outer = zipfile.ZipFile(io.BytesIO(resp.read()))
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"Download failed ({exc}). Place {', '.join(needed)} in {data_dir} and re-run.") from exc
    inner_name = next(n for n in outer.namelist() if n.endswith("CMAPSSData.zip"))
    inner = zipfile.ZipFile(io.BytesIO(outer.read(inner_name)))
    for n in needed:
        member = next(m for m in inner.namelist() if m.endswith(n))
        (data_dir / n).write_bytes(inner.read(member))


def to_builtin(obj):
    """Make numpy/pandas scalars JSON serialisable."""
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def parse_args() -> argparse.Namespace:
    d = MonitoringConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw" / "cmapss")
    p.add_argument("--subset", default="FD001")
    p.add_argument("--model-dir", type=Path, default=ROOT / "models")
    p.add_argument("--report-dir", type=Path, default=ROOT / "reports" / "generated")
    p.add_argument("--healthy-cycles", type=int, default=d.healthy_cycles)
    p.add_argument("--window-size", type=int, default=d.window_size)
    p.add_argument("--stride", type=int, default=d.stride)
    p.add_argument("--persistence", type=int, default=d.persistence)
    p.add_argument("--degraded-rul", type=int, default=d.degraded_rul)
    p.add_argument("--max-epochs", type=int, default=25)
    p.add_argument("--patience", type=int, default=d.patience)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--learning-rate", type=float, default=d.learning_rate)
    p.add_argument("--threshold-percentile", type=float, default=d.threshold_percentile)
    p.add_argument("--seed", type=int, default=d.seed)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    cfg = MonitoringConfig(
        window_size=args.window_size, stride=args.stride, healthy_cycles=args.healthy_cycles,
        seed=args.seed, learning_rate=args.learning_rate, batch_size=args.batch_size,
        max_epochs=args.max_epochs, patience=args.patience, threshold_percentile=args.threshold_percentile,
        persistence=args.persistence, degraded_rul=args.degraded_rul,
    )
    if cfg.healthy_cycles < cfg.window_size:
        raise SystemExit("healthy_cycles must be >= window_size to produce healthy windows")
    t0 = time.time()
    set_seed(cfg.seed)
    device = select_device()
    backend_info = "PyTorch" if TORCH_AVAILABLE else "NumPy-Vectorized-Engine"
    LOGGER.info("device: %s | backend: %s", device, backend_info)
    if not TORCH_AVAILABLE:
        LOGGER.warning(
            "PyTorch unavailable; using the NumPy LSTM fallback. Import diagnostic: %s",
            TORCH_IMPORT_ERROR,
        )

    # ---- 1-2. Data ---------------------------------------------------------
    ensure_dataset(args.data_dir, args.subset)
    train_df = load_cmapss_file(args.data_dir / f"train_{args.subset}.txt")
    test_df = load_cmapss_file(args.data_dir / f"test_{args.subset}.txt")
    rul = load_rul_file(args.data_dir / f"RUL_{args.subset}.txt")
    LOGGER.info("train rows=%d units=%d | test rows=%d units=%d", len(train_df), train_df[UNIT_COL].nunique(),
                len(test_df), test_df[UNIT_COL].nunique())

    # ---- 3. Unit-level split (no random-row split) -------------------------
    train_units, val_units = split_units(train_df, cfg.val_fraction, cfg.seed)
    pool = train_df[train_df[UNIT_COL].isin(train_units)]
    fit_units, calibration_units = split_units(pool, 0.10, cfg.seed + 1)
    tr = pool[pool[UNIT_COL].isin(fit_units)]
    calibration = pool[pool[UNIT_COL].isin(calibration_units)]
    va = train_df[train_df[UNIT_COL].isin(val_units)]

    # ---- 4. Constant features from TRAINING engines only -------------------
    features, dropped = find_constant_features(tr, cfg.constant_std_tol)
    LOGGER.info("kept %d features, dropped constant: %s", len(features), dropped)

    # ---- 5. Scaler fitted on HEALTHY TRAINING rows only --------------------
    healthy_rows = tr[tr[CYCLE_COL] <= cfg.healthy_cycles]
    scaler = fit_scaler(healthy_rows[features].to_numpy(dtype=np.float64))
    LOGGER.info("scaler backend: %s (fit on %d healthy training rows)", scaler.backend, len(healthy_rows))

    # ---- 6. Windows: healthy-only for training and early stopping ----------
    tr_win, tr_meta = make_windows(scale_frame(tr, features, scaler), features, cfg.window_size, cfg.stride)
    cal_win, cal_meta = make_windows(
        scale_frame(calibration, features, scaler), features, cfg.window_size, cfg.stride
    )
    va_win, va_meta = make_windows(scale_frame(va, features, scaler), features, cfg.window_size, cfg.stride)
    tr_h = tr_win[healthy_window_mask(tr_meta, cfg.healthy_cycles)]
    cal_h = cal_win[healthy_window_mask(cal_meta, cfg.healthy_cycles)]
    LOGGER.info("healthy windows: fit=%d  calibration(early-stopping)=%d", len(tr_h), len(cal_h))
    if len(tr_h) == 0:
        raise SystemExit("No healthy fit windows; increase healthy_cycles or reduce window_size.")
    if len(cal_h) == 0:
        raise SystemExit("No healthy calibration windows; reduce window_size or adjust the engine split.")

    # Build model (PyTorch or pure-NumPy equivalent)
    if TORCH_AVAILABLE:
        model = LSTMAutoencoder(len(features), cfg.hidden_size, cfg.latent_size, cfg.num_layers)
    else:
        from src.lstm_monitoring import NumpyLSTMAutoencoder
        model = NumpyLSTMAutoencoder(len(features), cfg.hidden_size, cfg.latent_size, cfg.seed)

    training = train_autoencoder(model, tr_h, cal_h, cfg, device)

    # ---- 7. Threshold calibrated on a dedicated held-out engine group -------
    healthy_calibration_scores = anomaly_scores(reconstruction_errors(model, cal_h, device))
    healthy_train_scores = anomaly_scores(reconstruction_errors(model, tr_h, device))
    threshold_source = healthy_calibration_scores
    threshold = compute_threshold(threshold_source, cfg.threshold_percentile)
    LOGGER.info("threshold (p%.0f healthy MSE) = %.6f", cfg.threshold_percentile, threshold)

    agent = LSTMMonitoringAgent(model, scaler, features, threshold, cfg, device)

    # ---- 8-9. Evaluate Fleet -------------------------------------------------
    val_failure = va.groupby(UNIT_COL)[CYCLE_COL].max().astype(int).to_dict()
    test_last = test_df.groupby(UNIT_COL)[CYCLE_COL].max().astype(int)
    test_failure = (test_last + rul.reindex(test_last.index)).astype(int).to_dict()

    val_events, val_engines, val_fleet = evaluate_fleet(agent.monitor(va), val_failure, cfg, features, "validation")
    test_events, test_engines, test_fleet = evaluate_fleet(agent.monitor(test_df), test_failure, cfg, features, "test")

    # ---- Direct PCA comparison on identical engine groups ------------------
    pca_agent = PCAMonitoringAgent.fit(
        tr,
        features,
        healthy_cycles=cfg.healthy_cycles,
        threshold_quantile=cfg.threshold_percentile / 100.0,
    )
    pca_val_scored = pca_agent.score(va, persistence=cfg.persistence)
    pca_test_scored = pca_agent.score(test_df, persistence=cfg.persistence)
    pca_val_metrics, _ = evaluate_alert_events(
        pca_val_scored,
        va.groupby(UNIT_COL)[CYCLE_COL].max(),
        cfg.healthy_cycles,
        cfg.degraded_rul,
    )
    pca_test_failure_cycles = test_last + rul.reindex(test_last.index)
    pca_test_metrics, _ = evaluate_alert_events(
        pca_test_scored,
        pca_test_failure_cycles,
        cfg.healthy_cycles,
        cfg.degraded_rul,
    )
    pca_comparison = {
        "evaluation_protocol": "same fit, validation and official test engine groups; same persistence value",
        "pca_fit_engines": len(fit_units),
        "lstm_fit_engines": len(fit_units),
        "validation": {
            "pca_detection_rate": pca_val_metrics["detection_rate"],
            "lstm_detection_rate": val_fleet["engine_detection_rate"],
            "pca_healthy_false_alarm_rate": pca_val_metrics["healthy_false_alarm_rate"],
            "lstm_healthy_false_alarm_rate": val_fleet["pooled_healthy_false_alarm_rate"],
            "pca_median_lead_time_cycles": pca_val_metrics["median_lead_time_cycles"],
            "lstm_median_lead_time_cycles": val_fleet["median_lead_time"],
        },
        "test": {
            "pca_detection_rate": pca_test_metrics["detection_rate"],
            "lstm_detection_rate": test_fleet["engine_detection_rate"],
            "pca_healthy_false_alarm_rate": pca_test_metrics["healthy_false_alarm_rate"],
            "lstm_healthy_false_alarm_rate": test_fleet["pooled_healthy_false_alarm_rate"],
            "pca_median_lead_time_cycles": pca_test_metrics["median_lead_time_cycles"],
            "lstm_median_lead_time_cycles": test_fleet["median_lead_time"],
        },
        "metric_alignment_note": (
            "PCA scores every telemetry row and LSTM scores the final cycle of each window; "
            "engine-level detection/lead time and healthy false-alarm rate are the comparable summaries."
        ),
    }

    example = None
    alerts = val_events[val_events["confirmed_alert"] & ~val_events["is_healthy_window"]]
    if len(alerts):
        example = agent.explain(alerts.iloc[0])
        LOGGER.info("example explanation: %s", example)

    # ---- 10. Save Artifacts --------------------------------------------------
    sub = args.subset.lower()
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.model_dir / f"{sub}_lstm_monitoring.pt"
    agent.save(model_path, extra={
        "fit_units": fit_units,
        "calibration_units": calibration_units,
        "validation_units": val_units,
        "dropped_features": dropped,
        "torch_import_error": TORCH_IMPORT_ERROR,
    })
    val_csv = args.report_dir / f"{sub}_lstm_monitoring_validation_events.csv"
    test_csv = args.report_dir / f"{sub}_lstm_monitoring_test_events.csv"
    val_events.to_csv(val_csv, index=False, float_format="%.6g")
    test_events.to_csv(test_csv, index=False, float_format="%.6g")

    report = {
        "model": "LSTM autoencoder anomaly monitor (NOT fault diagnosis)",
        "subset": args.subset,
        "device": str(device),
        "backend": backend_info,
        "config": cfg.to_dict(),
        "data": {
            "fit_units": fit_units, "calibration_units": calibration_units,
            "validation_units": val_units, "n_test_units": int(test_df[UNIT_COL].nunique()),
            "features_kept": features, "features_dropped_constant": dropped,
            "candidate_features": CANDIDATE_FEATURES, "scaler_backend": scaler.backend,
            "n_healthy_fit_windows": int(len(tr_h)), "n_healthy_calibration_windows": int(len(cal_h)),
        },
        "training": training,
        "threshold": {"value": threshold, "percentile": cfg.threshold_percentile,
                      "calibration_source": "dedicated held-out calibration engines",
                      "healthy_calibration_score_mean": float(healthy_calibration_scores.mean()),
                      "healthy_calibration_score_median": float(np.median(healthy_calibration_scores)),
                      "healthy_fit_score_mean": float(healthy_train_scores.mean()),
                      "healthy_fit_score_median": float(np.median(healthy_train_scores))},
        "runtime_environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": importlib.metadata.version("pandas"),
            "scikit_learn": importlib.metadata.version("scikit-learn"),
            "torch_available": TORCH_AVAILABLE,
            "torch_import_error": TORCH_IMPORT_ERROR,
        },
        "metric_definitions": {
            "healthy_window": f"window end cycle <= healthy_cycles ({cfg.healthy_cycles})",
            "degraded_window": f"RUL at window end <= degraded_rul ({cfg.degraded_rul})",
            "detection_rate": "share of degraded windows with a confirmed (persistent) alert",
            "healthy_false_alarm_rate": "share of healthy windows with a confirmed alert",
            "first_alert_cycle": "end cycle of first confirmed alert after the healthy period",
            "lead_time": "failure_cycle - first_alert_cycle",
            "engine_detection_rate": "engines with >=1 degraded window that raised a confirmed alert",
            "test_failure_cycle": "last observed cycle + official RUL",
        },
        "validation": {"fleet": val_fleet, "per_engine": val_engines},
        "test": {"fleet": test_fleet, "per_engine": test_engines},
        "pca_baseline_comparison": pca_comparison,
        "example_explanation": example,
        "outputs": {"model": str(model_path.relative_to(ROOT)), "validation_events": str(val_csv.relative_to(ROOT)),
                    "test_events": str(test_csv.relative_to(ROOT))},
        "runtime_seconds": round(time.time() - t0, 1),
    }
    report_path = args.report_dir / f"{sub}_lstm_monitoring.json"
    report_path.write_text(json.dumps(to_builtin(report), indent=2))

    summary_keys = ["n_engines", "n_engines_with_degraded_windows", "engine_detection_rate", "mean_detection_rate",
                    "pooled_window_detection_rate", "mean_healthy_false_alarm_rate", "pooled_healthy_false_alarm_rate",
                    "precision", "recall", "f1",
                    "n_engines_with_healthy_false_alarm", "n_engines_alerted", "median_lead_time", "mean_lead_time",
                    "isolated_spikes_suppressed"]
    print("\n=== VALIDATION ===\n" + json.dumps({k: val_fleet.get(k) for k in summary_keys}, indent=2))
    print("=== TEST ===\n" + json.dumps({k: test_fleet.get(k) for k in summary_keys}, indent=2))
    if pca_comparison:
        print("=== COMPARISON TO PCA BASELINE ===\n" + json.dumps(pca_comparison, indent=2))
    print(f"saved: {model_path}, {report_path}, {val_csv}, {test_csv}  ({report['runtime_seconds']} s)")


if __name__ == "__main__":
    main()
