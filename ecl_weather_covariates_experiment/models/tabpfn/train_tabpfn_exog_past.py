"""
TabPFN past-exogenous baseline for the independent ECL weather experiment.

Active fair setting:
- Uses past load lag/rolling features.
- Uses past shared Lisbon weather/calendar lag/rolling features.
- Does not use future observed weather from the test period (persisted from last val row).
- Uses deterministic future calendar (hour, day-of-week, month) which is known in advance.

This is an exploratory foundation-model baseline framed as tabular supervised
forecasting, not as a traditional trained time-series model.
"""

from __future__ import annotations

import argparse
import os
import ssl
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[2] / "outputs" / ".matplotlib"),
)

# macOS SSL fix — patch before any network call so HuggingFace downloads work.
try:
    import certifi
    ssl._create_default_https_context = lambda: ssl.create_default_context(
        cafile=certifi.where()
    )
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except ImportError:
    pass

import matplotlib  # noqa: E402
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR.parent
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

from common.common import (  # noqa: E402
    CALENDAR_COLUMNS,
    FIGURES_DIR,
    HORIZONS,
    RAW_RESULTS_DIR,
    WEATHER_COLUMNS,
    compute_metrics,
    dry_run_report,
    ensure_output_dirs,
    load_and_split,
    timestamp_index,
    select_clients,
    series_values,
)


RESULTS_CSV = RAW_RESULTS_DIR / "tabpfn_exog_past_weather_results.csv"
PROGRESS_LOG = RAW_RESULTS_DIR / "tabpfn_exog_past_weather_progress.log"
SAMPLE_FORECAST_PNG = FIGURES_DIR / "tabpfn_exog_past_weather_forecast_sample.png"
ERROR_BY_HORIZON_PNG = FIGURES_DIR / "tabpfn_exog_past_weather_error_by_horizon.png"
CONTEXT_ROWS = 1000
LAG_FEATURES = [1, 2, 3, 6, 12, 24, 48, 168]
ROLLING_WINDOWS = [24, 168]
EXOG_LAGS = [1, 24, 168]
EXOG_ROLLING_WINDOWS = [24, 168]
PAST_EXOG_COLUMNS = WEATHER_COLUMNS + CALENDAR_COLUMNS


def log_progress(message: str) -> None:
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line)
    PROGRESS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with PROGRESS_LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def build_feature_frame(
    target_scaled: np.ndarray,
    timestamps: pd.DatetimeIndex,
    exog_scaled: np.ndarray,
) -> pd.DataFrame:
    target = pd.Series(target_scaled, index=timestamps)
    df = pd.DataFrame(index=timestamps)

    for lag in LAG_FEATURES:
        df[f"lag_t{lag}"] = target.shift(lag)
    for window in ROLLING_WINDOWS:
        df[f"roll_{window}"] = target.shift(1).rolling(window=window, min_periods=1).mean()

    exog_df = pd.DataFrame(exog_scaled, index=timestamps, columns=PAST_EXOG_COLUMNS)
    for col in PAST_EXOG_COLUMNS:
        for lag in EXOG_LAGS:
            df[f"exog_{col}_lag{lag}"] = exog_df[col].shift(lag)
        for window in EXOG_ROLLING_WINDOWS:
            df[f"exog_{col}_roll{window}"] = (
                exog_df[col].shift(1).rolling(window=window, min_periods=1).mean()
            )

    df["target"] = target_scaled
    return df.dropna()


def build_next_row(
    target_history: list[float],
    exog_history: list[np.ndarray],
    feature_cols: list[str],
) -> pd.DataFrame:
    target_arr = np.asarray(target_history, dtype=float)
    exog_arr = np.asarray(exog_history, dtype=float)
    row: dict[str, float] = {}

    for lag in LAG_FEATURES:
        row[f"lag_t{lag}"] = float(target_arr[-lag])
    for window in ROLLING_WINDOWS:
        row[f"roll_{window}"] = float(target_arr[-window:].mean())

    for exog_idx, col in enumerate(PAST_EXOG_COLUMNS):
        for lag in EXOG_LAGS:
            row[f"exog_{col}_lag{lag}"] = float(exog_arr[-lag, exog_idx])
        for window in EXOG_ROLLING_WINDOWS:
            row[f"exog_{col}_roll{window}"] = float(exog_arr[-window:, exog_idx].mean())

    return pd.DataFrame([row])[feature_cols]


def make_tabpfn_regressor():
    try:
        from tabpfn import TabPFNRegressor
    except Exception as exc:
        raise RuntimeError(
            "Could not import tabpfn.TabPFNRegressor. Install/activate a Python "
            "environment with the tabpfn package before running this script."
        ) from exc
    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        return TabPFNRegressor(device=device)
    except TypeError:
        return TabPFNRegressor()


def evaluate_client(
    client_id: str,
    split,
    client_idx: int,
    total_clients: int,
    capture_sample: bool = False,
) -> tuple[list[dict], dict[str, np.ndarray | str]]:
    log_progress(f"Evaluating client {client_idx}/{total_clients} (id={client_id})")

    train_vals = series_values(split.train, client_id)
    val_vals = series_values(split.val, client_id)
    test_vals = series_values(split.test, client_id)
    train_val_vals = np.concatenate([train_vals, val_vals])

    train_exog = split.train[PAST_EXOG_COLUMNS].values.astype(float)
    val_exog = split.val[PAST_EXOG_COLUMNS].values.astype(float)
    train_val_exog = np.vstack([train_exog, val_exog])

    y_scaler = StandardScaler()
    target_scaled = y_scaler.fit_transform(train_val_vals.reshape(-1, 1)).ravel()

    exog_scaler = StandardScaler()
    exog_scaled = exog_scaler.fit_transform(train_val_exog)

    # Calendar features (hour, day_of_week, etc.) are deterministic and known in advance.
    # Only weather features (temperature, humidity, etc.) need persistence.
    n_weather = len(WEATHER_COLUMNS)
    test_calendar_raw = split.test[CALENDAR_COLUMNS].values.astype(float)
    test_calendar_scaled = (
        (test_calendar_raw - exog_scaler.mean_[n_weather:]) / exog_scaler.scale_[n_weather:]
    )
    persisted_weather = exog_scaled[-1, :n_weather].copy()

    features = build_feature_frame(
        target_scaled,
        timestamp_index(pd.concat([split.train, split.val])),
        exog_scaled,
    ).tail(CONTEXT_ROWS)
    feature_cols = [c for c in features.columns if c != "target"]

    model = make_tabpfn_regressor()
    t0 = time.time()
    model.fit(
        features[feature_cols].values.astype(np.float32),
        features["target"].values.astype(np.float32),
    )
    fit_time = time.time() - t0
    log_progress(f"Client {client_id}: fit complete in {fit_time:.2f}s")

    base_target_history = list(target_scaled)
    base_exog_history = [row for row in exog_scaled]

    records = []
    sample_forecast = {}
    for horizon in HORIZONS:
        if len(test_vals) < horizon:
            continue

        target_history = base_target_history.copy()
        exog_history = [row.copy() for row in base_exog_history]
        preds_scaled = []

        t1 = time.time()
        for step in range(horizon):
            row = build_next_row(target_history, exog_history, feature_cols)
            pred = float(model.predict(row.values.astype(np.float32))[0])
            preds_scaled.append(pred)
            target_history.append(pred)
            # Future weather: persisted from last val observation (unknown).
            # Future calendar: actual values (deterministic, known in advance).
            next_exog = np.empty(len(PAST_EXOG_COLUMNS))
            next_exog[:n_weather] = persisted_weather
            next_exog[n_weather:] = test_calendar_scaled[step]
            exog_history.append(next_exog)
        inference_time = time.time() - t1

        forecast = y_scaler.inverse_transform(np.asarray(preds_scaled).reshape(-1, 1)).ravel()
        actual = test_vals[:horizon]
        metrics = compute_metrics(actual, forecast)
        records.append({
            "client_id": client_id,
            "horizon": horizon,
            "MSE": round(metrics["MSE"], 4),
            "MAE": round(metrics["MAE"], 4),
            "RMSE": round(metrics["RMSE"], 4),
            "train_time_sec": round(fit_time, 2),
            "inference_time_sec": round(inference_time, 4),
            "model": "tabpfn_exog_past_weather",
            "exog_setting": "past_exog_context_only",
            "context_rows": len(features),
            "uses_future_exog": False,
            "future_exog_strategy": "weather_persistence_calendar_actual",
        })
        if capture_sample and not sample_forecast and horizon == 24:
            sample_forecast = {
                "client_id": client_id,
                "actual": actual.copy(),
                "pred": forecast.copy(),
            }
        log_progress(f"Client {client_id}: horizon {horizon}h RMSE={metrics['RMSE']:.3f}")

    return records, sample_forecast


def _load_existing_results() -> pd.DataFrame:
    if not RESULTS_CSV.exists():
        return pd.DataFrame()
    try:
        existing = pd.read_csv(RESULTS_CSV)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    required = {"client_id", "horizon", "RMSE"}
    if not required.issubset(existing.columns):
        return pd.DataFrame()
    return existing


def _save_results(records: list[dict]) -> pd.DataFrame:
    results = pd.DataFrame(records)
    results.to_csv(RESULTS_CSV, index=False)
    log_progress(f"Saved partial results -> {RESULTS_CSV} ({len(results)} rows)")
    return results


def _plot_forecast_sample(sample_forecast: dict[str, np.ndarray | str]) -> None:
    if not sample_forecast:
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    actual = np.asarray(sample_forecast["actual"], dtype=float)
    pred = np.asarray(sample_forecast["pred"], dtype=float)
    client_id = str(sample_forecast["client_id"])

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(range(1, len(actual) + 1), actual, label="Actual", linewidth=1.5)
    ax.plot(range(1, len(pred) + 1), pred, linestyle="--", label="Forecast", linewidth=1.5)
    ax.set_xlabel("Step (hours ahead)")
    ax.set_ylabel("Electricity load")
    ax.set_title(f"TabPFN past-weather exog: 24h forecast sample (client {client_id})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(SAMPLE_FORECAST_PNG, dpi=150)
    plt.close(fig)
    print(f"Saved {SAMPLE_FORECAST_PNG}")


def _plot_error_by_horizon(results: pd.DataFrame) -> None:
    if results.empty:
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    summary = results.groupby("horizon")[["RMSE", "MAE"]].mean().reindex(HORIZONS)
    x = np.arange(len(HORIZONS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    rmse_bars = ax.bar(x - width / 2, summary["RMSE"], width, label="RMSE")
    mae_bars = ax.bar(x + width / 2, summary["MAE"], width, label="MAE")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h}h" for h in HORIZONS])
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("Error")
    ax.set_title("TabPFN past-weather exog: average error by horizon")
    ax.legend()
    for bar in list(rmse_bars) + list(mae_bars):
        height = bar.get_height()
        if np.isfinite(height):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height * 1.01,
                f"{height:.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.tight_layout()
    fig.savefig(ERROR_BY_HORIZON_PNG, dpi=150)
    plt.close(fig)
    print(f"Saved {ERROR_BY_HORIZON_PNG}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Validate data path/splits without training.")
    parser.add_argument("--restart", action="store_true", help="Ignore any existing partial TabPFN results.")
    args = parser.parse_args()

    ensure_output_dirs()
    split = load_and_split()
    if args.dry_run:
        dry_run_report(split, "TabPFN exogenous past-weather-only")
        print("Setting: past exogenous lag/rolling features only; no future observed weather/calendar.")
        return

    selected_clients = select_clients(split.train, split.load_columns)
    existing = pd.DataFrame() if args.restart else _load_existing_results()
    completed_clients = set()
    if not existing.empty:
        complete_counts = existing.groupby(existing["client_id"].astype(str))["horizon"].nunique()
        completed_clients = set(complete_counts[complete_counts >= len(HORIZONS)].index)
        log_progress(
            "Resuming TabPFN: "
            f"{len(completed_clients)}/{len(selected_clients)} clients already complete"
        )

    all_records = existing.to_dict("records") if not existing.empty else []
    sample_forecast = {}
    for idx, client_id in enumerate(selected_clients, start=1):
        if str(client_id) in completed_clients:
            log_progress(f"Skipping completed client {idx}/{len(selected_clients)} (id={client_id})")
            continue

        records, maybe_sample = evaluate_client(
            client_id,
            split,
            idx,
            len(selected_clients),
            capture_sample=not sample_forecast,
        )
        all_records.extend(records)
        if maybe_sample and not sample_forecast:
            sample_forecast = maybe_sample
        _save_results(all_records)

    results = _save_results(all_records)
    log_progress(f"Saved final results -> {RESULTS_CSV}")
    _plot_forecast_sample(sample_forecast)
    _plot_error_by_horizon(results)


if __name__ == "__main__":
    main()
