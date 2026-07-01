"""
Load-only ARIMA baseline for the independent ECL weather experiment.

ARIMA is kept as a statistical benchmark. It does not use Lisbon weather or
calendar covariates, because requiring future exogenous values would move it
into an oracle setting. This script uses the same split, selected clients,
horizons and fixed forecast origin as the active weather-augmented models.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR.parent
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(SCRIPT_DIR.parents[1] / "outputs" / ".matplotlib"),
)
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common.common import (  # noqa: E402
    FIGURES_DIR,
    HORIZONS,
    RAW_RESULTS_DIR,
    compute_metrics,
    dry_run_report,
    ensure_output_dirs,
    load_and_split,
    select_clients,
    series_values,
)


RESULTS_CSV = RAW_RESULTS_DIR / "arima_load_only_results.csv"
SAMPLE_FORECAST_PNG = FIGURES_DIR / "arima_load_only_forecast_sample.png"
ERROR_BY_HORIZON_PNG = FIGURES_DIR / "arima_load_only_error_by_horizon.png"
TRAIN_WINDOW = 2000


def make_auto_arima(train_scaled: np.ndarray):
    try:
        import pmdarima as pm
    except Exception as exc:
        raise RuntimeError(
            "Could not import pmdarima. Install/activate the environment used "
            "for the dissertation ARIMA baseline before running this script."
        ) from exc

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pm.auto_arima(
            train_scaled,
            seasonal=False,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            information_criterion="aic",
            max_p=5,
            max_q=5,
            max_d=2,
            trace=False,
        )


def evaluate_client(client_id: str, split, client_idx: int, total_clients: int) -> list[dict]:
    print(f"\nTraining client {client_idx}/{total_clients} (id={client_id})...")

    train_vals = series_values(split.train, client_id)
    val_vals = series_values(split.val, client_id)
    test_vals = series_values(split.test, client_id)
    train_val_vals = np.concatenate([train_vals, val_vals])

    train_window = train_val_vals[-TRAIN_WINDOW:]
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_window.reshape(-1, 1)).ravel()

    t0 = time.time()
    model = make_auto_arima(train_scaled)
    train_time = time.time() - t0

    records = []
    for horizon in HORIZONS:
        if len(test_vals) < horizon:
            continue

        t1 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast_scaled = model.predict(n_periods=horizon)
        inference_time = time.time() - t1

        forecast = scaler.inverse_transform(
            np.asarray(forecast_scaled).reshape(-1, 1)
        ).ravel()
        actual = test_vals[:horizon]
        metrics = compute_metrics(actual, forecast)
        records.append({
            "client_id": client_id,
            "horizon": horizon,
            "MSE": round(metrics["MSE"], 4),
            "MAE": round(metrics["MAE"], 4),
            "RMSE": round(metrics["RMSE"], 4),
            "train_time_sec": round(train_time, 2),
            "inference_time_sec": round(inference_time, 4),
            "model": "arima_load_only",
            "exog_setting": "load_only_statistical_baseline",
            "uses_future_exog": False,
            "future_exog_strategy": "not_applicable_load_only",
            "arima_order": str(model.order),
            "seasonal_order": str(model.seasonal_order),
            "train_window_rows": TRAIN_WINDOW,
        })
        print(f"  Horizon {horizon:3d}h -> RMSE={metrics['RMSE']:.3f}")

    return records


def _plot_forecast_sample(sample_client_id: str, split) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    train_vals = series_values(split.train, sample_client_id)
    val_vals = series_values(split.val, sample_client_id)
    test_vals = series_values(split.test, sample_client_id)
    train_val_vals = np.concatenate([train_vals, val_vals])
    train_window = train_val_vals[-TRAIN_WINDOW:]
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_window.reshape(-1, 1)).ravel()
    model = make_auto_arima(train_scaled)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        forecast_scaled = model.predict(n_periods=24)
    forecast = scaler.inverse_transform(
        np.asarray(forecast_scaled).reshape(-1, 1)
    ).ravel()
    actual = test_vals[:24]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(range(1, 25), actual, label="Actual", linewidth=1.5)
    ax.plot(range(1, 25), forecast, linestyle="--", label="Forecast", linewidth=1.5)
    ax.set_xlabel("Step (hours ahead)")
    ax.set_ylabel("Electricity load")
    ax.set_title(f"ARIMA load-only: 24h forecast sample (client {sample_client_id})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(SAMPLE_FORECAST_PNG, dpi=150)
    plt.close(fig)
    print(f"Saved {SAMPLE_FORECAST_PNG}")


def _plot_error_by_horizon(results: pd.DataFrame) -> None:
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
    ax.set_title("ARIMA load-only: average error by horizon")
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
    args = parser.parse_args()

    ensure_output_dirs()
    split = load_and_split()
    if args.dry_run:
        dry_run_report(split, "ARIMA load-only statistical baseline")
        print("Setting: load-only ARIMA baseline; no weather/calendar covariates.")
        return

    selected_clients = select_clients(split.train, split.load_columns)
    all_records = []
    for idx, client_id in enumerate(selected_clients, start=1):
        all_records.extend(evaluate_client(client_id, split, idx, len(selected_clients)))

    results = pd.DataFrame(all_records)
    results.to_csv(RESULTS_CSV, index=False)
    print(f"Saved results -> {RESULTS_CSV}")

    if not results.empty:
        print("\nARIMA load-only summary:")
        summary = results.groupby("horizon")[["MSE", "MAE", "RMSE", "train_time_sec"]].mean()
        print(summary.round(4).to_string())

    _plot_forecast_sample(selected_clients[0], split)
    _plot_error_by_horizon(results)


if __name__ == "__main__":
    main()
