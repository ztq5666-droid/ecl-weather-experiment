"""
TabPFN past-exogenous baseline for the independent ECL weather experiment.

Active fair setting:
- Uses past load lag/rolling features.
- Uses past shared Lisbon weather/calendar lag/rolling features.
- Does not use future observed weather or future calendar from the test period.

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

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR.parent
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

from common.common import (  # noqa: E402
    CALENDAR_COLUMNS,
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
CONTEXT_ROWS = 1000
LAG_FEATURES = [1, 2, 3, 6, 12, 24, 48, 168]
ROLLING_WINDOWS = [24, 168]
EXOG_LAGS = [1, 24, 168]
EXOG_ROLLING_WINDOWS = [24, 168]
PAST_EXOG_COLUMNS = WEATHER_COLUMNS + CALENDAR_COLUMNS


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
    return TabPFNRegressor()


def evaluate_client(client_id: str, split, client_idx: int, total_clients: int) -> list[dict]:
    print(f"\nEvaluating client {client_idx}/{total_clients} (id={client_id})...")

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

    base_target_history = list(target_scaled)
    base_exog_history = [row for row in exog_scaled]
    persisted_exog = base_exog_history[-1]

    records = []
    for horizon in HORIZONS:
        if len(test_vals) < horizon:
            continue

        target_history = base_target_history.copy()
        exog_history = [row.copy() for row in base_exog_history]
        preds_scaled = []

        t1 = time.time()
        for _step in range(horizon):
            row = build_next_row(target_history, exog_history, feature_cols)
            pred = float(model.predict(row.values.astype(np.float32))[0])
            preds_scaled.append(pred)
            target_history.append(pred)
            exog_history.append(persisted_exog.copy())
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
            "future_exog_strategy": "persistence_from_last_validation_observation",
        })
        print(f"  Horizon {horizon:3d}h -> RMSE={metrics['RMSE']:.3f}")

    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Validate data path/splits without training.")
    args = parser.parse_args()

    ensure_output_dirs()
    split = load_and_split()
    if args.dry_run:
        dry_run_report(split, "TabPFN exogenous past-weather-only")
        print("Setting: past exogenous lag/rolling features only; no future observed weather/calendar.")
        return

    selected_clients = select_clients(split.train, split.load_columns)
    all_records = []
    for idx, client_id in enumerate(selected_clients, start=1):
        all_records.extend(evaluate_client(client_id, split, idx, len(selected_clients)))

    pd.DataFrame(all_records).to_csv(RESULTS_CSV, index=False)
    print(f"Saved results -> {RESULTS_CSV}")


if __name__ == "__main__":
    main()
