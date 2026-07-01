"""
XGBoost past-weather-only exogenous baseline.

This is the main fair-setting XGBoost extension:
- Uses past load lag/rolling features.
- Uses past Lisbon weather/calendar lag/rolling features only.
- Does not use future observed weather or future calendar from the test period.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR.parent
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

from common.common import (  # noqa: E402
    CALENDAR_COLUMNS,
    FIGURES_DIR,
    HORIZONS,
    RAW_RESULTS_DIR,
    RANDOM_SEED,
    WEATHER_COLUMNS,
    compute_metrics,
    dry_run_report,
    ensure_output_dirs,
    load_and_split,
    timestamp_index,
    select_clients,
    series_values,
)


RESULTS_CSV = RAW_RESULTS_DIR / "xgboost_exog_past_weather_results.csv"
FEATURE_IMPORTANCE_CSV = RAW_RESULTS_DIR / "xgboost_exog_past_weather_feature_importance.csv"

LAG_FEATURES = [1, 2, 3, 6, 12, 24, 48, 168]
ROLLING_WINDOWS = [24, 168]
WEATHER_LAGS = [1, 24, 168]
WEATHER_ROLLING_WINDOWS = [24, 168]
PAST_EXOG_COLUMNS = WEATHER_COLUMNS + CALENDAR_COLUMNS

XGB_PARAMS = dict(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=RANDOM_SEED,
    n_jobs=-1,
)


def _set_macos_openmp_path() -> None:
    libomp_dirs = [
        "/Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages/sklearn/.dylibs",
        "/opt/anaconda3/lib",
    ]
    existing = os.environ.get("DYLD_LIBRARY_PATH", "")
    for directory in libomp_dirs:
        if os.path.isdir(directory) and directory not in existing:
            os.environ["DYLD_LIBRARY_PATH"] = directory + (":" + existing if existing else "")
            break


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
        for lag in WEATHER_LAGS:
            df[f"exog_{col}_lag{lag}"] = exog_df[col].shift(lag)
        for window in WEATHER_ROLLING_WINDOWS:
            df[f"exog_{col}_roll{window}"] = (
                exog_df[col].shift(1).rolling(window=window, min_periods=1).mean()
            )

    df["target"] = target_scaled
    return df.dropna()


def build_next_feature_row(
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
        for lag in WEATHER_LAGS:
            row[f"exog_{col}_lag{lag}"] = float(exog_arr[-lag, exog_idx])
        for window in WEATHER_ROLLING_WINDOWS:
            row[f"exog_{col}_roll{window}"] = float(exog_arr[-window:, exog_idx].mean())

    return pd.DataFrame([row])[feature_cols]


def train_xgboost_model(x_train, y_train, x_val, y_val):
    _set_macos_openmp_path()
    import xgboost as xgb

    try:
        model = xgb.XGBRegressor(**XGB_PARAMS, early_stopping_rounds=50)
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
        return model
    except Exception:
        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(x_train, y_train, verbose=False)
        return model


def evaluate_client(client_id: str, split, client_idx: int, total_clients: int) -> tuple[list[dict], list[dict]]:
    print(f"\nTraining client {client_idx}/{total_clients} (id={client_id})...")

    train_vals = series_values(split.train, client_id)
    val_vals = series_values(split.val, client_id)
    test_vals = series_values(split.test, client_id)

    train_exog = split.train[PAST_EXOG_COLUMNS].values.astype(float)
    val_exog = split.val[PAST_EXOG_COLUMNS].values.astype(float)

    target_scaler = StandardScaler()
    train_scaled = target_scaler.fit_transform(train_vals.reshape(-1, 1)).ravel()
    val_scaled = target_scaler.transform(val_vals.reshape(-1, 1)).ravel()

    exog_scaler = StandardScaler()
    train_exog_scaled = exog_scaler.fit_transform(train_exog)
    val_exog_scaled = exog_scaler.transform(val_exog)

    # Calendar features (hour, day_of_week, etc.) are deterministic and known in advance.
    # Only weather features (temperature, humidity, etc.) need persistence.
    n_weather = len(WEATHER_COLUMNS)
    test_calendar_raw = split.test[CALENDAR_COLUMNS].values.astype(float)
    test_calendar_scaled = (
        (test_calendar_raw - exog_scaler.mean_[n_weather:]) / exog_scaler.scale_[n_weather:]
    )
    persisted_weather = val_exog_scaled[-1, :n_weather].copy()

    train_features = build_feature_frame(
        train_scaled,
        timestamp_index(split.train),
        train_exog_scaled,
    )
    train_val_features = build_feature_frame(
        np.concatenate([train_scaled, val_scaled]),
        timestamp_index(pd.concat([split.train, split.val])),
        np.vstack([train_exog_scaled, val_exog_scaled]),
    )
    val_features = train_val_features.iloc[len(train_features):]

    if train_features.empty or val_features.empty:
        print(f"Skipping client {client_id}: insufficient rows after feature construction.")
        return [], []

    feature_cols = [c for c in train_features.columns if c != "target"]
    t0 = time.time()
    model = train_xgboost_model(
        train_features[feature_cols].values.astype(np.float32),
        train_features["target"].values.astype(np.float32),
        val_features[feature_cols].values.astype(np.float32),
        val_features["target"].values.astype(np.float32),
    )
    train_time = time.time() - t0

    feature_importance_records = [
        {
            "client_id": client_id,
            "feature": feature,
            "importance": float(importance),
            "model": "xgboost_exog_past_weather",
            "exog_setting": "past_exog_context_only",
        }
        for feature, importance in zip(feature_cols, model.feature_importances_)
    ]

    records = []
    base_target_history = list(np.concatenate([train_scaled, val_scaled]))
    base_exog_history = [
        row for row in np.vstack([train_exog_scaled, val_exog_scaled])
    ]

    for horizon in HORIZONS:
        if len(test_vals) < horizon:
            continue

        # Fixed-origin evaluation. Future weather is unknown → persisted from last
        # validation observation. Future calendar is deterministic → actual values used.
        target_history = base_target_history.copy()
        exog_history = [row.copy() for row in base_exog_history]
        preds_scaled = []

        t1 = time.time()
        for step in range(horizon):
            next_row = build_next_feature_row(
                target_history,
                exog_history,
                feature_cols,
            )
            pred_scaled = float(model.predict(next_row.values.astype(np.float32))[0])
            preds_scaled.append(pred_scaled)
            target_history.append(pred_scaled)
            next_exog = np.empty(len(PAST_EXOG_COLUMNS))
            next_exog[:n_weather] = persisted_weather
            next_exog[n_weather:] = test_calendar_scaled[step]
            exog_history.append(next_exog)
        inference_time = time.time() - t1

        forecast = target_scaler.inverse_transform(
            np.asarray(preds_scaled).reshape(-1, 1)
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
            "model": "xgboost_exog_past_weather",
            "exog_setting": "past_exog_context_only",
            "uses_future_exog": False,
            "future_exog_strategy": "weather_persistence_calendar_actual",
        })
        print(f"  Horizon {horizon:3d}h -> RMSE={metrics['RMSE']:.3f}")

    return records, feature_importance_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Validate data path/splits without training.")
    args = parser.parse_args()

    ensure_output_dirs()
    split = load_and_split()
    if args.dry_run:
        dry_run_report(split, "XGBoost exogenous past-weather-only")
        print("Setting: past exogenous lag/rolling features only; no future observed weather/calendar.")
        return

    selected_clients = select_clients(split.train, split.load_columns)
    all_records = []
    all_importance_records = []
    for idx, client_id in enumerate(selected_clients, start=1):
        records, importance_records = evaluate_client(client_id, split, idx, len(selected_clients))
        all_records.extend(records)
        all_importance_records.extend(importance_records)

    results = pd.DataFrame(all_records)
    results.to_csv(RESULTS_CSV, index=False)
    print(f"Saved results -> {RESULTS_CSV}")
    pd.DataFrame(all_importance_records).to_csv(FEATURE_IMPORTANCE_CSV, index=False)
    print(f"Saved feature importance -> {FEATURE_IMPORTANCE_CSV}")
    print(f"Diagnostics figure directory reserved at -> {FIGURES_DIR}")


if __name__ == "__main__":
    main()
