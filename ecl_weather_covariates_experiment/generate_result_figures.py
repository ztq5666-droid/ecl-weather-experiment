"""
Generate active-result figures for the independent ECL weather experiment.

Only active fair-setting metrics in outputs/raw_metrics are read. Archived
oracle-weather results are intentionally excluded.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / ".matplotlib"))

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RAW_RESULTS_DIR = ROOT / "outputs" / "raw_metrics"
DIAGNOSTIC_DIR = ROOT / "outputs" / "figures" / "model_diagnostics"
COMPARISON_DIR = ROOT / "outputs" / "figures" / "performance_comparison"

HORIZONS = [24, 48, 168]
ACTIVE_RESULT_FILES = [
    RAW_RESULTS_DIR / "arima_load_only_results.csv",
    RAW_RESULTS_DIR / "xgboost_exog_past_weather_results.csv",
    RAW_RESULTS_DIR / "lstm_exog_results.csv",
    RAW_RESULTS_DIR / "itransformer_exog_results.csv",
    RAW_RESULTS_DIR / "tabpfn_exog_past_weather_results.csv",
]
MODEL_LABELS = {
    "arima_load_only": "ARIMA load-only",
    "xgboost_exog_past_weather": "XGBoost past weather",
    "lstm_exog": "LSTM exog",
    "itransformer_style_exog": "iTransformer-style exog",
    "tabpfn_exog_past_weather": "TabPFN tabular exog",
}


def load_active_results() -> pd.DataFrame:
    frames = []
    for path in ACTIVE_RESULT_FILES:
        if path.exists():
            df = pd.read_csv(path)
            df["source_file"] = path.name
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No active result CSVs found in {RAW_RESULTS_DIR}")

    results = pd.concat(frames, ignore_index=True)
    if "uses_future_exog" in results.columns:
        invalid = results[results["uses_future_exog"].astype(str).str.lower() == "true"]
        if not invalid.empty:
            files = sorted(invalid["source_file"].unique())
            raise ValueError(f"Active results include future exog rows: {files}")
    return results


def display_model_name(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def save_error_by_horizon(results: pd.DataFrame) -> None:
    DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)
    for model, df in results.groupby("model"):
        summary = df.groupby("horizon")[["RMSE", "MAE"]].mean().reindex(HORIZONS)
        if summary.dropna(how="all").empty:
            continue

        x = np.arange(len(HORIZONS))
        width = 0.35
        fig, ax = plt.subplots(figsize=(8, 5))
        rmse_bars = ax.bar(x - width / 2, summary["RMSE"], width, label="RMSE")
        mae_bars = ax.bar(x + width / 2, summary["MAE"], width, label="MAE")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{h}h" for h in HORIZONS])
        ax.set_xlabel("Forecast horizon")
        ax.set_ylabel("Error")
        ax.set_title(f"{display_model_name(model)}: average error by horizon")
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
        out_path = DIAGNOSTIC_DIR / f"{model}_error_by_horizon.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved {out_path}")


def save_rmse_comparison(results: pd.DataFrame) -> None:
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    summary = (
        results.groupby(["model", "horizon"])["RMSE"]
        .mean()
        .reset_index()
    )
    pivot = summary.pivot(index="horizon", columns="model", values="RMSE").reindex(HORIZONS)
    pivot = pivot[[c for c in MODEL_LABELS if c in pivot.columns]]

    fig, ax = plt.subplots(figsize=(9, 5))
    for model in pivot.columns:
        ax.plot(
            pivot.index,
            pivot[model],
            marker="o",
            linewidth=2,
            label=display_model_name(model),
        )
    ax.set_xlabel("Forecast horizon (hours)")
    ax.set_ylabel("Average RMSE")
    ax.set_title("Active model comparison: RMSE by forecast horizon")
    ax.set_xticks(HORIZONS)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out_path = COMPARISON_DIR / "active_models_rmse_by_horizon.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def save_xgboost_importance() -> None:
    path = RAW_RESULTS_DIR / "xgboost_exog_past_weather_feature_importance.csv"
    if not path.exists():
        return

    importance = pd.read_csv(path)
    summary = (
        importance.groupby("feature")["importance"]
        .mean()
        .sort_values(ascending=False)
        .head(20)
        .sort_values()
    )
    if summary.empty:
        return

    DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(summary.index, summary.values)
    ax.set_xlabel("Mean feature importance")
    ax.set_title("XGBoost past-weather feature importance")
    fig.tight_layout()
    out_path = DIAGNOSTIC_DIR / "xgboost_exog_past_weather_top_features.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    results = load_active_results()
    save_error_by_horizon(results)
    save_rmse_comparison(results)
    save_xgboost_importance()


if __name__ == "__main__":
    main()
