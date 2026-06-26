"""
LSTM exogenous baseline for the independent ECL weather experiment.

Reference pattern: dissertation/models/lstm/train_lstm.py

This per-client model adds historical Lisbon weather/calendar covariates as
input channels. The target remains a single electricity load series.
"""

from __future__ import annotations

import argparse
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
    HORIZONS,
    RAW_RESULTS_DIR,
    RANDOM_SEED,
    compute_metrics,
    dry_run_report,
    ensure_output_dirs,
    exog_values,
    load_and_split,
    select_clients,
    series_values,
)


RESULTS_CSV = RAW_RESULTS_DIR / "lstm_exog_results.csv"
SEQ_LEN = 168
FORECAST_LEN = 168
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2
LEARNING_RATE = 0.001
MAX_EPOCHS = 50
PATIENCE = 10
BATCH_SIZE = 32


def get_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_feature_matrix(target_scaled: np.ndarray, exog_scaled: np.ndarray) -> np.ndarray:
    return np.column_stack([target_scaled, exog_scaled]).astype(np.float32)


def evaluate_client(client_id: str, split, client_idx: int, total_clients: int, device) -> list[dict]:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    class WindowDataset(Dataset):
        def __init__(self, features: np.ndarray, target: np.ndarray):
            self.features = features.astype(np.float32)
            self.target = target.astype(np.float32)

        def __len__(self) -> int:
            return max(0, len(self.features) - SEQ_LEN - FORECAST_LEN + 1)

        def __getitem__(self, idx):
            x = self.features[idx:idx + SEQ_LEN]
            y = self.target[idx + SEQ_LEN:idx + SEQ_LEN + FORECAST_LEN]
            return torch.tensor(x), torch.tensor(y)

    class LSTMExogForecaster(nn.Module):
        def __init__(self, input_size: int):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=HIDDEN_SIZE,
                num_layers=NUM_LAYERS,
                dropout=DROPOUT if NUM_LAYERS > 1 else 0.0,
                batch_first=True,
            )
            self.fc = nn.Linear(HIDDEN_SIZE, FORECAST_LEN)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    print(f"\nTraining client {client_idx}/{total_clients} (id={client_id})...")

    train_vals = series_values(split.train, client_id)
    val_vals = series_values(split.val, client_id)
    test_vals = series_values(split.test, client_id)

    train_exog = exog_values(split.train, split.exog_columns)
    val_exog = exog_values(split.val, split.exog_columns)

    y_scaler = StandardScaler()
    train_scaled = y_scaler.fit_transform(train_vals.reshape(-1, 1)).ravel()
    val_scaled = y_scaler.transform(val_vals.reshape(-1, 1)).ravel()
    test_scaled = y_scaler.transform(test_vals.reshape(-1, 1)).ravel()

    x_scaler = StandardScaler()
    train_exog_scaled = x_scaler.fit_transform(train_exog)
    val_exog_scaled = x_scaler.transform(val_exog)

    train_features = make_feature_matrix(train_scaled, train_exog_scaled)
    val_context_features = make_feature_matrix(
        np.concatenate([train_scaled[-SEQ_LEN:], val_scaled]),
        np.vstack([train_exog_scaled[-SEQ_LEN:], val_exog_scaled]),
    )
    val_context_target = np.concatenate([train_scaled[-SEQ_LEN:], val_scaled])

    train_dataset = WindowDataset(train_features, train_scaled)
    val_dataset = WindowDataset(val_context_features, val_context_target)
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print(f"Skipping client {client_id}: insufficient sequence windows.")
        return []

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = LSTMExogForecaster(input_size=train_features.shape[1]).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    best_state = None
    best_val = float("inf")
    bad_epochs = 0
    t0 = time.time()
    for _epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimiser.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimiser.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                val_losses.append(float(loss_fn(model(xb), yb).cpu()))
        mean_val = float(np.mean(val_losses))
        if mean_val < best_val:
            best_val = mean_val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    train_time = time.time() - t0

    train_val_exog_scaled = np.vstack([train_exog_scaled, val_exog_scaled])
    train_val_target_scaled = np.concatenate([train_scaled, val_scaled])
    context_features = make_feature_matrix(
        train_val_target_scaled[-SEQ_LEN:],
        train_val_exog_scaled[-SEQ_LEN:],
    )

    t1 = time.time()
    model.eval()
    with torch.no_grad():
        pred_scaled = model(
            torch.tensor(context_features[None, :, :], dtype=torch.float32, device=device)
        ).cpu().numpy().ravel()
    inference_time = time.time() - t1
    forecast = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()

    records = []
    for horizon in HORIZONS:
        if len(test_vals) < horizon:
            continue
        actual = test_vals[:horizon]
        pred = forecast[:horizon]
        metrics = compute_metrics(actual, pred)
        records.append({
            "client_id": client_id,
            "horizon": horizon,
            "MSE": round(metrics["MSE"], 4),
            "MAE": round(metrics["MAE"], 4),
            "RMSE": round(metrics["RMSE"], 4),
            "train_time_sec": round(train_time, 2),
            "inference_time_sec": round(inference_time, 4),
            "model": "lstm_exog",
            "exog_setting": "past_exog_context_only",
            "uses_future_exog": False,
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
        dry_run_report(split, "LSTM exogenous")
        return

    device = get_device()
    selected_clients = select_clients(split.train, split.load_columns)
    all_records = []
    for idx, client_id in enumerate(selected_clients, start=1):
        all_records.extend(evaluate_client(client_id, split, idx, len(selected_clients), device))

    pd.DataFrame(all_records).to_csv(RESULTS_CSV, index=False)
    print(f"Saved results -> {RESULTS_CSV}")


if __name__ == "__main__":
    main()
