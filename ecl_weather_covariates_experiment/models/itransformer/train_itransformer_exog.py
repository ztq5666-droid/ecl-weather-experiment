"""
iTransformer-style exogenous baseline for the independent ECL weather experiment.

Reference pattern: dissertation/models/itransformer/train_itransformer.py

This independent script implements an inverted-variable Transformer locally:
time windows are projected into variable tokens, and self-attention operates
across variables. The variable set is all ECL clients plus shared Lisbon
weather/calendar covariates.
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
    load_and_split,
    select_clients,
)


RESULTS_CSV = RAW_RESULTS_DIR / "itransformer_exog_results.csv"
SEQ_LEN = 96
PRED_LEN = 168
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 2
D_FF = 256
DROPOUT = 0.1
LEARNING_RATE = 0.0001
MAX_EPOCHS = 10
PATIENCE = 3
BATCH_SIZE = 32


def get_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def matrix(df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return df[columns].ffill().bfill().values.astype(np.float32)


def evaluate(split, device) -> list[dict]:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    all_columns = split.load_columns + split.exog_columns
    n_load = len(split.load_columns)
    selected_clients = select_clients(split.train, split.load_columns)
    selected_indices = [split.load_columns.index(c) for c in selected_clients]

    class MultiWindowDataset(Dataset):
        def __init__(self, data: np.ndarray):
            self.data = data.astype(np.float32)

        def __len__(self) -> int:
            return max(0, len(self.data) - SEQ_LEN - PRED_LEN + 1)

        def __getitem__(self, idx):
            x = self.data[idx:idx + SEQ_LEN]
            y = self.data[idx + SEQ_LEN:idx + SEQ_LEN + PRED_LEN]
            return torch.tensor(x), torch.tensor(y)

    class InvertedTransformer(nn.Module):
        def __init__(self, seq_len: int, pred_len: int, n_vars: int):
            super().__init__()
            self.value_embedding = nn.Linear(seq_len, D_MODEL)
            layer = nn.TransformerEncoderLayer(
                d_model=D_MODEL,
                nhead=N_HEADS,
                dim_feedforward=D_FF,
                dropout=DROPOUT,
                activation="gelu",
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=N_LAYERS)
            self.projection = nn.Linear(D_MODEL, pred_len)
            self.n_vars = n_vars

        def forward(self, x):
            # x: [B, T, N] -> variable tokens [B, N, T]
            tokens = x.permute(0, 2, 1)
            emb = self.value_embedding(tokens)
            encoded = self.encoder(emb)
            out = self.projection(encoded)
            return out.permute(0, 2, 1)  # [B, pred_len, N]

    train = matrix(split.train, all_columns)
    val = matrix(split.val, all_columns)
    test = matrix(split.test, all_columns)

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train)
    val_scaled = scaler.transform(val)
    test_scaled = scaler.transform(test)

    train_dataset = MultiWindowDataset(train_scaled)
    val_dataset = MultiWindowDataset(np.vstack([train_scaled[-SEQ_LEN:], val_scaled]))
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise RuntimeError("Insufficient rows for iTransformer exogenous windows.")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = InvertedTransformer(SEQ_LEN, PRED_LEN, len(all_columns)).to(device)
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
            pred = model(xb)
            loss = loss_fn(pred[:, :, :n_load], yb[:, :, :n_load])
            loss.backward()
            optimiser.step()

        model.eval()
        losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                losses.append(float(loss_fn(pred[:, :, :n_load], yb[:, :, :n_load]).cpu()))
        mean_val = float(np.mean(losses))
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

    context = np.vstack([train_scaled, val_scaled])[-SEQ_LEN:]
    t1 = time.time()
    model.eval()
    with torch.no_grad():
        pred_scaled = model(torch.tensor(context[None, :, :], dtype=torch.float32, device=device))
    inference_time = time.time() - t1

    pred_full = scaler.inverse_transform(pred_scaled.cpu().numpy()[0])[:, :n_load]
    actual_full = test[:PRED_LEN, :n_load]

    records = []
    for client_id, client_idx in zip(selected_clients, selected_indices):
        for horizon in HORIZONS:
            actual = actual_full[:horizon, client_idx]
            pred = pred_full[:horizon, client_idx]
            metrics = compute_metrics(actual, pred)
            records.append({
                "client_id": client_id,
                "horizon": horizon,
                "MSE": round(metrics["MSE"], 4),
                "MAE": round(metrics["MAE"], 4),
                "RMSE": round(metrics["RMSE"], 4),
                "train_time_sec": round(train_time, 2),
                "inference_time_sec": round(inference_time, 4),
                "model": "itransformer_style_exog",
                "exog_setting": "past_exog_context_only",
                "n_variables": len(all_columns),
                "uses_future_exog": False,
            })
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Validate data path/splits without training.")
    args = parser.parse_args()

    ensure_output_dirs()
    split = load_and_split()
    if args.dry_run:
        dry_run_report(split, "iTransformer exogenous")
        print(f"Variables used by model: {len(split.load_columns) + len(split.exog_columns)}")
        return

    device = get_device()
    records = evaluate(split, device)
    pd.DataFrame(records).to_csv(RESULTS_CSV, index=False)
    print(f"Saved results -> {RESULTS_CSV}")


if __name__ == "__main__":
    main()
