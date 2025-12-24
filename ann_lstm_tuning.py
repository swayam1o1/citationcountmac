"""ann_lstm_tuning.py

Hyperparameter tuning for ANN and LSTM regressors predicting 2022 citations.

Inputs
- Excel: /Users/swayam/Downloads/1992.xlsx (Sheet 0)
  Required columns: 'Article Id', 2021, 2022
- Similarity CSV (workspace): papers_vs_combined_topics_tfidf_cosine.csv
  Required columns: paper_id, cosine_tfidf_similarity

Features (ONLY)
- cites_2021
- cosine_tfidf_similarity

Target
- cites_2022

Outputs
- ann/<exp_id>/config_and_metrics.json
- ann/<exp_id>/test_predictions.csv
- lstm/<exp_id>/config_and_metrics.json
- lstm/<exp_id>/test_predictions.csv

Each experiment folder contains:
- config_and_metrics.json: model_type + hyperparams + split + metrics
- test_predictions.csv: paper_id,y_true,y_pred

Notes
- Deterministic splitting via SEED.
- Early stopping on validation (test) loss, patience=PATIENCE.
- Device: CUDA > MPS > CPU.

This script intentionally does NOT write into the existing MLP outputs. MLP output
folder is handled by the existing mlp script.
"""

from __future__ import annotations

import json
import math
import hashlib
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

EXCEL_PATH = Path("/Users/swayam/Downloads/1992.xlsx")
SIM_CSV_PATH = Path("papers_vs_combined_topics_tfidf_cosine.csv")

OUT_ANN = Path("ann")
OUT_LSTM = Path("lstm")

SEED = 42
PATIENCE = 10
SKIP_EXISTING = True


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(math.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


def load_and_merge() -> pd.DataFrame:
    if not EXCEL_PATH.exists():
        raise FileNotFoundError(f"Excel not found: {EXCEL_PATH}")
    if not SIM_CSV_PATH.exists():
        raise FileNotFoundError(f"Similarity CSV not found: {SIM_CSV_PATH.resolve()}")

    df_xl = pd.read_excel(EXCEL_PATH, sheet_name=0)
    df_xl = df_xl.loc[:, [c for c in df_xl.columns if not str(c).startswith("Unnamed")]]

    if "Article Id" not in df_xl.columns:
        raise ValueError("Excel missing required column: 'Article Id'")
    if 2021 not in df_xl.columns or 2022 not in df_xl.columns:
        raise ValueError("Excel missing required year columns: 2021 and/or 2022")

    df_xl = df_xl[["Article Id", 2021, 2022]].rename(
        columns={"Article Id": "paper_id", 2021: "cites_2021", 2022: "cites_2022"}
    )
    df_xl["paper_id"] = df_xl["paper_id"].astype(str)

    df_sim = pd.read_csv(SIM_CSV_PATH)
    expected_cols = {"paper_id", "cosine_tfidf_similarity"}
    if not expected_cols.issubset(df_sim.columns):
        raise ValueError(f"Similarity CSV must contain columns {expected_cols}, got {list(df_sim.columns)}")
    df_sim["paper_id"] = df_sim["paper_id"].astype(str)

    df = df_xl.merge(df_sim, on="paper_id", how="inner")

    df["cites_2021"] = pd.to_numeric(df["cites_2021"], errors="coerce")
    df["cites_2022"] = pd.to_numeric(df["cites_2022"], errors="coerce")
    df["cosine_tfidf_similarity"] = pd.to_numeric(df["cosine_tfidf_similarity"], errors="coerce")

    df = df.dropna(subset=["cites_2021", "cosine_tfidf_similarity", "cites_2022"]).copy()
    df = df[(df["cites_2021"] >= 0) & (df["cites_2022"] >= 0)].copy()

    df = df.set_index("paper_id").sort_index()
    return df


def standardize_train_only(
    x_train: np.ndarray, x_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std = np.where(std == 0, 1.0, std)

    x_train_z = (x_train - mean) / std
    x_test_z = (x_test - mean) / std
    return x_train_z, x_test_z, mean, std


def make_split_with_ids(
    df: pd.DataFrame, train_ratio: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    n = len(df)
    idx = np.arange(n)
    rng = np.random.RandomState(SEED)
    rng.shuffle(idx)

    n_train = int(round(train_ratio * n))
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]

    X = df[["cites_2021", "cosine_tfidf_similarity"]].to_numpy(dtype=np.float32)
    y = df[["cites_2022"]].to_numpy(dtype=np.float32)

    paper_ids = df.index.to_numpy(dtype=str)
    test_ids = paper_ids[test_idx].tolist()

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx], test_ids


class ANNRegressor(nn.Module):
    def __init__(self, hidden_sizes: List[int], activation: str, dropout: float):
        super().__init__()
        act = _activation(activation)

        layers: List[nn.Module] = []
        in_dim = 2
        for h in hidden_sizes:
            layers.extend([nn.Linear(in_dim, h), nn.BatchNorm1d(h), act, nn.Dropout(p=dropout)])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LSTMRegressor(nn.Module):
    """Treat the 2 features as a length-2 sequence with 1 feature per timestep."""

    def __init__(self, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        # input_size=1 because each timestep holds one scalar feature
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # x_seq: (batch, seq_len=2, input_size=1)
        out, _ = self.lstm(x_seq)
        last = out[:, -1, :]  # (batch, hidden)
        return self.head(last)


def _activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "leakyrelu":
        return nn.LeakyReLU(0.01)
    if name == "elu":
        return nn.ELU(1.0)
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    raise ValueError(f"Unknown activation: {name}")


def _build_optimizer(name: str, params, lr: float) -> torch.optim.Optimizer:
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    raise ValueError(f"Unknown optimizer: {name}")


def train_predict(
    model: nn.Module,
    device: torch.device,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_ids: List[str],
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    optimizer: str,
) -> Tuple[Dict[str, float], int, np.ndarray]:
    """Train with early stopping; return metrics, best_epoch, predictions for test."""

    X_train_z, X_test_z, _, _ = standardize_train_only(X_train, X_test)

    # Prepare tensors
    Xtr = torch.tensor(X_train_z, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.float32)
    Xte = torch.tensor(X_test_z, dtype=torch.float32)
    yte = torch.tensor(y_test, dtype=torch.float32)

    if isinstance(model, LSTMRegressor):
        # reshape to (batch, seq_len=2, input_size=1)
        Xtr = Xtr.view(-1, 2, 1)
        Xte = Xte.view(-1, 2, 1)

    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)

    model = model.to(device)
    opt = _build_optimizer(optimizer, model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    no_improve = 0

    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            pred_te = model(Xte.to(device))
            val_loss = float(loss_fn(pred_te, yte.to(device)).item())

        if val_loss + 1e-12 < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = ep
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        y_pred = model(Xte.to(device)).detach().cpu().numpy().reshape(-1)

    metrics = regression_metrics(y_test, y_pred.reshape(-1, 1))
    return metrics, best_epoch, y_pred


def write_experiment(
    base_dir: Path,
    exp_name: str,
    *,
    config: Dict[str, object],
    metrics: Dict[str, float],
    best_epoch: int,
    test_ids: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    exp_dir = base_dir / exp_name
    if exp_dir.exists():
        if SKIP_EXISTING:
            return
        raise FileExistsError(f"Experiment directory already exists: {exp_dir}")
    exp_dir.mkdir(parents=True, exist_ok=False)

    payload = {
        "config": config,
        "metrics": metrics,
        "best_epoch": int(best_epoch),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (exp_dir / "config_and_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    dfp = pd.DataFrame({"paper_id": test_ids, "y_true": y_true.reshape(-1).astype(float), "y_pred": y_pred.reshape(-1).astype(float)})
    dfp.to_csv(exp_dir / "test_predictions.csv", index=False)


def _stable_exp_id(config: Dict[str, object]) -> str:
    """Create a stable experiment id from config (so resume works)."""
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    h = hashlib.sha1(payload).hexdigest()[:12]
    return f"exp_{h}"


def main() -> None:
    set_seed(SEED)
    device = get_device()

    df = load_and_merge()
    print(f"Merged dataset size: {len(df)}")
    print(f"Device: {device}")

    OUT_ANN.mkdir(exist_ok=True)
    OUT_LSTM.mkdir(exist_ok=True)

    # Match the MLP script's 7,500-combination grid.
    # Total = 5(train_ratios)*4(epochs)*5(batch)*5(lr)*3(opt)*5(activation) = 7500
    train_ratios = [0.5, 0.6, 0.7, 0.8, 0.9]
    common_epochs = [20, 30, 40, 50]
    common_batch_sizes = [8, 16, 32, 64, 128]
    common_lrs = [1e-4, 5e-4, 1e-3, 3e-3, 1e-2]
    common_opts = ["adam", "rmsprop", "sgd"]

    # We reuse the same 5 activations dimension to hit 7,500.
    activation_grid = ["relu", "leakyrelu", "elu", "tanh", "sigmoid"]

    # Model-specific knobs are held constant so the grid cardinality stays 7,500.
    ann_hidden_sizes = [128, 64, 32]
    ann_dropout = 0.2

    lstm_hidden_size = 32
    lstm_num_layers = 2
    lstm_dropout = 0.2

    ann_written = 0
    lstm_written = 0

    for tr in train_ratios:
        X_train, X_test, y_train, y_test, test_ids = make_split_with_ids(df, tr)

        # ANN (7,500 experiments)
        for epochs in common_epochs:
            for bs in common_batch_sizes:
                for lr in common_lrs:
                    for opt in common_opts:
                        for act in activation_grid:
                            cfg = {
                                "model_type": "ann",
                                "train_ratio": float(tr),
                                "epochs": int(epochs),
                                "batch_size": int(bs),
                                "learning_rate": float(lr),
                                "optimizer": str(opt),
                                "hidden_sizes": ann_hidden_sizes,
                                "activation": str(act),
                                "dropout": float(ann_dropout),
                                "features": ["cites_2021", "cosine_tfidf_similarity"],
                                "target": "cites_2022",
                                "seed": SEED,
                            }
                            exp_name = _stable_exp_id(cfg)
                            if SKIP_EXISTING and (OUT_ANN / exp_name).exists():
                                continue

                            model = ANNRegressor(hidden_sizes=ann_hidden_sizes, activation=act, dropout=float(ann_dropout))
                            metrics, best_epoch, y_pred = train_predict(
                                model,
                                device,
                                X_train,
                                y_train,
                                X_test,
                                y_test,
                                test_ids,
                                epochs=int(epochs),
                                batch_size=int(bs),
                                lr=float(lr),
                                optimizer=str(opt),
                            )
                            write_experiment(
                                OUT_ANN,
                                exp_name,
                                config=cfg,
                                metrics=metrics,
                                best_epoch=best_epoch,
                                test_ids=test_ids,
                                y_true=y_test,
                                y_pred=y_pred,
                            )
                            ann_written += 1

        # LSTM (7,500 experiments)
        for epochs in common_epochs:
            for bs in common_batch_sizes:
                for lr in common_lrs:
                    for opt in common_opts:
                        for act in activation_grid:
                            # For LSTM we map the activation hyperparameter to the post-LSTM head nonlinearity.
                            # This keeps the grid cardinality aligned with the MLP search.
                            cfg = {
                                "model_type": "lstm",
                                "train_ratio": float(tr),
                                "epochs": int(epochs),
                                "batch_size": int(bs),
                                "learning_rate": float(lr),
                                "optimizer": str(opt),
                                "hidden_size": int(lstm_hidden_size),
                                "num_layers": int(lstm_num_layers),
                                "dropout": float(lstm_dropout),
                                "head_activation": str(act),
                                "features": ["cites_2021", "cosine_tfidf_similarity"],
                                "target": "cites_2022",
                                "seed": SEED,
                                "sequence_representation": "[cites_2021, cosine_tfidf_similarity] as 2 timesteps",
                            }
                            exp_name = _stable_exp_id(cfg)
                            if SKIP_EXISTING and (OUT_LSTM / exp_name).exists():
                                continue

                            # Build LSTM and add a head activation module.
                            base = LSTMRegressor(hidden_size=int(lstm_hidden_size), num_layers=int(lstm_num_layers), dropout=float(lstm_dropout))
                            base.head = nn.Sequential(nn.LayerNorm(lstm_hidden_size), _activation(act), nn.Linear(lstm_hidden_size, 1))

                            metrics, best_epoch, y_pred = train_predict(
                                base,
                                device,
                                X_train,
                                y_train,
                                X_test,
                                y_test,
                                test_ids,
                                epochs=int(epochs),
                                batch_size=int(bs),
                                lr=float(lr),
                                optimizer=str(opt),
                            )
                            write_experiment(
                                OUT_LSTM,
                                exp_name,
                                config=cfg,
                                metrics=metrics,
                                best_epoch=best_epoch,
                                test_ids=test_ids,
                                y_true=y_test,
                                y_pred=y_pred,
                            )
                            lstm_written += 1

    print(f"Done. New experiments written: ANN={ann_written} | LSTM={lstm_written}")


if __name__ == "__main__":
    main()
