"""mlp_hyperparam_search_2022_citations.py

Full hyperparameter search for an MLP predicting 2022 citations.

Uses ONLY 2 features per paper:
  1) 2021 citation count (from /Users/swayam/Downloads/1992.xlsx)
  2) TF-IDF overall similarity (from papers_vs_combined_topics_tfidf_cosine.csv)

Requirements implemented:
- Load 2021/2022 citations from Excel and similarity from CSV
- Merge into a single DataFrame indexed by paper_id
- Standardize features before training (fit on train only)
- Train/test splits: [0.5, 0.6, 0.7, 0.8, 0.9]
- Hyperparam grid:
    epochs:        [20, 30, 40, 50]
    batch_size:    [8, 16, 32, 64, 128]
    learning_rate: [1e-4, 5e-4, 1e-3, 3e-3, 1e-2]
    optimizer:     ['adam', 'rmsprop', 'sgd']
    activation:    ['relu', 'leakyrelu', 'elu', 'tanh', 'sigmoid']
- Fixed MLP: [128, 64, 32], dropout=0.2, batch norm enabled
- Early stopping with patience=10 (monitors validation loss on test set)
- Metrics per experiment: MSE, RMSE, MAE, R²
- Save results CSV and print top 10 by R²
- Device selection: GPU if available else CPU (supports Apple MPS)

Outputs:
  - mlp_hyperparam_search_results.csv

Notes:
- We use a deterministic random seed.
- We do NOT use any embeddings/LLMs; this is plain supervised regression.
"""

from __future__ import annotations

import itertools
import math
import os
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
OUTPUT_CSV = Path("mlp_hyperparam_search_results.csv")
PREDICTIONS_CSV = Path("mlp_hyperparam_search_test_predictions.csv")
RESULTS_DIR = Path("results")
PROGRESS_EVERY = 50  # print progress every N experiments

SEED = 42


def _append_row_csv(path: Path, row: Dict[str, object], *, header: List[str]) -> None:
    """Append a single row to a CSV, creating it (with header) if missing."""
    file_exists = path.exists() and path.stat().st_size > 0
    mode = "a" if file_exists else "w"
    with path.open(mode, encoding="utf-8", newline="") as f:
        if not file_exists:
            f.write(",".join(header) + "\n")

        def esc(v: object) -> str:
            # Values here are numeric or short strings; escape commas/quotes defensively.
            s = "" if v is None else str(v)
            if any(ch in s for ch in [",", "\"", "\n", "\r"]):
                s = '"' + s.replace('"', '""') + '"'
            return s

        f.write(",".join(esc(row.get(h)) for h in header) + "\n")


def _load_existing_results(path: Path) -> pd.DataFrame:
    """Load existing results if present; else return empty DF."""
    if path.exists() and path.stat().st_size > 0:
        try:
            return pd.read_csv(path)
        except Exception:
            # If partially-written, don't crash; start fresh.
            return pd.DataFrame()
    return pd.DataFrame()


def _exp_key(d: Dict[str, object]) -> str:
    """Uniquely identify an experiment configuration."""
    return (
        f"tr={d['train_ratio']}|ep={d['epochs']}|bs={d['batch_size']}|lr={d['learning_rate']}|"
        f"opt={d['optimizer']}|act={d['activation']}"
    )


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    # Prefer CUDA, else Apple MPS, else CPU
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def activation_from_name(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "leakyrelu":
        return nn.LeakyReLU(negative_slope=0.01)
    if name == "elu":
        return nn.ELU(alpha=1.0)
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    raise ValueError(f"Unknown activation: {name}")


class MLPRegressor(nn.Module):
    def __init__(self, activation: str, dropout: float = 0.2):
        super().__init__()
        act = activation_from_name(activation)

        # Fixed architecture: 2 -> 128 -> 64 -> 32 -> 1
        self.net = nn.Sequential(
            nn.Linear(2, 128),
            nn.BatchNorm1d(128),
            act,
            nn.Dropout(p=dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            act,
            nn.Dropout(p=dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            act,
            nn.Dropout(p=dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_optimizer(name: str, params, lr: float) -> torch.optim.Optimizer:
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    raise ValueError(f"Unknown optimizer: {name}")


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(math.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    # R^2
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


def load_and_merge() -> pd.DataFrame:
    """Load Excel + similarity CSV, merge on paper_id, and return clean DF."""
    if not EXCEL_PATH.exists():
        raise FileNotFoundError(f"Excel not found: {EXCEL_PATH}")
    if not SIM_CSV_PATH.exists():
        raise FileNotFoundError(f"Similarity CSV not found: {SIM_CSV_PATH.resolve()}")

    # Excel schema confirmed earlier:
    # columns include: 'Article Id', 2021, 2022
    df_xl = pd.read_excel(EXCEL_PATH, sheet_name=0)

    # Drop unnamed columns if present
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

    # Clean numeric
    df["cites_2021"] = pd.to_numeric(df["cites_2021"], errors="coerce")
    df["cites_2022"] = pd.to_numeric(df["cites_2022"], errors="coerce")
    df["cosine_tfidf_similarity"] = pd.to_numeric(df["cosine_tfidf_similarity"], errors="coerce")

    # Drop missing
    df = df.dropna(subset=["cites_2021", "cosine_tfidf_similarity", "cites_2022"]).copy()

    # Keep non-negative citations
    df = df[(df["cites_2021"] >= 0) & (df["cites_2022"] >= 0)].copy()

    df = df.set_index("paper_id").sort_index()
    return df


def standardize_train_only(x_train: np.ndarray, x_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Standardize using train mean/std only."""
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std = np.where(std == 0, 1.0, std)

    x_train_z = (x_train - mean) / std
    x_test_z = (x_test - mean) / std
    return x_train_z, x_test_z, mean, std


def make_split(df: pd.DataFrame, train_ratio: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic shuffle split."""
    n = len(df)
    idx = np.arange(n)
    rng = np.random.RandomState(SEED)
    rng.shuffle(idx)

    n_train = int(round(train_ratio * n))
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]

    X = df[["cites_2021", "cosine_tfidf_similarity"]].to_numpy(dtype=np.float32)
    y = df[["cites_2022"]].to_numpy(dtype=np.float32)

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    return X_train, X_test, y_train, y_test


def make_split_with_ids(
    df: pd.DataFrame, train_ratio: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """Deterministic shuffle split that also returns paper_ids for train/test."""
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
    train_ids = paper_ids[train_idx].tolist()
    test_ids = paper_ids[test_idx].tolist()

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    return X_train, X_test, y_train, y_test, train_ids, test_ids


def train_one(
    device: torch.device,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    optimizer_name: str,
    activation: str,
    patience: int = 10,
) -> Tuple[Dict[str, float], int]:
    """Train one configuration with early stopping on test loss."""

    # Standardize (train-only)
    X_train_z, X_test_z, _, _ = standardize_train_only(X_train, X_test)

    Xtr = torch.tensor(X_train_z, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.float32)
    Xte = torch.tensor(X_test_z, dtype=torch.float32)
    yte = torch.tensor(y_test, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)

    model = MLPRegressor(activation=activation, dropout=0.2).to(device)
    opt = build_optimizer(optimizer_name, model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

        # Validation (test) loss for early stopping
        model.eval()
        with torch.no_grad():
            pred_te = model(Xte.to(device))
            val_loss = float(loss_fn(pred_te, yte.to(device)).item())

        if val_loss + 1e-12 < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final metrics on test
    model.eval()
    with torch.no_grad():
        y_pred = model(Xte.to(device)).detach().cpu().numpy()
    metrics = regression_metrics(y_test, y_pred)
    return metrics, best_epoch


def train_one_with_predictions(
    device: torch.device,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_ids: List[str],
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    optimizer_name: str,
    activation: str,
    patience: int = 10,
) -> Tuple[Dict[str, float], int, pd.DataFrame]:
    """Train one configuration and also return per-paper y_true/y_pred for test set."""
    metrics, best_epoch = train_one(
        device,
        X_train,
        y_train,
        X_test,
        y_test,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        optimizer_name=optimizer_name,
        activation=activation,
        patience=patience,
    )

    # Recompute predictions once (standardizing the same way as train_one)
    X_train_z, X_test_z, _, _ = standardize_train_only(X_train, X_test)
    Xte = torch.tensor(X_test_z, dtype=torch.float32)

    model = MLPRegressor(activation=activation, dropout=0.2).to(device)
    opt = build_optimizer(optimizer_name, model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train_z, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    # Train again up to best_epoch (cheap vs returning the full model state)
    for _ in range(best_epoch):
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
        y_pred = model(Xte.to(device)).detach().cpu().numpy().reshape(-1)

    y_true = y_test.reshape(-1)
    preds_df = pd.DataFrame(
        {
            "paper_id": test_ids,
            "y_true": y_true.astype(float),
            "y_pred": y_pred.astype(float),
        }
    )
    return metrics, best_epoch, preds_df


@dataclass
class ExperimentResult:
    train_ratio: float
    epochs: int
    batch_size: int
    learning_rate: float
    optimizer: str
    activation: str
    best_epoch: int
    mse: float
    rmse: float
    mae: float
    r2: float
    n_train: int
    n_test: int


def main() -> None:
    set_seed(SEED)
    device = get_device()

    print("\n" + "=" * 70)
    print("MLP HYPERPARAMETER SEARCH (predict 2022 citations)")
    print("Features: cites_2021 + cosine_tfidf_similarity")
    print(f"Device: {device}")
    print("=" * 70 + "\n")

    df = load_and_merge()
    print(f"Merged dataset size: {len(df)} papers")

    # Where we write per-experiment prediction CSVs (one file per experiment)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_ratios = [0.5, 0.6, 0.7, 0.8, 0.9]
    epochs_grid = [20, 30, 40, 50]
    batch_sizes = [8, 16, 32, 64, 128]
    lrs = [1e-4, 5e-4, 1e-3, 3e-3, 1e-2]
    optimizers = ["adam", "rmsprop", "sgd"]
    activations = ["relu", "leakyrelu", "elu", "tanh", "sigmoid"]

    # Full grid size
    total_experiments = (
        len(train_ratios)
        * len(epochs_grid)
        * len(batch_sizes)
        * len(lrs)
        * len(optimizers)
        * len(activations)
    )
    print(f"Total experiments to run: {total_experiments}")

    # Resume support: if OUTPUT_CSV exists, skip already-run configs.
    existing = _load_existing_results(OUTPUT_CSV)
    done_keys = set()
    if not existing.empty:
        required_cols = {"train_ratio", "epochs", "batch_size", "learning_rate", "optimizer", "activation"}
        if required_cols.issubset(existing.columns):
            for _, r in existing.iterrows():
                done_keys.add(
                    _exp_key(
                        {
                            "train_ratio": float(r["train_ratio"]),
                            "epochs": int(r["epochs"]),
                            "batch_size": int(r["batch_size"]),
                            "learning_rate": float(r["learning_rate"]),
                            "optimizer": str(r["optimizer"]),
                            "activation": str(r["activation"]),
                        }
                    )
                )

    if done_keys:
        print(f"Resuming: found {len(done_keys)} completed experiments in {OUTPUT_CSV}")

    header = [
        "train_ratio",
        "epochs",
        "batch_size",
        "learning_rate",
        "optimizer",
        "activation",
        "best_epoch",
        "mse",
        "rmse",
        "mae",
        "r2",
        "n_train",
        "n_test",
    ]

    pred_header = [
        "experiment_key",
        "train_ratio",
        "epochs",
        "batch_size",
        "learning_rate",
        "optimizer",
        "activation",
        "best_epoch",
        "paper_id",
        "y_true",
        "y_pred",
    ]

    exp_num = 0
    written = 0

    for train_ratio in train_ratios:
        X_train, X_test, y_train, y_test, train_ids, test_ids = make_split_with_ids(df, train_ratio)
        n_train, n_test = len(X_train), len(X_test)
        if n_test < 5:
            print(f"Skipping train_ratio={train_ratio} because test set too small: n_test={n_test}")
            continue

        for epochs, batch_size, lr, opt_name, act in itertools.product(
            epochs_grid, batch_sizes, lrs, optimizers, activations
        ):
            key = _exp_key(
                {
                    "train_ratio": train_ratio,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": lr,
                    "optimizer": opt_name,
                    "activation": act,
                }
            )
            if key in done_keys:
                continue

            exp_num += 1
            metrics, best_epoch, preds_df = train_one_with_predictions(
                device,
                X_train,
                y_train,
                X_test,
                y_test,
                test_ids,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=lr,
                optimizer_name=opt_name,
                activation=act,
                patience=10,
            )


            row = {
                "train_ratio": train_ratio,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": lr,
                "optimizer": opt_name,
                "activation": act,
                "best_epoch": best_epoch,
                "mse": metrics["mse"],
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "r2": metrics["r2"],
                "n_train": n_train,
                "n_test": n_test,
            }

            _append_row_csv(OUTPUT_CSV, row, header=header)
            written += 1
            done_keys.add(key)

            # Write per-experiment predictions for ALL test samples.
            # One file per experiment makes analysis + parallelism easier.
            preds_df = preds_df.copy()
            preds_df.insert(0, "experiment_key", key)
            preds_df.insert(1, "train_ratio", train_ratio)
            preds_df.insert(2, "epochs", epochs)
            preds_df.insert(3, "batch_size", batch_size)
            preds_df.insert(4, "learning_rate", lr)
            preds_df.insert(5, "optimizer", opt_name)
            preds_df.insert(6, "activation", act)
            preds_df.insert(7, "best_epoch", best_epoch)

            # Safe filename (no characters like '=' or '|')
            safe_key = (
                f"tr_{train_ratio}_ep_{epochs}_bs_{batch_size}_lr_{lr}_opt_{opt_name}_act_{act}"
            ).replace(".", "p")
            exp_pred_path = RESULTS_DIR / f"test_predictions_{safe_key}.csv"
            preds_df.to_csv(exp_pred_path, index=False)

            # Optional: also append to a single aggregated predictions CSV.
            for _, pr in preds_df.iterrows():
                _append_row_csv(
                    PREDICTIONS_CSV,
                    {
                        "experiment_key": pr["experiment_key"],
                        "train_ratio": pr["train_ratio"],
                        "epochs": pr["epochs"],
                        "batch_size": pr["batch_size"],
                        "learning_rate": pr["learning_rate"],
                        "optimizer": pr["optimizer"],
                        "activation": pr["activation"],
                        "best_epoch": pr["best_epoch"],
                        "paper_id": pr["paper_id"],
                        "y_true": pr["y_true"],
                        "y_pred": pr["y_pred"],
                    },
                    header=pred_header,
                )

            if exp_num % PROGRESS_EVERY == 0:
                print(
                    f"Progress: +{written} written this run | {len(done_keys)}/{total_experiments} total done"
                )

    print(f"\nSaved results to: {OUTPUT_CSV.resolve()}")

    # Print top 10 by R^2 from combined (previous + current)
    df_out = _load_existing_results(OUTPUT_CSV)
    df_top = df_out.sort_values(by=["r2", "rmse"], ascending=[False, True]).head(10)
    print("\n" + "=" * 70)
    print("TOP 10 EXPERIMENTS BY R²")
    print("=" * 70)
    cols = [
        "train_ratio",
        "epochs",
        "batch_size",
        "learning_rate",
        "optimizer",
        "activation",
        "best_epoch",
        "mse",
        "rmse",
        "mae",
        "r2",
    ]
    print(df_top[cols].to_string(index=False))


if __name__ == "__main__":
    main()
