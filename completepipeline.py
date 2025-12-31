"""complete_pipeline_llama_similarity_models.py

End-to-end pipeline (single script) for:
1) Loading full-text papers from a directory
2) Filtering papers with 2021 citations >= threshold
3) Full Text analysis:
   - preprocess text using the existing NLTK-based pipeline
   - extract Top-N topics (N=5 and N=10) via LLaMA 3.2 (Ollama API)
   - consolidate topics into trending topics with singular/plural merging
4) Deep Text analysis:
   - no preprocessing (raw text)
   - extract Top-N topics (N=5 and N=10) via LLaMA 3.2
   - consolidate topics into trending topics with singular/plural merging
5) Compute a single TF-IDF cosine similarity per paper against ONE combined trending-topics document
   - write 4 similarity CSVs:
       - fulltext_n5, fulltext_n10, deeptext_n5, deeptext_n10
6) Evaluate 20 models (5 architectures × 4 analyses) using hyperparameters from the 4 top_models*.json files
   - produce a table like the provided screenshot
   - save actual-vs-predicted values (test set) per model

Assumptions/notes
- Uses Ollama endpoint at http://localhost:11434/api/generate by default.
- Uses R²*100 as the "Accuracy" field to match the screenshot layout.
- Uses MAPE (mean absolute percentage error)*100 as "PE".

Run:
  python3 complete_pipeline_llama_similarity_models.py \
    --papers-dir /Users/swayamprabha/Downloads/1992 \
    --citations-xlsx /Users/swayamprabha/Downloads/1992.xlsx

You likely want Ollama running:
  ollama serve
  ollama pull llama3.2:latest
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import string
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

# ML deps
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ----------------------------
# Preprocessing (mirrors preprocess.py)
# ----------------------------

def ensure_nltk() -> None:
    import nltk

    required = ["punkt", "stopwords"]
    for item in required:
        try:
            nltk.data.find(f"tokenizers/{item}" if item == "punkt" else f"corpora/{item}")
        except LookupError:
            nltk.download(item, quiet=True)


def preprocess_text_nltk(text: str, *, language: str = "english") -> str:
    if not text or not text.strip():
        return ""

    # Local imports to avoid making nltk a hard import until needed
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize

    ensure_nltk()
    stop = set(stopwords.words(language))

    # clean_text
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text.lower())

    cleaned: List[str] = []
    for tok in tokens:
        if all(c in string.punctuation for c in tok):
            continue
        if tok.isdigit():
            continue
        if tok in stop:
            continue
        cleaned.append(tok)

    return " ".join(cleaned)


# ----------------------------
# Topic consolidation (mirrors consolidate_all_topics.py)
# ----------------------------


def normalize_topic(topic: str) -> str:
    return " ".join(topic.lower().strip().split())


def get_canonical_form(topic: str) -> str:
    topic = topic.lower().strip()

    no_singularize = {
        "mass",
        "process",
        "class",
        "gross",
        "ross",
        "basis",
        "analysis",
        "thesis",
        "genesis",
        "axis",
        "crisis",
        "emphasis",
        "oasis",
        "parenthesis",
        "synopsis",
        "diagnosis",
    }

    if topic in no_singularize:
        return topic

    if topic.endswith("theories"):
        return topic[:-3] + "y"
    elif topic.endswith("ies") and len(topic) > 4 and topic[-4] not in "aeiou":
        return topic[:-3] + "y"
    elif topic.endswith("ices") and len(topic) > 5:
        return topic[:-4] + "ix"
    elif topic.endswith("s") and len(topic) > 4:
        if any(topic.endswith(x) for x in ["ss", "ous", "ius", "sis", "xis"]):
            return topic
        if topic.endswith("as") and " " not in topic:
            return topic
        return topic[:-1]

    return topic


def consolidate_topics_with_counts(
    topics_by_paper: Dict[str, List[str]],
) -> Dict[str, Any]:
    # normalize
    normalized_topics: Dict[str, List[Tuple[str, str]]] = {}
    for paper_id, topics in topics_by_paper.items():
        for t in topics:
            norm = normalize_topic(t)
            normalized_topics.setdefault(norm, []).append((norm, paper_id))

    # merge by canonical
    canonical_to_pairs: Dict[str, List[Tuple[str, str]]] = {}
    for topic, pairs in normalized_topics.items():
        canon = get_canonical_form(topic)
        canonical_to_pairs.setdefault(canon, []).extend(pairs)

    topic_to_papers: Dict[str, set[str]] = {}
    topic_counts: Dict[str, int] = {}
    merge_log: List[Dict[str, Any]] = []

    for canon, pairs in canonical_to_pairs.items():
        topic_to_papers.setdefault(canon, set())
        topic_counts.setdefault(canon, 0)

        original_forms: Dict[str, Dict[str, Any]] = {}
        for orig, pid in pairs:
            original_forms.setdefault(orig, {"count": 0, "papers": set()})
            original_forms[orig]["count"] += 1
            original_forms[orig]["papers"].add(pid)

            topic_to_papers[canon].add(pid)
            topic_counts[canon] += 1

        if len(original_forms) > 1:
            merged_from = []
            for form, d in sorted(original_forms.items()):
                merged_from.append(
                    {
                        "variant": form,
                        "count": d["count"],
                        "papers_mentioning": len(d["papers"]),
                    }
                )
            merge_log.append(
                {
                    "canonical": canon,
                    "total_count": topic_counts[canon],
                    "total_papers_mentioning": len(topic_to_papers[canon]),
                    "merged_from": merged_from,
                }
            )

    consolidated: List[Dict[str, Any]] = []
    for topic, papers in topic_to_papers.items():
        paper_ids = sorted(papers)
        consolidated.append(
            {
                "topic": topic,
                "count": topic_counts[topic],
                "papers_mentioning": len(paper_ids),
                "paper_ids": paper_ids,
            }
        )

    consolidated.sort(key=lambda x: (-x["papers_mentioning"], -x["count"], x["topic"]))

    summary = {
        "total_unique_topics": len(consolidated),
        "total_topic_mentions": sum(x["count"] for x in consolidated),
        "total_papers": len(topics_by_paper),
        "average_topics_per_paper": (sum(x["count"] for x in consolidated) / len(topics_by_paper))
        if topics_by_paper
        else 0.0,
        "topics_merged": len(merge_log),
    }

    return {"summary": summary, "topics": consolidated, "merge_log": merge_log}


# ----------------------------
# LLaMA topic extraction (parameterized Top-N)
# ----------------------------


@dataclass(frozen=True)
class LlamaConfig:
    api_url: str
    model: str
    temperature: float
    max_tokens: int
    max_retry_attempts: int


class LlamaTopicExtractor:
    def __init__(self, cfg: LlamaConfig, *, min_words: int = 1, max_words: int = 6):
        self.cfg = cfg
        self.min_words = min_words
        self.max_words = max_words

    def _build_prompt(self, paper_text: str, *, n_topics: int) -> str:
        return f"""You are a scientific topic extraction specialist. Analyze the following research paper and extract EXACTLY {n_topics} trending technical/physics topics.

STRICT REQUIREMENTS:
- Extract EXACTLY {n_topics} topics, no more, no less
- Each topic must be {self.min_words}-{self.max_words} words
- Topics must be technical/physics-specific (equations, laws, effects, models, algorithms, mathematical formulations)
- NO generic terms like \"analysis\", \"study\", \"method\"
- NO full sentences
- NO explanations

OUTPUT FORMAT (JSON only):
{{
  \"topics\": [""" + ", ".join([f"\"topic{i+1}\"" for i in range(n_topics)]) + "]
}}

PAPER TEXT:
{paper_text[:8000]}

Return ONLY valid JSON with exactly {n_topics} topics."""

    def _call_llama(self, prompt: str) -> Optional[str]:
        payload = {
            "model": self.cfg.model,
            "prompt": prompt,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
            "stream": False,
            "format": "json",
        }
        try:
            resp = requests.post(self.cfg.api_url, json=payload, timeout=180)
            if resp.status_code != 200:
                return None
            data = resp.json()
            return data.get("response", "")
        except Exception:
            return None

    def _parse_topics(self, response_text: str) -> Optional[List[str]]:
        try:
            data = json.loads(response_text)
            topics = data.get("topics")
            if isinstance(topics, list) and all(isinstance(t, str) for t in topics):
                return topics
        except Exception:
            pass

        # fallback: extract quoted strings
        try:
            matches = re.findall(r'"([^"]+)"', response_text)
            if matches:
                return matches
        except Exception:
            return None
        return None

    def _validate_topics(self, topics: List[str], *, n_topics: int) -> bool:
        if len(topics) != n_topics:
            return False

        for t in topics:
            wc = len(t.strip().split())
            if wc < self.min_words or wc > self.max_words:
                return False
        return True

    def extract_topics(self, paper_id: str, paper_text: str, *, n_topics: int) -> Optional[List[str]]:
        for attempt in range(1, self.cfg.max_retry_attempts + 1):
            prompt = self._build_prompt(paper_text, n_topics=n_topics)
            resp = self._call_llama(prompt)
            if not resp:
                time.sleep(2)
                continue
            topics = self._parse_topics(resp)
            if not topics:
                time.sleep(2)
                continue
            topics = [t.strip() for t in topics[:n_topics]]
            if self._validate_topics(topics, n_topics=n_topics):
                return topics
            time.sleep(2)
        return None


# ----------------------------
# Similarity (single combined topic-doc vs each paper)
# ----------------------------


def compute_single_tfidf_similarity(
    *,
    paper_texts: Dict[str, str],
    trending_topics: List[str],
) -> Dict[str, float]:
    combined_topic_doc = " ".join(trending_topics)
    ids = sorted(paper_texts.keys())
    docs = [paper_texts[i] for i in ids]

    all_docs = [combined_topic_doc] + docs

    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
    )
    tfidf = vectorizer.fit_transform(all_docs)
    topics_vec = tfidf[0:1]
    papers_vec = tfidf[1:]

    sims = cosine_similarity(papers_vec, topics_vec).ravel()
    return {pid: float(s) for pid, s in zip(ids, sims)}


def write_similarity_csv(path: Path, sims: Dict[str, float], *, as_percent: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["paper_id", "cosine_tfidf_similarity"])
        for pid in sorted(sims.keys()):
            v = sims[pid] * 100.0 if as_percent else sims[pid]
            w.writerow([pid, f"{v:.6f}"])


# ----------------------------
# Model evaluation utilities (based on tuning scripts)
# ----------------------------


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


def standardize_train_only(x_train: np.ndarray, x_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return (x_train - mean) / std, (x_test - mean) / std


def make_split_with_ids(
    df: pd.DataFrame, train_ratio: float, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    n = len(df)
    idx = np.arange(n)
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)

    n_train = int(round(train_ratio * n))
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]

    X = df[["cites_2021", "cosine_tfidf_similarity"]].to_numpy(dtype=np.float32)
    y = df[["cites_2022"]].to_numpy(dtype=np.float32)
    paper_ids = df.index.to_numpy(dtype=str)
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx], paper_ids[test_idx].tolist()


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(math.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    # PE: MAPE (%). Guard against division by zero.
    denom = np.where(y_true == 0, np.nan, y_true)
    mape = float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100.0)

    # Accuracy: use R^2 * 100 to match table style.
    acc = float(max(0.0, min(100.0, r2 * 100.0))) if not math.isnan(r2) else float("nan")

    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "accuracy": acc, "pe": mape}


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


class ANNRegressor(nn.Module):
    def __init__(self, hidden_sizes: List[int], activation: str, dropout: float):
        super().__init__()
        act = _activation(activation)
        layers: List[nn.Module] = []
        in_dim = 2
        for h in hidden_sizes:
            layers.extend([nn.Linear(in_dim, h), nn.LayerNorm(h), act, nn.Dropout(p=dropout)])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LSTMRegressor(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, dropout: float, head_activation: str = "relu"):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(nn.LayerNorm(hidden_size), _activation(head_activation), nn.Linear(hidden_size, 1))

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x_seq)
        last = out[:, -1, :]
        return self.head(last)


class MLPRegressor(nn.Module):
    def __init__(self, activation: str, dropout: float = 0.2):
        super().__init__()
        act = _activation(activation)
        self.net = nn.Sequential(
            nn.Linear(2, 128),
            nn.LayerNorm(128),
            act,
            nn.Dropout(p=dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            act,
            nn.Dropout(p=dropout),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            act,
            nn.Dropout(p=dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNNRegressor(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dropout: float, head_activation: str):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=kernel_size, padding="same"),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding="same"),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), _activation(head_activation), nn.Linear(channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.feat(x))


class RNNRegressor(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, dropout: float, head_activation: str):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity="tanh",
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(nn.LayerNorm(hidden_size), _activation(head_activation), nn.Linear(hidden_size, 1))

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x_seq)
        last = out[:, -1, :]
        return self.head(last)


def train_predict(
    *,
    model: nn.Module,
    device: torch.device,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    optimizer: str,
    patience: int = 10,
) -> Tuple[Dict[str, float], int, np.ndarray]:
    X_train_z, X_test_z = standardize_train_only(X_train, X_test)

    Xtr = torch.tensor(X_train_z, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.float32)
    Xte = torch.tensor(X_test_z, dtype=torch.float32)
    yte = torch.tensor(y_test, dtype=torch.float32)

    # reshape for sequence models
    if isinstance(model, CNNRegressor):
        Xtr = Xtr.view(-1, 1, 2)
        Xte = Xte.view(-1, 1, 2)
    elif isinstance(model, (RNNRegressor, LSTMRegressor)):
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

        if no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        y_pred = model(Xte.to(device)).detach().cpu().numpy().reshape(-1)

    metrics = regression_metrics(y_test.reshape(-1), y_pred)
    return metrics, best_epoch, y_pred


def build_model_from_hparams(model_type: str, hp: Dict[str, Any]) -> nn.Module:
    t = model_type.lower()
    if t == "ann":
        return ANNRegressor(hidden_sizes=list(hp["hidden_sizes"]), activation=str(hp["activation"]), dropout=float(hp.get("dropout", 0.2)))
    if t == "mlp":
        return MLPRegressor(activation=str(hp["activation"]), dropout=float(hp.get("dropout", 0.2)))
    if t == "cnn":
        return CNNRegressor(
            channels=int(hp["channels"]),
            kernel_size=int(hp["kernel_size"]),
            dropout=float(hp.get("dropout", 0.2)),
            head_activation=str(hp.get("head_activation", "relu")),
        )
    if t == "rnn":
        return RNNRegressor(
            hidden_size=int(hp["hidden_size"]),
            num_layers=int(hp["num_layers"]),
            dropout=float(hp.get("dropout", 0.2)),
            head_activation=str(hp.get("head_activation", "relu")),
        )
    if t == "lstm":
        return LSTMRegressor(
            hidden_size=int(hp["hidden_size"]),
            num_layers=int(hp["num_layers"]),
            dropout=float(hp.get("dropout", 0.2)),
            head_activation=str(hp.get("head_activation", "relu")),
        )
    raise ValueError(f"Unknown model_type: {model_type}")


def load_and_merge_dataset(
    *,
    excel_path: Path,
    sim_csv_path: Path,
    paper_ids_filter: Optional[set[str]],
) -> pd.DataFrame:
    df_xl = pd.read_excel(excel_path, sheet_name=0)
    df_xl = df_xl.loc[:, [c for c in df_xl.columns if not str(c).startswith("Unnamed")]]

    if "Article Id" not in df_xl.columns:
        raise ValueError("Excel missing required column: 'Article Id'")
    if 2021 not in df_xl.columns or 2022 not in df_xl.columns:
        raise ValueError("Excel missing required year columns: 2021 and/or 2022")

    df_xl = df_xl[["Article Id", 2021, 2022]].rename(columns={"Article Id": "paper_id", 2021: "cites_2021", 2022: "cites_2022"})
    df_xl["paper_id"] = df_xl["paper_id"].astype(str)

    df_sim = pd.read_csv(sim_csv_path)
    expected_cols = {"paper_id", "cosine_tfidf_similarity"}
    if not expected_cols.issubset(df_sim.columns):
        raise ValueError(f"Similarity CSV must contain columns {expected_cols}, got {list(df_sim.columns)}")
    df_sim["paper_id"] = df_sim["paper_id"].astype(str)

    df = df_xl.merge(df_sim, on="paper_id", how="inner")

    if paper_ids_filter is not None:
        df = df[df["paper_id"].isin(paper_ids_filter)].copy()

    df["cites_2021"] = pd.to_numeric(df["cites_2021"], errors="coerce")
    df["cites_2022"] = pd.to_numeric(df["cites_2022"], errors="coerce")
    df["cosine_tfidf_similarity"] = pd.to_numeric(df["cosine_tfidf_similarity"], errors="coerce")

    df = df.dropna(subset=["cites_2021", "cosine_tfidf_similarity", "cites_2022"]).copy()
    df = df[(df["cites_2021"] >= 0) & (df["cites_2022"] >= 0)].copy()

    return df.set_index("paper_id").sort_index()


# ----------------------------
# Orchestration
# ----------------------------


def load_citation_filtered_ids(
    *,
    citations_xlsx: Path,
    year: int,
    threshold: int,
) -> set[str]:
    df = pd.read_excel(citations_xlsx)

    year_col = None
    for col in df.columns:
        if str(year) in str(col):
            year_col = col
            break
    if year_col is None:
        raise ValueError(f"Could not find citation column containing '{year}'")

    id_col = df.columns[0]
    filtered = df[df[year_col] >= threshold]
    return set(str(pid).strip() for pid in filtered[id_col].values)


def read_papers_texts(papers_dir: Path, paper_ids: set[str]) -> Dict[str, str]:
    texts: Dict[str, str] = {}
    for p in sorted(papers_dir.glob("*.txt")):
        pid = p.stem
        if pid not in paper_ids:
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        if txt.strip():
            texts[pid] = txt
    return texts


def extract_topics_for_corpus(
    *,
    extractor: LlamaTopicExtractor,
    paper_texts: Dict[str, str],
    n_topics: int,
    out_dir: Path,
    skip_existing: bool,
) -> Dict[str, List[str]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_topics_path = out_dir / "all_topics.json"

    if skip_existing and all_topics_path.exists():
        return json.loads(all_topics_path.read_text(encoding="utf-8"))

    all_topics: Dict[str, List[str]] = {}

    for i, (pid, text) in enumerate(sorted(paper_texts.items()), 1):
        per_paper_path = out_dir / f"{pid}_topics.json"
        if skip_existing and per_paper_path.exists():
            try:
                data = json.loads(per_paper_path.read_text(encoding="utf-8"))
                topics = data.get("topics")
                if isinstance(topics, list) and len(topics) == n_topics:
                    all_topics[pid] = topics
                    continue
            except Exception:
                pass

        topics = extractor.extract_topics(pid, text, n_topics=n_topics)
        if not topics:
            continue
        all_topics[pid] = topics
        per_paper_path.write_text(
            json.dumps({"paper_id": pid, "topics": topics, "topic_count": len(topics)}, indent=2),
            encoding="utf-8",
        )

    all_topics_path.write_text(json.dumps(all_topics, indent=2), encoding="utf-8")
    return all_topics


def write_predictions_csv(
    out_path: Path,
    test_ids: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["paper_id", "y_true", "y_pred"])
        for pid, yt, yp in zip(test_ids, y_true.reshape(-1), y_pred.reshape(-1)):
            w.writerow([pid, f"{float(yt):.6f}", f"{float(yp):.6f}"])


def evaluate_models_for_analysis(
    *,
    analysis_name: str,
    sim_csv: Path,
    top_models_json: Path,
    excel_path: Path,
    paper_ids_filter: set[str],
    out_dir: Path,
) -> Dict[str, Dict[str, float]]:
    data = json.loads(top_models_json.read_text(encoding="utf-8"))
    best = data.get("best", {})

    df = load_and_merge_dataset(excel_path=excel_path, sim_csv_path=sim_csv, paper_ids_filter=paper_ids_filter)

    device = get_device()

    metrics_by_model: Dict[str, Dict[str, float]] = {}

    for model_type in ["ann", "lstm", "rnn", "mlp", "cnn"]:
        entry = best.get(model_type)
        if not entry:
            continue
        hp = entry.get("hyperparameters", {})

        seed = int(hp.get("seed", 42))
        train_ratio = float(hp.get("train_ratio", 0.8))
        epochs = int(hp.get("epochs", 30))
        batch_size = int(hp.get("batch_size", 32))
        lr = float(hp.get("learning_rate", hp.get("lr", 1e-3)))
        optimizer = str(hp.get("optimizer", "adam"))

        set_seed(seed)
        X_train, X_test, y_train, y_test, test_ids = make_split_with_ids(df, train_ratio=train_ratio, seed=seed)

        model = build_model_from_hparams(model_type, hp)
        m, best_epoch, y_pred = train_predict(
            model=model,
            device=device,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            optimizer=optimizer,
            patience=10,
        )

        # Persist per-model artifacts
        model_out_dir = out_dir / analysis_name / model_type
        model_out_dir.mkdir(parents=True, exist_ok=True)

        cfg_and_metrics = {
            "analysis": analysis_name,
            "model_type": model_type,
            "hyperparameters": hp,
            "metrics": m,
            "best_epoch": best_epoch,
            "n_rows": int(len(df)),
            "test_size": int(len(test_ids)),
            "similarity_csv": str(sim_csv),
        }
        (model_out_dir / "config_and_metrics.json").write_text(json.dumps(cfg_and_metrics, indent=2), encoding="utf-8")

        write_predictions_csv(model_out_dir / "test_predictions.csv", test_ids, y_test, y_pred)

        metrics_by_model[model_type] = m

    return metrics_by_model


def format_table(metrics: Dict[str, Dict[str, Dict[str, float]]]) -> List[Dict[str, Any]]:
    # Output rows compatible with the screenshot layout.
    # metrics: analysis -> model_type -> metric dict
    rows: List[Dict[str, Any]] = []

    def row(analysis: str, model: str, m: Dict[str, float]) -> Dict[str, Any]:
        return {
            "Analysis": analysis,
            "Algorithm": model.upper(),
            "Accuracy": round(float(m.get("accuracy", float("nan"))), 2),
            "MAE": round(float(m.get("mae", float("nan"))), 4),
            "MSE": round(float(m.get("mse", float("nan"))), 4),
            "PE": int(round(float(m.get("pe", float("nan"))))) if not math.isnan(float(m.get("pe", float("nan")))) else None,
        }

    for analysis in [
        "fulltext_n5",
        "fulltext_n10",
        "deeptext_n5",
        "deeptext_n10",
    ]:
        by_model = metrics.get(analysis, {})
        for model in ["ann", "lstm", "rnn", "mlp", "cnn"]:
            if model in by_model:
                rows.append(row(analysis, model, by_model[model]))

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--papers-dir", type=str, required=True)
    parser.add_argument("--citations-xlsx", type=str, required=True)
    parser.add_argument("--year", type=int, default=2021)
    parser.add_argument("--threshold", type=int, default=20)

    parser.add_argument("--llama-api-url", type=str, default="http://localhost:11434/api/generate")
    parser.add_argument("--llama-model", type=str, default="llama3.2:latest")
    parser.add_argument("--llama-temperature", type=float, default=0.3)
    parser.add_argument("--llama-max-tokens", type=int, default=200)
    parser.add_argument("--llama-retries", type=int, default=3)

    parser.add_argument("--skip-existing", action="store_true", default=True)

    parser.add_argument("--out-root", type=str, default=str(Path("pipeline_outputs")))

    # Top models files
    parser.add_argument("--top-fulltext-n5", type=str, default="top_models_similarity1992fulltext_5.json")
    parser.add_argument("--top-fulltext-n10", type=str, default="top_models_similarity1992fulltext_10.json")
    parser.add_argument("--top-deeptext-n5", type=str, default="top_models_deeptext_n5.json")
    parser.add_argument("--top-deeptext-n10", type=str, default="top_models_deeptext_n10.json")

    args = parser.parse_args()

    papers_dir = Path(args.papers_dir)
    citations_xlsx = Path(args.citations_xlsx)
    out_root = Path(args.out_root)

    out_root.mkdir(parents=True, exist_ok=True)

    # 1) Filter papers
    paper_ids = load_citation_filtered_ids(citations_xlsx=citations_xlsx, year=args.year, threshold=args.threshold)

    raw_texts = read_papers_texts(papers_dir, paper_ids)
    if not raw_texts:
        raise SystemExit("No papers found after filtering by citations + .txt presence")

    # 2) Build corpora
    fulltext_preprocessed = {pid: preprocess_text_nltk(txt) for pid, txt in raw_texts.items()}
    deeptext_raw = raw_texts

    # 3) Extract topics
    llama_cfg = LlamaConfig(
        api_url=args.llama_api_url,
        model=args.llama_model,
        temperature=args.llama_temperature,
        max_tokens=args.llama_max_tokens,
        max_retry_attempts=args.llama_retries,
    )
    extractor = LlamaTopicExtractor(llama_cfg)

    topics_dir = out_root / f"topics_{args.year}_c{args.threshold}"

    # Fulltext topics
    ft5 = extract_topics_for_corpus(
        extractor=extractor,
        paper_texts=fulltext_preprocessed,
        n_topics=5,
        out_dir=topics_dir / "fulltext_n5",
        skip_existing=args.skip_existing,
    )
    ft10 = extract_topics_for_corpus(
        extractor=extractor,
        paper_texts=fulltext_preprocessed,
        n_topics=10,
        out_dir=topics_dir / "fulltext_n10",
        skip_existing=args.skip_existing,
    )

    # Deeptext topics
    dt5 = extract_topics_for_corpus(
        extractor=extractor,
        paper_texts=deeptext_raw,
        n_topics=5,
        out_dir=topics_dir / "deeptext_n5",
        skip_existing=args.skip_existing,
    )
    dt10 = extract_topics_for_corpus(
        extractor=extractor,
        paper_texts=deeptext_raw,
        n_topics=10,
        out_dir=topics_dir / "deeptext_n10",
        skip_existing=args.skip_existing,
    )

    # 4) Consolidate trending topics
    cons_dir = out_root / f"consolidated_{args.year}_c{args.threshold}"
    cons_dir.mkdir(parents=True, exist_ok=True)

    ft5_cons = consolidate_topics_with_counts(ft5)
    ft10_cons = consolidate_topics_with_counts(ft10)
    dt5_cons = consolidate_topics_with_counts(dt5)
    dt10_cons = consolidate_topics_with_counts(dt10)

    (cons_dir / "consolidated_topics_fulltext_n5.json").write_text(json.dumps(ft5_cons, indent=2), encoding="utf-8")
    (cons_dir / "consolidated_topics_fulltext_n10.json").write_text(json.dumps(ft10_cons, indent=2), encoding="utf-8")
    (cons_dir / "consolidated_topics_deeptext_n5.json").write_text(json.dumps(dt5_cons, indent=2), encoding="utf-8")
    (cons_dir / "consolidated_topics_deeptext_n10.json").write_text(json.dumps(dt10_cons, indent=2), encoding="utf-8")

    # 5) Similarities (single scalar per paper)
    sim_dir = out_root / f"similarities_{args.year}_c{args.threshold}"
    sim_dir.mkdir(parents=True, exist_ok=True)

    ft5_topics = [t["topic"] for t in ft5_cons["topics"]]
    ft10_topics = [t["topic"] for t in ft10_cons["topics"]]
    dt5_topics = [t["topic"] for t in dt5_cons["topics"]]
    dt10_topics = [t["topic"] for t in dt10_cons["topics"]]

    sims_ft5 = compute_single_tfidf_similarity(paper_texts=fulltext_preprocessed, trending_topics=ft5_topics)
    sims_ft10 = compute_single_tfidf_similarity(paper_texts=fulltext_preprocessed, trending_topics=ft10_topics)
    sims_dt5 = compute_single_tfidf_similarity(paper_texts=deeptext_raw, trending_topics=dt5_topics)
    sims_dt10 = compute_single_tfidf_similarity(paper_texts=deeptext_raw, trending_topics=dt10_topics)

    sim_ft5_csv = sim_dir / "similarity_fulltext_n5.csv"
    sim_ft10_csv = sim_dir / "similarity_fulltext_n10.csv"
    sim_dt5_csv = sim_dir / "similarity_deeptext_n5.csv"
    sim_dt10_csv = sim_dir / "similarity_deeptext_n10.csv"

    write_similarity_csv(sim_ft5_csv, sims_ft5, as_percent=True)
    write_similarity_csv(sim_ft10_csv, sims_ft10, as_percent=True)
    write_similarity_csv(sim_dt5_csv, sims_dt5, as_percent=True)
    write_similarity_csv(sim_dt10_csv, sims_dt10, as_percent=True)

    # 6) Evaluate 20 models
    eval_dir = out_root / f"model_eval_{args.year}_c{args.threshold}"

    metrics_all: Dict[str, Dict[str, Dict[str, float]]] = {}

    metrics_all["fulltext_n5"] = evaluate_models_for_analysis(
        analysis_name="fulltext_n5",
        sim_csv=sim_ft5_csv,
        top_models_json=Path(args.top_fulltext_n5),
        excel_path=citations_xlsx,
        paper_ids_filter=set(raw_texts.keys()),
        out_dir=eval_dir,
    )
    metrics_all["fulltext_n10"] = evaluate_models_for_analysis(
        analysis_name="fulltext_n10",
        sim_csv=sim_ft10_csv,
        top_models_json=Path(args.top_fulltext_n10),
        excel_path=citations_xlsx,
        paper_ids_filter=set(raw_texts.keys()),
        out_dir=eval_dir,
    )
    metrics_all["deeptext_n5"] = evaluate_models_for_analysis(
        analysis_name="deeptext_n5",
        sim_csv=sim_dt5_csv,
        top_models_json=Path(args.top_deeptext_n5),
        excel_path=citations_xlsx,
        paper_ids_filter=set(raw_texts.keys()),
        out_dir=eval_dir,
    )
    metrics_all["deeptext_n10"] = evaluate_models_for_analysis(
        analysis_name="deeptext_n10",
        sim_csv=sim_dt10_csv,
        top_models_json=Path(args.top_deeptext_n10),
        excel_path=citations_xlsx,
        paper_ids_filter=set(raw_texts.keys()),
        out_dir=eval_dir,
    )

    # 7) Write table outputs
    table_rows = format_table(metrics_all)
    table_json = out_root / f"results_table_{args.year}_c{args.threshold}.json"
    table_csv = out_root / f"results_table_{args.year}_c{args.threshold}.csv"

    table_json.write_text(json.dumps({"rows": table_rows}, indent=2), encoding="utf-8")

    with table_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["Analysis", "Algorithm", "Accuracy", "MAE", "MSE", "PE"])
        w.writeheader()
        w.writerows(table_rows)

    print(f"Done. Similarities in: {sim_dir}")
    print(f"Done. Model eval in: {eval_dir}")
    print(f"Table: {table_csv}")


if __name__ == "__main__":
    main()
