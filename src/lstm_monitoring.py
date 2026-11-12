"""
LSTM Autoencoder Monitoring Agent for NASA C-MAPSS turbofan data.

SCOPE: anomaly MONITORING only. The model learns what "healthy early-life"
sensor behaviour looks like and flags departures from it. It does NOT perform
fault diagnosis (it cannot name a failed component or failure mode). The
per-feature reconstruction error is an explanation of *which signals deviate
most*, not a root-cause diagnosis.

Supports PyTorch with automatic, deterministic, vectorized pure-NumPy/SciPy
fallback when PyTorch native libraries are restricted or unavailable.
"""
from __future__ import annotations

import copy
import json
import logging
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

try:
    from sklearn.preprocessing import StandardScaler as _SklearnStandardScaler
except ImportError:  # pragma: no cover
    _SklearnStandardScaler = None

# Attempt torch import safely without crashing on OS App Control / DLL restrictions
TORCH_AVAILABLE = False
TORCH_IMPORT_ERROR: str | None = None
torch = None
nn = None
try:
    import torch as _torch
    import torch.nn as _nn
    from torch.utils.data import DataLoader as _DataLoader, TensorDataset as _TensorDataset
    # Test tensor creation to verify native DLLs loaded successfully
    _t = _torch.zeros(1)
    torch = _torch
    nn = _nn
    DataLoader = _DataLoader
    TensorDataset = _TensorDataset
    TORCH_AVAILABLE = True
except Exception as exc:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    TORCH_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

LOGGER = logging.getLogger("lstm_monitoring")

# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------
UNIT_COL = "unit_id"
CYCLE_COL = "cycle"
SETTING_COLS = [f"op_setting_{i}" for i in range(1, 4)]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
CMAPSS_COLUMNS = [UNIT_COL, CYCLE_COL] + SETTING_COLS + SENSOR_COLS  # 26 columns
CANDIDATE_FEATURES = SETTING_COLS + SENSOR_COLS


@dataclass
class MonitoringConfig:
    """All tunable hyper-parameters in one place (serialised into the checkpoint)."""

    window_size: int = 30          # cycles per window
    stride: int = 1                # step between consecutive windows
    healthy_cycles: int = 60       # early-life cycles assumed healthy (configurable)
    val_fraction: float = 0.2      # fraction of ENGINES (not rows) held out
    seed: int = 42
    hidden_size: int = 64          # encoder/decoder LSTM hidden size
    latent_size: int = 32          # bottleneck size
    num_layers: int = 1
    learning_rate: float = 1e-3
    batch_size: int = 128
    max_epochs: int = 50
    patience: int = 10             # early-stopping patience (epochs)
    threshold_percentile: float = 99.0
    persistence: int = 3           # consecutive alert windows needed to confirm
    degraded_rul: int = 50         # windows with RUL <= this are "degraded" for detection rate
    constant_std_tol: float = 1e-8
    top_k_features: int = 3

    def to_dict(self) -> Dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# 2. Data loading
# ---------------------------------------------------------------------------
def load_cmapss_file(path: Path | str) -> pd.DataFrame:
    """Parse a whitespace-separated C-MAPSS train/test file into 26 named columns."""
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.dropna(axis=1, how="all")
    if df.shape[1] != len(CMAPSS_COLUMNS):
        raise ValueError(f"{path}: expected 26 columns, found {df.shape[1]}")
    df.columns = CMAPSS_COLUMNS
    df[UNIT_COL] = df[UNIT_COL].astype(int)
    df[CYCLE_COL] = df[CYCLE_COL].astype(int)
    return df.sort_values([UNIT_COL, CYCLE_COL]).reset_index(drop=True)


def load_rul_file(path: Path | str) -> pd.Series:
    """Official RUL file: row i holds the true remaining life of test unit i+1."""
    rul = pd.read_csv(path, sep=r"\s+", header=None).dropna(axis=1, how="all").iloc[:, 0]
    return pd.Series(rul.to_numpy(dtype=int), index=np.arange(1, len(rul) + 1), name="rul")


# ---------------------------------------------------------------------------
# 3. Leakage-safe preprocessing
# ---------------------------------------------------------------------------
def split_units(df: pd.DataFrame, val_fraction: float, seed: int) -> Tuple[List[int], List[int]]:
    """Split by ENGINE UNIT.

    LEAKAGE CONTROL: rows from one engine are strongly autocorrelated.
    Whole engines are assigned to exactly one side.
    """
    units = np.sort(df[UNIT_COL].unique())
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(units)
    n_val = int(round(len(units) * val_fraction))
    val_units = sorted(int(u) for u in shuffled[:n_val])
    train_units = sorted(int(u) for u in shuffled[n_val:])
    assert not set(train_units) & set(val_units), "unit overlap between splits"
    return train_units, val_units


def find_constant_features(train_df: pd.DataFrame, tol: float) -> Tuple[List[str], List[str]]:
    """Identify constant columns using the TRAINING engines only."""
    stds = train_df[CANDIDATE_FEATURES].std(ddof=0)
    nunique = train_df[CANDIDATE_FEATURES].nunique()
    dropped = [c for c in CANDIDATE_FEATURES if nunique[c] <= 1 or stds[c] <= tol]
    kept = [c for c in CANDIDATE_FEATURES if c not in dropped]
    return kept, dropped


class FittedScaler:
    """Minimal serialisable z-score scaler (same semantics as sklearn StandardScaler)."""

    def __init__(self, mean: np.ndarray, scale: np.ndarray, backend: str):
        self.mean_ = np.asarray(mean, dtype=np.float64)
        self.scale_ = np.asarray(scale, dtype=np.float64)
        self.backend = backend

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean_) / self.scale_).astype(np.float32)

    def state_dict(self) -> Dict:
        return {"mean": self.mean_.tolist(), "scale": self.scale_.tolist(), "backend": self.backend}

    @classmethod
    def from_state_dict(cls, state: Dict) -> "FittedScaler":
        return cls(np.array(state["mean"]), np.array(state["scale"]), state.get("backend", "unknown"))


def fit_scaler(healthy_train_values: np.ndarray) -> FittedScaler:
    """Fit a StandardScaler on HEALTHY TRAINING rows only."""
    if _SklearnStandardScaler is not None:
        sk = _SklearnStandardScaler().fit(healthy_train_values)
        return FittedScaler(sk.mean_, sk.scale_, backend="sklearn.StandardScaler")
    mean = healthy_train_values.mean(axis=0)
    scale = healthy_train_values.std(axis=0, ddof=0)
    scale = np.where(scale < np.finfo(np.float64).eps, 1.0, scale)
    return FittedScaler(mean, scale, backend="numpy (StandardScaler-equivalent)")


def scale_frame(df: pd.DataFrame, features: Sequence[str], scaler: FittedScaler) -> pd.DataFrame:
    out = df[[UNIT_COL, CYCLE_COL]].copy()
    out[list(features)] = scaler.transform(df[list(features)].to_numpy(dtype=np.float64))
    return out


# ---------------------------------------------------------------------------
# 4. Windowing
# ---------------------------------------------------------------------------
def make_windows(
    df: pd.DataFrame, features: Sequence[str], window_size: int, stride: int = 1
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Build chronological sliding windows PER ENGINE (never across engines)."""
    windows, meta = [], []
    for unit, g in df.groupby(UNIT_COL, sort=True):
        g = g.sort_values(CYCLE_COL)
        values = g[list(features)].to_numpy(dtype=np.float32)
        cycles = g[CYCLE_COL].to_numpy()
        if len(g) < window_size:
            LOGGER.warning("unit %s has %d cycles < window %d; skipped", unit, len(g), window_size)
            continue
        view = np.lib.stride_tricks.sliding_window_view(values, window_size, axis=0)
        view = view.transpose(0, 2, 1)[::stride]
        starts = np.arange(0, len(g) - window_size + 1)[::stride]
        windows.append(np.ascontiguousarray(view))
        meta.append(pd.DataFrame({
            UNIT_COL: int(unit),
            "window_start_cycle": cycles[starts],
            "window_end_cycle": cycles[starts + window_size - 1],
        }))
    if not windows:
        return np.empty((0, window_size, len(features)), np.float32), pd.DataFrame(
            columns=[UNIT_COL, "window_start_cycle", "window_end_cycle"])
    return np.concatenate(windows), pd.concat(meta, ignore_index=True)


def healthy_window_mask(meta: pd.DataFrame, healthy_cycles: int) -> np.ndarray:
    """A window is healthy only if ALL of its cycles lie inside the early-life period."""
    return (meta["window_end_cycle"] <= healthy_cycles).to_numpy()


# ---------------------------------------------------------------------------
# 5. Pure-NumPy Vectorized LSTM Engine (resilient, deterministic, no DLL blocks)
# ---------------------------------------------------------------------------
class NumpyLSTMAutoencoder:
    """Vectorized sequence-to-sequence LSTM autoencoder with exact BPTT and Adam."""

    def __init__(self, n_features: int, hidden_size: int = 64, latent_size: int = 32, seed: int = 42):
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.seed = seed
        rng = np.random.default_rng(seed)

        # Xavier initialization
        std_enc = np.float32(np.sqrt(2.0 / (n_features + hidden_size)))
        std_dec = np.float32(np.sqrt(2.0 / (latent_size + hidden_size)))
        std_lat = np.float32(np.sqrt(2.0 / (hidden_size + latent_size)))
        std_out = np.float32(np.sqrt(2.0 / (hidden_size + n_features)))

        self.params: Dict[str, np.ndarray] = {
            "W_enc": rng.normal(0, std_enc, (n_features, 4 * hidden_size)).astype(np.float32),
            "U_enc": rng.normal(0, std_enc, (hidden_size, 4 * hidden_size)).astype(np.float32),
            "b_enc": np.zeros(4 * hidden_size, dtype=np.float32),
            "W_z": rng.normal(0, std_lat, (hidden_size, latent_size)).astype(np.float32),
            "b_z": np.zeros(latent_size, dtype=np.float32),
            "W_dec": rng.normal(0, std_dec, (latent_size, 4 * hidden_size)).astype(np.float32),
            "U_dec": rng.normal(0, std_dec, (hidden_size, 4 * hidden_size)).astype(np.float32),
            "b_dec": np.zeros(4 * hidden_size, dtype=np.float32),
            "W_out": rng.normal(0, std_out, (hidden_size, n_features)).astype(np.float32),
            "b_out": np.zeros(n_features, dtype=np.float32),
        }
        # Initialize forget gates bias to 1.0
        self.params["b_enc"][hidden_size : 2 * hidden_size] = 1.0
        self.params["b_dec"][hidden_size : 2 * hidden_size] = 1.0

        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.t_step = 0

    @staticmethod
    def _lstm_step(x: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray,
                   W: np.ndarray, U: np.ndarray, b: np.ndarray, H: int):
        gates = x @ W + h_prev @ U + b
        i = 1.0 / (1.0 + np.exp(-np.clip(gates[:, :H], -15.0, 15.0)))
        f = 1.0 / (1.0 + np.exp(-np.clip(gates[:, H : 2 * H], -15.0, 15.0)))
        o = 1.0 / (1.0 + np.exp(-np.clip(gates[:, 2 * H : 3 * H], -15.0, 15.0)))
        ct = np.tanh(gates[:, 3 * H :])
        c = f * c_prev + i * ct
        tc = np.tanh(c)
        h = o * tc
        return h, c, (x, h_prev, c_prev, i, f, o, ct, tc, c)

    @staticmethod
    def _lstm_bstep(dh: np.ndarray, dc_next: np.ndarray, cache,
                    W: np.ndarray, U: np.ndarray, H: int):
        x, h_prev, c_prev, i, f, o, ct, tc, c = cache
        dc = dh * o * (1.0 - tc ** 2) + dc_next
        do = dh * tc * o * (1.0 - o)
        dct = dc * i * (1.0 - ct ** 2)
        di = dc * ct * i * (1.0 - i)
        df = dc * c_prev * f * (1.0 - f)
        dc_prev = dc * f

        dgates = np.concatenate([di, df, do, dct], axis=1)
        dW = x.T @ dgates
        dU = h_prev.T @ dgates
        db = np.sum(dgates, axis=0)
        dx = dgates @ W.T
        dh_prev = dgates @ U.T
        return dx, dh_prev, dc_prev, dW, dU, db

    def predict(self, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
        """Reconstruct sequence windows: (N, T, F) -> (N, T, F)."""
        reconstructed = []
        H = self.hidden_size
        for start in range(0, len(X), batch_size):
            batch = X[start : start + batch_size]
            B, T, D = batch.shape
            h = np.zeros((B, H), dtype=np.float32)
            c = np.zeros((B, H), dtype=np.float32)
            for t in range(T):
                h, c, _ = self._lstm_step(batch[:, t], h, c, self.params["W_enc"], self.params["U_enc"], self.params["b_enc"], H)
            z = h @ self.params["W_z"] + self.params["b_z"]

            h_d = np.zeros((B, H), dtype=np.float32)
            c_d = np.zeros((B, H), dtype=np.float32)
            out_batch = np.zeros((B, T, D), dtype=np.float32)
            for t in range(T):
                h_d, c_d, _ = self._lstm_step(z, h_d, c_d, self.params["W_dec"], self.params["U_dec"], self.params["b_dec"], H)
                out_batch[:, t] = h_d @ self.params["W_out"] + self.params["b_out"]
            reconstructed.append(out_batch)
        return np.concatenate(reconstructed, axis=0) if reconstructed else np.empty_like(X)

    def train_step(self, X: np.ndarray, lr: float = 1e-3, clip_norm: float = 5.0) -> float:
        B, T, D = X.shape
        H = self.hidden_size

        # Forward
        h = np.zeros((B, H), dtype=np.float32)
        c = np.zeros((B, H), dtype=np.float32)
        enc_cache = []
        for t in range(T):
            h, c, cache = self._lstm_step(X[:, t], h, c, self.params["W_enc"], self.params["U_enc"], self.params["b_enc"], H)
            enc_cache.append(cache)
        z = h @ self.params["W_z"] + self.params["b_z"]

        h_d = np.zeros((B, H), dtype=np.float32)
        c_d = np.zeros((B, H), dtype=np.float32)
        dec_cache = []
        out = np.zeros((B, T, D), dtype=np.float32)
        for t in range(T):
            h_d, c_d, cache = self._lstm_step(z, h_d, c_d, self.params["W_dec"], self.params["U_dec"], self.params["b_dec"], H)
            dec_cache.append(cache)
            out[:, t] = h_d @ self.params["W_out"] + self.params["b_out"]

        loss = float(np.mean((out - X) ** 2))
        dOut = (2.0 / (B * T * D)) * (out - X)

        # Backward Decoder
        grads = {k: np.zeros_like(v) for k, v in self.params.items()}
        grads["b_out"] = np.sum(dOut, axis=(0, 1))
        dh_dec = np.zeros((B, H), dtype=np.float32)
        dc_dec = np.zeros((B, H), dtype=np.float32)
        dz = np.zeros_like(z)
        for t in reversed(range(T)):
            dh = dOut[:, t] @ self.params["W_out"].T + dh_dec
            grads["W_out"] += dec_cache[t][1].T @ dOut[:, t]
            dx, dh_dec, dc_dec, dW, dU, db = self._lstm_bstep(
                dh, dc_dec, dec_cache[t], self.params["W_dec"], self.params["U_dec"], H)
            dz += dx
            grads["W_dec"] += dW
            grads["U_dec"] += dU
            grads["b_dec"] += db

        # Backward Latent
        grads["W_z"] = h.T @ dz
        grads["b_z"] = np.sum(dz, axis=0)
        dh_enc = dz @ self.params["W_z"].T

        # Backward Encoder
        dc_enc = np.zeros((B, H), dtype=np.float32)
        dh_enc_prev = np.zeros((B, H), dtype=np.float32)
        for t in reversed(range(T)):
            dh = dh_enc if t == T - 1 else dh_enc_prev
            _, dh_enc_prev, dc_enc, dW, dU, db = self._lstm_bstep(
                dh, dc_enc, enc_cache[t], self.params["W_enc"], self.params["U_enc"], H)
            grads["W_enc"] += dW
            grads["U_enc"] += dU
            grads["b_enc"] += db

        # Gradient clipping
        gnorm = float(np.sqrt(sum(np.sum(g ** 2) for g in grads.values())))
        if gnorm > clip_norm:
            scale = clip_norm / max(gnorm, 1e-12)
            for k in grads:
                grads[k] *= scale

        # Adam optimization
        self.t_step += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for k in self.params:
            self.m[k] = beta1 * self.m[k] + (1 - beta1) * grads[k]
            self.v[k] = beta2 * self.v[k] + (1 - beta2) * (grads[k] ** 2)
            m_hat = self.m[k] / (1.0 - beta1 ** self.t_step)
            v_hat = self.v[k] / (1.0 - beta2 ** self.t_step)
            self.params[k] -= (lr * m_hat / (np.sqrt(v_hat) + eps)).astype(np.float32)

        return loss

    def state_dict(self) -> Dict[str, Any]:
        return {k: v.copy() for k, v in self.params.items()}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        for k in self.params:
            if k in state:
                self.params[k] = np.asarray(state[k], dtype=np.float32)

    def to(self, device: Any) -> "NumpyLSTMAutoencoder":
        return self

    def eval(self) -> "NumpyLSTMAutoencoder":
        return self


# If PyTorch is available, define the nn.Module version; otherwise use the NumPy engine
if TORCH_AVAILABLE:
    class PyTorchLSTMAutoencoder(nn.Module):
        def __init__(self, n_features: int, hidden_size: int = 64, latent_size: int = 32, num_layers: int = 1):
            super().__init__()
            self.n_features = n_features
            self.encoder = nn.LSTM(n_features, hidden_size, num_layers=num_layers, batch_first=True)
            self.to_latent = nn.Linear(hidden_size, latent_size)
            self.decoder = nn.LSTM(latent_size, hidden_size, num_layers=num_layers, batch_first=True)
            self.output_layer = nn.Linear(hidden_size, n_features)

        def encode(self, x: torch.Tensor) -> torch.Tensor:
            _, (h_n, _) = self.encoder(x)
            return self.to_latent(h_n[-1])

        def decode(self, z: torch.Tensor, seq_len: int) -> torch.Tensor:
            repeated = z.unsqueeze(1).repeat(1, seq_len, 1)
            out, _ = self.decoder(repeated)
            return self.output_layer(out)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.decode(self.encode(x), x.size(1))

    LSTMAutoencoder = PyTorchLSTMAutoencoder
else:
    LSTMAutoencoder = NumpyLSTMAutoencoder


# ---------------------------------------------------------------------------
# 6. Training & Execution
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (if available) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE and torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def select_device() -> Any:
    if TORCH_AVAILABLE and torch is not None and torch.cuda.is_available():
        return torch.device("cuda")
    return "cpu"


def train_autoencoder(
    model: Any,
    train_windows: np.ndarray,
    val_windows: np.ndarray,
    cfg: MonitoringConfig,
    device: Any = "cpu",
) -> Dict:
    """Train with MSE + Adam, gradient clipping, and early stopping on HEALTHY windows."""
    if isinstance(model, NumpyLSTMAutoencoder):
        return _train_numpy_autoencoder(model, train_windows, val_windows, cfg)
    elif TORCH_AVAILABLE and isinstance(model, nn.Module):
        return _train_torch_autoencoder(model, train_windows, val_windows, cfg, device)
    else:
        # Fallback to numpy training if unknown
        return _train_numpy_autoencoder(model, train_windows, val_windows, cfg)


def _train_numpy_autoencoder(
    model: NumpyLSTMAutoencoder,
    train_windows: np.ndarray,
    val_windows: np.ndarray,
    cfg: MonitoringConfig,
) -> Dict:
    rng = np.random.default_rng(cfg.seed)
    n_samples = len(train_windows)
    best_loss = math.inf
    best_state = None
    best_epoch = 0
    bad_epochs = 0
    history = []

    for epoch in range(1, cfg.max_epochs + 1):
        indices = rng.permutation(n_samples)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_samples, cfg.batch_size):
            batch = train_windows[indices[start : start + cfg.batch_size]]
            loss = model.train_step(batch, lr=cfg.learning_rate, clip_norm=5.0)
            epoch_loss += loss * len(batch)
            n_batches += len(batch)
        train_loss = float(epoch_loss / max(n_batches, 1))

        # Evaluate on validation
        val_pred = model.predict(val_windows, batch_size=cfg.batch_size)
        val_loss = float(np.mean((val_pred - val_windows) ** 2))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        LOGGER.info("epoch %02d  train_mse=%.5f  val_mse=%.5f", epoch, train_loss, val_loss)

        if val_loss < best_loss - 1e-7:
            best_loss, best_epoch, bad_epochs = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                LOGGER.info("early stopping at epoch %d (best epoch %d)", epoch, best_epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "epochs_run": len(history),
        "early_stopped": len(history) < cfg.max_epochs,
    }


def _train_torch_autoencoder(
    model: nn.Module,
    train_windows: np.ndarray,
    val_windows: np.ndarray,
    cfg: MonitoringConfig,
    device: Any,
) -> Dict:
    model.to(device)
    generator = torch.Generator().manual_seed(cfg.seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_windows)),
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=generator,
    )
    val_tensor = torch.from_numpy(val_windows)
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    loss_fn = nn.MSELoss()

    best_loss, best_state, best_epoch, bad_epochs = math.inf, None, 0, 0
    history = []
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        total, count = 0.0, 0
        for (batch,) in loader:
            batch = batch.to(device)
            optimiser.zero_grad()
            pred = model(batch)
            loss = loss_fn(pred, batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimiser.step()
            total += loss.item() * batch.size(0)
            count += batch.size(0)
        train_loss = total / max(count, 1)

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(val_tensor.to(device)), val_tensor.to(device)).item())
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        LOGGER.info("epoch %02d  train_mse=%.5f  val_mse=%.5f", epoch, train_loss, val_loss)

        if val_loss < best_loss - 1e-7:
            best_loss, best_epoch, bad_epochs = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                LOGGER.info("early stopping at epoch %d (best epoch %d)", epoch, best_epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "epochs_run": len(history),
        "early_stopped": len(history) < cfg.max_epochs,
    }


# ---------------------------------------------------------------------------
# 7. Scoring, threshold, persistent alerts
# ---------------------------------------------------------------------------
def reconstruction_errors(model: Any, windows: np.ndarray, device: Any = "cpu",
                          batch_size: int = 512) -> np.ndarray:
    """Per-window, per-feature mean squared reconstruction error -> (N, F)."""
    if len(windows) == 0:
        return np.empty((0, windows.shape[-1] if windows.ndim == 3 else 0), dtype=np.float32)

    if isinstance(model, NumpyLSTMAutoencoder):
        reconstructed = model.predict(windows, batch_size=batch_size)
        sq_err = (windows - reconstructed) ** 2  # (N, T, F)
        return sq_err.mean(axis=1)             # (N, F)
    elif TORCH_AVAILABLE and isinstance(model, nn.Module):
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(windows), batch_size):
                b = torch.from_numpy(windows[i : i + batch_size]).to(device)
                err = (model(b) - b) ** 2
                out.append(err.mean(dim=1).cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.empty((0, windows.shape[-1]), dtype=np.float32)
    else:
        reconstructed = model.predict(windows)
        sq_err = (windows - reconstructed) ** 2
        return sq_err.mean(axis=1)


def anomaly_scores(per_feature_errors: np.ndarray) -> np.ndarray:
    return per_feature_errors.mean(axis=1)


def compute_threshold(healthy_train_scores: np.ndarray, percentile: float) -> float:
    """THRESHOLD = percentile of healthy reconstruction errors."""
    return float(np.percentile(healthy_train_scores, percentile))


def apply_persistence(raw_alerts: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """ALERT PERSISTENCE: confirm an alert only after k consecutive raw alerts."""
    consecutive = np.zeros(len(raw_alerts), dtype=int)
    run = 0
    for i, flag in enumerate(raw_alerts):
        run = run + 1 if flag else 0
        consecutive[i] = run
    return consecutive, consecutive >= k


# ---------------------------------------------------------------------------
# 8. The Monitoring Agent
# ---------------------------------------------------------------------------
class LSTMMonitoringAgent:
    """Wraps model + preprocessing + threshold + alert logic into one monitor."""

    def __init__(self, model: Any, scaler: FittedScaler, features: List[str],
                 threshold: float, cfg: MonitoringConfig, device: Optional[Any] = None):
        self.device = device or select_device()
        self.model = model
        if hasattr(self.model, "to") and TORCH_AVAILABLE and isinstance(self.model, nn.Module):
            self.model = self.model.to(self.device).eval()
        self.scaler = scaler
        self.features = list(features)
        self.threshold = float(threshold)
        self.cfg = cfg

    def monitor(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score every window of every engine in `df` and apply persistent alerts."""
        scaled = scale_frame(df, self.features, self.scaler)
        windows, meta = make_windows(scaled, self.features, self.cfg.window_size, self.cfg.stride)
        per_feature = reconstruction_errors(self.model, windows, self.device)
        scores = anomaly_scores(per_feature)

        result = meta.copy()
        result["anomaly_score"] = scores
        result["threshold"] = self.threshold
        result["raw_alert"] = scores > self.threshold
        result["consecutive_alerts"] = 0
        result["confirmed_alert"] = False
        for unit, idx in result.groupby(UNIT_COL).groups.items():
            idx = np.asarray(idx)
            cons, conf = apply_persistence(result.loc[idx, "raw_alert"].to_numpy(), self.cfg.persistence)
            result.loc[idx, "consecutive_alerts"] = cons
            result.loc[idx, "confirmed_alert"] = conf

        # Feature attribution
        shares = per_feature / np.clip(per_feature.sum(axis=1, keepdims=True), 1e-12, None)
        top_idx = np.argsort(-per_feature, axis=1)[:, : self.cfg.top_k_features]
        result["top_features"] = [
            ";".join(f"{self.features[j]}:{shares[r, j]:.3f}" for j in row) for r, row in enumerate(top_idx)
        ]
        err_cols = pd.DataFrame(per_feature, columns=[f"err_{f}" for f in self.features])
        return pd.concat([result.reset_index(drop=True), err_cols], axis=1)

    def explain(self, row: pd.Series) -> str:
        status = "CONFIRMED ALERT" if row["confirmed_alert"] else ("raw alert (unconfirmed)" if row["raw_alert"] else "normal")
        contributors = ", ".join(
            f"{p.split(':')[0]} ({float(p.split(':')[1]) * 100:.0f}% of error)" for p in row["top_features"].split(";"))
        return (f"Unit {int(row[UNIT_COL])}, cycles {int(row['window_start_cycle'])}-{int(row['window_end_cycle'])}: "
                f"{status}; score {row['anomaly_score']:.4f} vs threshold {self.threshold:.4f} "
                f"({int(row['consecutive_alerts'])} consecutive). Largest deviations: {contributors}. "
                f"This indicates departure from learned healthy behaviour, not a diagnosed fault.")

    def save(self, path: Path | str, extra: Optional[Dict] = None) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        is_torch = TORCH_AVAILABLE and isinstance(self.model, nn.Module)
        payload = {
            "model_state_dict": self.model.state_dict(),
            "n_features": len(self.features),
            "features": self.features,
            "scaler": self.scaler.state_dict(),
            "threshold": self.threshold,
            "config": self.cfg.to_dict(),
            "backend": "torch" if is_torch else "numpy",
            "extra": extra or {},
        }
        if is_torch:
            torch.save(payload, p)
        else:
            joblib.dump(payload, p)

    @classmethod
    def load(cls, path: Path | str, device: Optional[Any] = None) -> "LSTMMonitoringAgent":
        device = device or select_device()
        p = Path(path)
        ckpt = None
        if TORCH_AVAILABLE and torch is not None:
            try:
                ckpt = torch.load(p, map_location=device)
            except Exception:
                ckpt = None
        if ckpt is None:
            ckpt = joblib.load(p)

        cfg = MonitoringConfig(**ckpt["config"])
        backend = ckpt.get("backend", "numpy")
        if backend == "torch" and TORCH_AVAILABLE and nn is not None:
            model = LSTMAutoencoder(ckpt["n_features"], cfg.hidden_size, cfg.latent_size, cfg.num_layers)
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model = NumpyLSTMAutoencoder(ckpt["n_features"], cfg.hidden_size, cfg.latent_size, cfg.seed)
            model.load_state_dict(ckpt["model_state_dict"])

        return cls(model, FittedScaler.from_state_dict(ckpt["scaler"]), ckpt["features"],
                   ckpt["threshold"], cfg, device)


# ---------------------------------------------------------------------------
# 9. Evaluation
# ---------------------------------------------------------------------------
def _valid(values: Sequence[Optional[float]]) -> List[float]:
    return [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]


def _safe_mean(values: Sequence[Optional[float]]) -> Optional[float]:
    vals = _valid(values)
    return float(np.mean(vals)) if vals else None


def _safe_median(values: Sequence[Optional[float]]) -> Optional[float]:
    vals = _valid(values)
    return float(np.median(vals)) if vals else None


def evaluate_fleet(events: pd.DataFrame, failure_cycles: Dict[int, int], cfg: MonitoringConfig,
                   features: List[str], split: str) -> Tuple[pd.DataFrame, List[Dict], Dict]:
    """Per-engine and fleet metrics including precision, recall, and F1."""
    ev = events.copy()
    ev.insert(0, "split", split)
    ev["failure_cycle"] = ev[UNIT_COL].map(failure_cycles).astype(int)
    ev["rul_at_window_end"] = ev["failure_cycle"] - ev["window_end_cycle"]
    ev["is_healthy_window"] = ev["window_end_cycle"] <= cfg.healthy_cycles
    ev["is_degraded_window"] = ev["rul_at_window_end"] <= cfg.degraded_rul
    err_cols = [f"err_{f}" for f in features]

    # Window-level precision, recall, F1
    tp = int((ev["is_degraded_window"] & ev["confirmed_alert"]).sum())
    fp = int((ev["is_healthy_window"] & ev["confirmed_alert"]).sum())
    fn = int((ev["is_degraded_window"] & ~ev["confirmed_alert"]).sum())
    tn = int((ev["is_healthy_window"] & ~ev["confirmed_alert"]).sum())
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    per_engine = []
    for unit, g in ev.groupby(UNIT_COL, sort=True):
        healthy, degraded = g[g["is_healthy_window"]], g[g["is_degraded_window"]]
        post = g[(~g["is_healthy_window"]) & g["confirmed_alert"]]
        first = post.iloc[0] if len(post) else None
        raw = g["raw_alert"].to_numpy()
        runs = np.diff(np.flatnonzero(np.diff(np.r_[0, raw.astype(int), 0])))[::2] if raw.any() else np.array([])
        e_tp = int((degraded["confirmed_alert"]).sum())
        e_fp = int((healthy["confirmed_alert"]).sum())
        e_fn = int((~degraded["confirmed_alert"]).sum())
        e_prec = float(e_tp / (e_tp + e_fp)) if (e_tp + e_fp) > 0 else 0.0
        e_rec = float(e_tp / (e_tp + e_fn)) if (e_tp + e_fn) > 0 else 0.0
        e_f1 = float(2 * e_prec * e_rec / (e_prec + e_rec)) if (e_prec + e_rec) > 0 else 0.0
        r_rec = {
            "unit_id": int(unit),
            "last_observed_cycle": int(g["window_end_cycle"].max()),
            "failure_cycle": int(g["failure_cycle"].iloc[0]),
            "n_windows": int(len(g)),
            "n_healthy_windows": int(len(healthy)),
            "n_degraded_windows": int(len(degraded)),
            "n_raw_alert_windows": int(raw.sum()),
            "n_confirmed_alert_windows": int(g["confirmed_alert"].sum()),
            "isolated_spikes_suppressed": int((runs < cfg.persistence).sum()),
            "detection_rate": float(degraded["confirmed_alert"].mean()) if len(degraded) else None,
            "healthy_false_alarm_rate": float(healthy["confirmed_alert"].mean()) if len(healthy) else None,
            "precision": e_prec,
            "recall": e_rec,
            "f1": e_f1,
            "first_alert_cycle": int(first["window_end_cycle"]) if first is not None else None,
            "lead_time": int(first["failure_cycle"] - first["window_end_cycle"]) if first is not None else None,
            "top_features_at_first_alert": first["top_features"] if first is not None else None,
        }
        per_engine.append(r_rec)

    pe = pd.DataFrame(per_engine)
    with_degraded = pe[pe["n_degraded_windows"] > 0]
    leads = pe["lead_time"].dropna().tolist()
    confirmed = ev[ev["confirmed_alert"] & ~ev["is_healthy_window"]]
    contrib = {}
    if len(confirmed):
        shares = confirmed[err_cols].to_numpy()
        shares = shares / np.clip(shares.sum(axis=1, keepdims=True), 1e-12, None)
        contrib = dict(sorted(((f, float(s)) for f, s in zip(features, shares.mean(axis=0))),
                              key=lambda kv: -kv[1]))

    fleet = {
        "n_engines": int(len(pe)),
        "n_windows": int(len(ev)),
        "n_engines_with_degraded_windows": int(len(with_degraded)),
        "engine_detection_rate": float(with_degraded["first_alert_cycle"].notna().mean()) if len(with_degraded) else None,
        "mean_detection_rate": _safe_mean(pe["detection_rate"].tolist()),
        "pooled_window_detection_rate": float(ev.loc[ev["is_degraded_window"], "confirmed_alert"].mean())
        if ev["is_degraded_window"].any() else None,
        "mean_healthy_false_alarm_rate": _safe_mean(pe["healthy_false_alarm_rate"].tolist()),
        "pooled_healthy_false_alarm_rate": float(ev.loc[ev["is_healthy_window"], "confirmed_alert"].mean())
        if ev["is_healthy_window"].any() else None,
        "pooled_healthy_raw_alert_rate": float(ev.loc[ev["is_healthy_window"], "raw_alert"].mean())
        if ev["is_healthy_window"].any() else None,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "n_engines_with_healthy_false_alarm": int((pe["healthy_false_alarm_rate"].fillna(0) > 0).sum()),
        "n_engines_alerted": int(pe["first_alert_cycle"].notna().sum()),
        "median_lead_time": _safe_median(leads),
        "mean_lead_time": _safe_mean(leads),
        "min_lead_time": float(min(leads)) if leads else None,
        "max_lead_time": float(max(leads)) if leads else None,
        "isolated_spikes_suppressed": int(pe["isolated_spikes_suppressed"].sum()),
        "mean_feature_error_share_in_confirmed_alerts": contrib,
        "healthy_cycle_assumption": f"cycle <= {cfg.healthy_cycles} assumed healthy",
        "degraded_rul_assumption": f"rul <= {cfg.degraded_rul} assumed degraded",
    }
    per_engine = [{k: (None if (isinstance(v, float) and math.isnan(v)) else v) for k, v in r.items()}
                  for r in per_engine]
    return ev, per_engine, fleet
