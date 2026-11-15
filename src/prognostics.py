"""Prognostics Agent for RUL prediction, sequence modeling, and calibrated uncertainty."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import torch as _torch
    from torch import nn as _torch_nn
except Exception:  # optional models must not make the base package unimportable
    _torch = None
    _torch_nn = None

from src.cmapss_data import (
    fit_feature_scaler,
    make_windows,
    split_engine_units,
    summarize_windows,
    transform_features,
)


if _torch_nn is not None:

    class _CNNSequenceRegressor(_torch_nn.Module):
        """Compact temporal CNN with masked-free global mean/max pooling."""

        def __init__(self, in_features: int, hidden_size: int) -> None:
            super().__init__()
            channels = max(16, hidden_size)
            self.convolutions = _torch_nn.Sequential(
                _torch_nn.Conv1d(in_features, channels, kernel_size=5, padding=2),
                _torch_nn.GELU(),
                _torch_nn.Conv1d(channels, channels * 2, kernel_size=3, padding=1),
                _torch_nn.GELU(),
            )
            self.head = _torch_nn.Sequential(
                _torch_nn.Linear(channels * 4, channels * 2),
                _torch_nn.GELU(),
                _torch_nn.Dropout(0.1),
                _torch_nn.Linear(channels * 2, 1),
            )

        def forward(self, values):
            encoded = self.convolutions(values.transpose(1, 2))
            pooled = _torch.cat(
                [encoded.mean(dim=-1), encoded.amax(dim=-1)], dim=1
            )
            return self.head(pooled).squeeze(-1)


    class _TransformerSequenceRegressor(_torch_nn.Module):
        """Small encoder-only Transformer for fixed-length sensor windows."""

        def __init__(
            self,
            in_features: int,
            window_size: int,
            hidden_size: int,
            attention_heads: int,
            layers: int,
        ) -> None:
            super().__init__()
            if hidden_size % attention_heads:
                raise ValueError("transformer hidden size must be divisible by attention_heads")
            self.input_projection = _torch_nn.Linear(in_features, hidden_size)
            self.position = _torch_nn.Parameter(
                _torch.zeros(1, window_size, hidden_size)
            )
            _torch_nn.init.normal_(self.position, mean=0.0, std=0.02)
            encoder_layer = _torch_nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=attention_heads,
                dim_feedforward=hidden_size * 2,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = _torch_nn.TransformerEncoder(
                encoder_layer, num_layers=layers, enable_nested_tensor=False
            )
            self.head = _torch_nn.Sequential(
                _torch_nn.LayerNorm(hidden_size * 2),
                _torch_nn.Linear(hidden_size * 2, hidden_size),
                _torch_nn.GELU(),
                _torch_nn.Dropout(0.1),
                _torch_nn.Linear(hidden_size, 1),
            )

        def forward(self, values):
            encoded = self.input_projection(values)
            encoded = encoded + self.position[:, : encoded.shape[1], :]
            encoded = self.encoder(encoded)
            pooled = _torch.cat([encoded.mean(dim=1), encoded[:, -1]], dim=1)
            return self.head(pooled).squeeze(-1)

else:  # pragma: no cover - exercised on optional-dependency-free machines
    _CNNSequenceRegressor = None
    _TransformerSequenceRegressor = None


class TorchSequenceRULModel:
    """Trainable CNN or Transformer RUL regressor with inner-fold early stopping."""

    def __init__(
        self,
        architecture: str,
        in_features: int,
        window_size: int,
        *,
        hidden_size: int = 32,
        attention_heads: int = 4,
        transformer_layers: int = 2,
        random_state: int = 42,
    ) -> None:
        if _torch is None or _torch_nn is None:
            raise RuntimeError(
                "PyTorch CNN/Transformer models require PyTorch; install the CPU "
                "wheel using the official PyTorch instructions."
            )
        if architecture not in {"pytorch_cnn", "pytorch_transformer"}:
            raise ValueError("architecture must be 'pytorch_cnn' or 'pytorch_transformer'")
        if in_features < 1 or window_size < 2 or hidden_size < 1:
            raise ValueError("feature count, window size, and hidden size must be positive")
        _torch.manual_seed(int(random_state))
        if architecture == "pytorch_cnn":
            network = _CNNSequenceRegressor(in_features, hidden_size)
        else:
            network = _TransformerSequenceRegressor(
                in_features,
                window_size,
                hidden_size,
                attention_heads,
                transformer_layers,
            )
        self.architecture = architecture
        self.model = network.to("cpu")
        self.random_state = int(random_state)
        self.target_mean_ = 0.0
        self.target_scale_ = 1.0
        self.epochs_trained_: int | None = None
        self.best_epoch_: int | None = None
        self.best_validation_loss_: float | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        *,
        max_epochs: int = 12,
        batch_size: int = 128,
        lr: float = 1e-3,
        patience: int = 3,
    ) -> "TorchSequenceRULModel":
        train_x = np.asarray(X, dtype=np.float32)
        train_y = np.asarray(y, dtype=np.float32).reshape(-1)
        if train_x.ndim != 3 or len(train_x) != len(train_y) or not len(train_y):
            raise ValueError("X must be non-empty (samples, time, features) aligned with y")
        if not np.isfinite(train_x).all() or not np.isfinite(train_y).all():
            raise ValueError("training sequences and targets must be finite")
        if max_epochs < 1 or batch_size < 1 or lr <= 0 or patience < 1:
            raise ValueError("training epochs, batch size, rate, and patience must be positive")
        if (X_val is None) != (y_val is None):
            raise ValueError("X_val and y_val must either both be supplied or both omitted")

        val_x: np.ndarray | None = None
        val_y: np.ndarray | None = None
        if X_val is not None and y_val is not None:
            val_x = np.asarray(X_val, dtype=np.float32)
            val_y = np.asarray(y_val, dtype=np.float32).reshape(-1)
            if (
                val_x.ndim != 3
                or len(val_x) != len(val_y)
                or not len(val_y)
                or val_x.shape[1:] != train_x.shape[1:]
            ):
                raise ValueError("validation sequences must align with training sequence shape")
            if not np.isfinite(val_x).all() or not np.isfinite(val_y).all():
                raise ValueError("validation sequences and targets must be finite")

        self.target_mean_ = float(train_y.mean())
        self.target_scale_ = float(train_y.std()) or 1.0
        scaled_train_y = (train_y - self.target_mean_) / self.target_scale_
        scaled_val_y = (
            None
            if val_y is None
            else (val_y - self.target_mean_) / self.target_scale_
        )
        tensor_x = _torch.as_tensor(train_x, dtype=_torch.float32)
        tensor_y = _torch.as_tensor(scaled_train_y, dtype=_torch.float32)
        tensor_val_x = (
            None
            if val_x is None
            else _torch.as_tensor(val_x, dtype=_torch.float32)
        )
        tensor_val_y = (
            None
            if scaled_val_y is None
            else _torch.as_tensor(scaled_val_y, dtype=_torch.float32)
        )

        optimizer = _torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        rng = np.random.default_rng(self.random_state)
        best_loss = float("inf")
        best_state: dict[str, Any] | None = None
        best_epoch = 0
        epochs_without_improvement = 0
        for epoch in range(1, max_epochs + 1):
            self.model.train()
            permutation = rng.permutation(len(tensor_x))
            train_losses: list[float] = []
            for start in range(0, len(permutation), batch_size):
                indices = permutation[start : start + batch_size]
                batch_x = tensor_x[indices]
                batch_y = tensor_y[indices]
                optimizer.zero_grad(set_to_none=True)
                predictions = self.model(batch_x)
                loss = _torch_nn.functional.mse_loss(predictions, batch_y)
                loss.backward()
                _torch_nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                train_losses.append(float(loss.detach().item()))

            if tensor_val_x is not None and tensor_val_y is not None:
                self.model.eval()
                with _torch.no_grad():
                    validation_loss = float(
                        _torch_nn.functional.mse_loss(
                            self.model(tensor_val_x), tensor_val_y
                        ).item()
                    )
            else:
                validation_loss = float(np.mean(train_losses))
            if validation_loss < best_loss - 1e-7:
                best_loss = validation_loss
                best_epoch = epoch
                best_state = {
                    name: value.detach().clone()
                    for name, value in self.model.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    break

        if best_state is None:
            raise RuntimeError("sequence model training did not produce a finite checkpoint")
        self.model.load_state_dict(best_state)
        self.model.eval()
        self.epochs_trained_ = epoch
        self.best_epoch_ = best_epoch
        self.best_validation_loss_ = best_loss if tensor_val_x is not None else None
        return self

    def predict(self, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
        values = np.asarray(X, dtype=np.float32)
        if values.ndim != 3 or not np.isfinite(values).all():
            raise ValueError("X must be a finite (samples, time, features) array")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        predictions: list[np.ndarray] = []
        self.model.eval()
        with _torch.no_grad():
            for start in range(0, len(values), batch_size):
                batch = _torch.as_tensor(
                    values[start : start + batch_size], dtype=_torch.float32
                )
                scaled = self.model(batch).cpu().numpy()
                predictions.append(scaled * self.target_scale_ + self.target_mean_)
        if not predictions:
            return np.empty((0,), dtype=np.float32)
        return np.maximum(np.concatenate(predictions), 0.0)


def nasa_score(targets: np.ndarray, predictions: np.ndarray) -> float:
    """Asymmetric C-MAPSS scoring function (heavier penalty for over-estimation)."""
    difference = np.asarray(predictions, dtype=float) - np.asarray(targets, dtype=float)
    penalties = np.where(
        difference < 0,
        np.exp(-difference / 13.0) - 1.0,
        np.exp(difference / 10.0) - 1.0,
    )
    return float(np.sum(penalties))


def compute_rul_metrics(targets: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
    """Compute standard regression metrics and NASA asymmetric score."""
    t = np.asarray(targets, dtype=float)
    p = np.maximum(np.asarray(predictions, dtype=float), 0.0)
    mae = float(mean_absolute_error(t, p))
    rmse = float(np.sqrt(mean_squared_error(t, p)))
    r2 = float(r2_score(t, p)) if len(t) > 1 and np.var(t) > 1e-12 else 0.0
    score = nasa_score(t, p)
    return {"mae": mae, "rmse": rmse, "r2": r2, "nasa_score": score}


def select_rul_stratified_rows(
    predictions: pd.DataFrame,
    rul_reference: float = 125.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Select one pseudo-current window per engine across a fixed RUL range.

    This creates an evaluation sample from run-to-failure trajectories without
    evaluating every correlated sliding window or restricting validation to
    the trivial RUL-zero failure endpoint. The selection uses validation RUL
    labels only to define the evaluation point; models receive telemetry only.
    """
    required = {"unit_id", "rul"}
    missing = required.difference(predictions.columns)
    if missing:
        raise KeyError(f"predictions missing required columns: {sorted(missing)}")
    if not np.isfinite(predictions["rul"].to_numpy(dtype=float)).all():
        raise ValueError("RUL values used for point selection must be finite")
    units = sorted(int(unit) for unit in predictions["unit_id"].unique())
    if not units:
        return predictions.iloc[0:0].copy()
    rng = np.random.default_rng(seed)
    targets = np.linspace(0.0, float(rul_reference), num=len(units), endpoint=False)
    rng.shuffle(targets)
    selected_indices = []
    for unit_id, target in zip(units, targets):
        candidates = predictions[predictions["unit_id"] == unit_id]
        in_range = candidates[candidates["rul"] <= rul_reference]
        if not in_range.empty:
            candidates = in_range
        position = int(np.argmin(np.abs(candidates["rul"].to_numpy(dtype=float) - target)))
        selected_indices.append(candidates.index[position])
    return predictions.loc[selected_indices].sort_values("unit_id").copy()


@dataclass(frozen=True)
class PrognosticsConfig:
    window_size: int = 30
    n_estimators: int = 150
    min_samples_leaf: int = 2
    max_features: float = 0.70
    random_state: int = 42
    rul_reference: float = 125.0
    model_type: str = "random_forest"
    training_window_stride: int = 1
    include_cycle_feature: bool = False
    planning_margin_cycles: tuple[int, ...] = ()
    lstm_hidden_size: int = 24
    lstm_epochs: int = 12
    lstm_batch_size: int = 128
    lstm_learning_rate: float = 1e-3
    lstm_training_stride: int = 5
    lstm_validation_fraction: float = 0.15
    lstm_early_stopping_patience: int = 3
    xgb_n_estimators: int = 250
    xgb_max_depth: int = 5
    xgb_learning_rate: float = 0.04
    xgb_subsample: float = 0.9
    xgb_colsample_bytree: float = 0.8
    torch_hidden_size: int = 32
    torch_epochs: int = 8
    torch_batch_size: int = 128
    torch_learning_rate: float = 1e-3
    torch_training_stride: int = 5
    torch_validation_fraction: float = 0.15
    torch_early_stopping_patience: int = 2
    torch_attention_heads: int = 4
    torch_transformer_layers: int = 2

    def as_dict(self) -> dict:
        return asdict(self)


def _append_cycle_channel(
    windows: np.ndarray,
    metadata: pd.DataFrame,
) -> np.ndarray:
    """Append normalized per-timestep engine age to sequence windows."""

    if len(windows) != len(metadata):
        raise ValueError("window metadata length does not match the sequence count")
    offsets = np.arange(windows.shape[1], dtype=np.float32)[None, :]
    first_cycle = (
        metadata["end_cycle"].to_numpy(dtype=np.float32)[:, None]
        - windows.shape[1]
        + 1
    )
    cycle_channel = (first_cycle + offsets)[:, :, None] / 100.0
    return np.concatenate([windows, cycle_channel], axis=2).astype(np.float32)


class NaiveRULModel:
    """Naive baseline predicting constant mean RUL or linear lifetime decay."""

    def __init__(self, default_rul: float = 125.0):
        self.default_rul = default_rul

    def fit(self, targets: np.ndarray) -> "NaiveRULModel":
        self.default_rul = float(np.mean(targets))
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.full(len(features), self.default_rul, dtype=np.float32)


class ClassicalRidgeRULModel:
    """Classical linear baseline using Ridge regression on summary features."""

    def __init__(self, alpha: float = 1.0, random_state: int = 42):
        self.model = Ridge(alpha=alpha, random_state=random_state)

    def fit(self, features: np.ndarray, targets: np.ndarray) -> "ClassicalRidgeRULModel":
        self.model.fit(features, targets)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.maximum(self.model.predict(features), 0.0).astype(np.float32)


class NumpyLSTMRULModel:
    """Deterministic sequence-to-regression LSTM for RUL prediction."""

    def __init__(self, in_features: int, hidden_size: int = 48, seed: int = 42):
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.seed = seed
        self.target_mean_ = 0.0
        self.target_scale_ = 1.0
        rng = np.random.default_rng(seed)
        std_lstm = np.float32(np.sqrt(2.0 / (in_features + hidden_size)))
        std_head = np.float32(np.sqrt(2.0 / hidden_size))

        self.params: Dict[str, np.ndarray] = {
            "W": rng.normal(0, std_lstm, (in_features, 4 * hidden_size)).astype(np.float32),
            "U": rng.normal(0, std_lstm, (hidden_size, 4 * hidden_size)).astype(np.float32),
            "b": np.zeros(4 * hidden_size, dtype=np.float32),
            "W_head": rng.normal(0, std_head, (hidden_size, 1)).astype(np.float32),
            "b_head": np.zeros(1, dtype=np.float32),
        }
        self.params["b"][hidden_size : 2 * hidden_size] = 1.0
        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.t_step = 0

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        max_epochs: int = 15,
        batch_size: int = 128,
        lr: float = 1e-3,
        patience: int = 3,
    ) -> "NumpyLSTMRULModel":
        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32).reshape(-1)
        if X_train.ndim != 3 or X_train.shape[2] != self.in_features:
            raise ValueError("X_train must have shape (samples, time, in_features)")
        if len(X_train) != len(y_train) or len(y_train) == 0:
            raise ValueError("X_train and y_train must have the same non-zero sample count")
        if not np.isfinite(X_train).all() or not np.isfinite(y_train).all():
            raise ValueError("training sequences and RUL targets must be finite")
        if (X_val is None) != (y_val is None):
            raise ValueError("X_val and y_val must either both be supplied or both omitted")
        if X_val is not None:
            X_val = np.asarray(X_val, dtype=np.float32)
            y_val = np.asarray(y_val, dtype=np.float32).reshape(-1)
            if X_val.ndim != 3 or X_val.shape[2] != self.in_features:
                raise ValueError("X_val must have shape (samples, time, in_features)")
            if len(X_val) != len(y_val) or len(y_val) == 0:
                raise ValueError("X_val and y_val must have the same non-zero sample count")
            if not np.isfinite(X_val).all() or not np.isfinite(y_val).all():
                raise ValueError("validation sequences and RUL targets must be finite")
        if max_epochs < 1 or batch_size < 1 or lr <= 0:
            raise ValueError("max_epochs, batch_size, and lr must be positive")
        if patience < 1:
            raise ValueError("patience must be at least one")
        self.target_mean_ = float(y_train.mean())
        target_std = float(y_train.std())
        self.target_scale_ = target_std if target_std > 1e-8 else 1.0
        y_train_scaled = (y_train - self.target_mean_) / self.target_scale_
        rng = np.random.default_rng(self.seed)
        n = len(X_train)
        H = self.hidden_size
        best_validation_loss = float("inf")
        best_params: Dict[str, np.ndarray] | None = None
        best_epoch = 0
        epochs_without_improvement = 0
        epochs_trained = 0
        for epoch in range(1, max_epochs + 1):
            idx = rng.permutation(n)
            for s in range(0, n, batch_size):
                b_idx = idx[s : s + batch_size]
                X_b, y_b = X_train[b_idx], y_train_scaled[b_idx]
                B, T, D = X_b.shape

                # Forward
                h = np.zeros((B, H), dtype=np.float32)
                c = np.zeros((B, H), dtype=np.float32)
                caches = []
                for t in range(T):
                    h_prev = h.copy()
                    c_prev = c.copy()
                    gates = X_b[:, t] @ self.params["W"] + h @ self.params["U"] + self.params["b"]
                    i_g = 1.0 / (1.0 + np.exp(-np.clip(gates[:, :H], -15.0, 15.0)))
                    f_g = 1.0 / (1.0 + np.exp(-np.clip(gates[:, H : 2 * H], -15.0, 15.0)))
                    o_g = 1.0 / (1.0 + np.exp(-np.clip(gates[:, 2 * H : 3 * H], -15.0, 15.0)))
                    c_t = np.tanh(gates[:, 3 * H :])
                    c = f_g * c + i_g * c_t
                    tc = np.tanh(c)
                    h = o_g * tc
                    caches.append((X_b[:, t], h_prev, c_prev, i_g, f_g, o_g, c_t, tc, c))

                pred = (h @ self.params["W_head"] + self.params["b_head"]).squeeze(-1)
                err = pred - y_b
                dPred = (2.0 / B) * err

                # Backward head
                grads = {k: np.zeros_like(v) for k, v in self.params.items()}
                grads["W_head"] = h.T @ dPred[:, None]
                grads["b_head"] = np.sum(dPred, keepdims=True)
                dh = dPred[:, None] @ self.params["W_head"].T

                # Backward LSTM
                dc = np.zeros((B, H), dtype=np.float32)
                dh_prev = np.zeros((B, H), dtype=np.float32)
                for t in reversed(range(T)):
                    dh_cur = dh if t == T - 1 else dh_prev
                    x_t, h_p, c_p, i_g, f_g, o_g, c_t, tc, c_cur = caches[t]
                    dc = dh_cur * o_g * (1.0 - tc ** 2) + dc
                    do = dh_cur * tc * o_g * (1.0 - o_g)
                    dct = dc * i_g * (1.0 - c_t ** 2)
                    di = dc * c_t * i_g * (1.0 - i_g)
                    df = dc * c_p * f_g * (1.0 - f_g)
                    dc = dc * f_g
                    dgates = np.concatenate([di, df, do, dct], axis=1)
                    grads["W"] += x_t.T @ dgates
                    grads["U"] += h_p.T @ dgates
                    grads["b"] += np.sum(dgates, axis=0)
                    dh_prev = dgates @ self.params["U"].T

                # Clip gradients
                norm = float(np.sqrt(sum(np.sum(g ** 2) for g in grads.values())))
                if norm > 5.0:
                    for k in grads:
                        grads[k] *= 5.0 / norm

                # Adam
                self.t_step += 1
                for k in self.params:
                    self.m[k] = 0.9 * self.m[k] + 0.1 * grads[k]
                    self.v[k] = 0.999 * self.v[k] + 0.001 * (grads[k] ** 2)
                    m_hat = self.m[k] / (1.0 - 0.9 ** self.t_step)
                    v_hat = self.v[k] / (1.0 - 0.999 ** self.t_step)
                    self.params[k] -= (lr * m_hat / (np.sqrt(v_hat) + 1e-8)).astype(np.float32)

            epochs_trained = epoch
            if X_val is not None and y_val is not None:
                validation_predictions = self.predict(X_val)
                validation_loss = float(np.mean((validation_predictions - y_val) ** 2))
                if validation_loss < best_validation_loss - 1e-6:
                    best_validation_loss = validation_loss
                    best_params = {
                        name: value.copy() for name, value in self.params.items()
                    }
                    best_epoch = epoch
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        break

        if best_params is not None:
            self.params = best_params
        self.epochs_trained_ = epochs_trained
        self.best_epoch_ = best_epoch if best_params is not None else epochs_trained
        self.best_validation_loss_ = (
            best_validation_loss if best_params is not None else None
        )

        return self

    def predict(self, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
        preds = []
        H = self.hidden_size
        for s in range(0, len(X), batch_size):
            batch = X[s : s + batch_size]
            B, T, D = batch.shape
            h = np.zeros((B, H), dtype=np.float32)
            c = np.zeros((B, H), dtype=np.float32)
            for t in range(T):
                gates = batch[:, t] @ self.params["W"] + h @ self.params["U"] + self.params["b"]
                i_g = 1.0 / (1.0 + np.exp(-np.clip(gates[:, :H], -15.0, 15.0)))
                f_g = 1.0 / (1.0 + np.exp(-np.clip(gates[:, H : 2 * H], -15.0, 15.0)))
                o_g = 1.0 / (1.0 + np.exp(-np.clip(gates[:, 2 * H : 3 * H], -15.0, 15.0)))
                c_t = np.tanh(gates[:, 3 * H :])
                c = f_g * c + i_g * c_t
                h = o_g * np.tanh(c)
            pred_scaled = (h @ self.params["W_head"] + self.params["b_head"]).squeeze(-1)
            pred = pred_scaled * self.target_scale_ + self.target_mean_
            preds.append(np.maximum(pred, 0.0))
        return np.concatenate(preds, axis=0) if preds else np.empty((0,), dtype=np.float32)


class PrognosticsAgent:
    """Predict Remaining Useful Life from a telemetry window.

    The primary baseline is Random Forest (MAE ~20.52 on FD001).
    Includes conformal quantile uncertainty calibration for verified coverage.
    """

    def __init__(
        self,
        model: Any,
        scaler: Any,
        feature_columns: list[str],
        config: PrognosticsConfig,
        conformal_q90: float = 25.0,
        conformal_q95: float = 35.0,
        lower_bound_margin_95: float = 35.0,
    ) -> None:
        self.model = model
        self.scaler = scaler
        self.feature_columns = list(feature_columns)
        self.config = config
        self.conformal_q90 = float(conformal_q90)
        self.conformal_q95 = float(conformal_q95)
        self.lower_bound_margin_95 = float(lower_bound_margin_95)

    @classmethod
    def fit(
        cls,
        training_frame: pd.DataFrame,
        feature_columns: list[str],
        config: PrognosticsConfig | None = None,
        validation_frame: pd.DataFrame | None = None,
    ) -> "PrognosticsAgent":
        config = config or PrognosticsConfig()
        if "rul" not in training_frame.columns:
            raise ValueError("training_frame must contain a row-level 'rul' column")
        supported_models = {
            "random_forest",
            "xgboost",
            "ridge",
            "naive",
            "lstm",
            "pytorch_cnn",
            "pytorch_transformer",
        }
        if config.model_type not in supported_models:
            raise ValueError(
                f"model_type must be one of {sorted(supported_models)}, "
                f"got {config.model_type!r}"
            )
        if config.window_size < 2:
            raise ValueError("window_size must be at least two")

        sequence_models = {"lstm", "pytorch_cnn", "pytorch_transformer"}
        if config.model_type in sequence_models:
            is_lstm = config.model_type == "lstm"
            validation_fraction = (
                config.lstm_validation_fraction
                if is_lstm
                else config.torch_validation_fraction
            )
            training_stride = (
                config.lstm_training_stride
                if is_lstm
                else config.torch_training_stride
            )
            if is_lstm:
                if config.lstm_hidden_size < 1:
                    raise ValueError("LSTM hidden size must be positive")
            elif (
                config.torch_hidden_size < 1
                or config.torch_epochs < 1
                or config.torch_batch_size < 1
                or config.torch_learning_rate <= 0
            ):
                raise ValueError("PyTorch sequence model parameters must be positive")
            if training_stride < 1:
                raise ValueError("sequence training stride must be positive")
            if not 0.0 <= validation_fraction < 0.5:
                raise ValueError("sequence validation fraction must be in [0, 0.5)")

            sequence_train_frame = training_frame
            sequence_validation_frame: pd.DataFrame | None = None
            if validation_frame is not None:
                if "rul" not in validation_frame.columns:
                    raise ValueError(
                        "validation_frame must contain row-level 'rul' labels"
                    )
                fit_units = set(training_frame["unit_id"].unique())
                validation_units = set(validation_frame["unit_id"].unique())
                if not fit_units.isdisjoint(validation_units):
                    raise ValueError(
                        "sequence early-stop engines must be disjoint from fit engines"
                    )
                sequence_validation_frame = validation_frame
            elif training_frame["unit_id"].nunique() >= 8 and validation_fraction > 0.0:
                inner_split = split_engine_units(
                    training_frame,
                    validation_fraction=validation_fraction,
                    seed=config.random_state,
                )
                sequence_train_frame = training_frame[
                    training_frame["unit_id"].isin(inner_split.train_units)
                ].copy()
                sequence_validation_frame = training_frame[
                    training_frame["unit_id"].isin(inner_split.validation_units)
                ].copy()
                if not set(inner_split.train_units).isdisjoint(
                    inner_split.validation_units
                ):
                    raise RuntimeError("engine-level sequence early-stop split overlaps")

            scaler = fit_feature_scaler(sequence_train_frame, feature_columns)
            scaled_train = transform_features(
                sequence_train_frame, feature_columns, scaler
            )
            windows, targets, metadata = make_windows(
                scaled_train,
                feature_columns,
                window_size=config.window_size,
                target_column="rul",
                stride=training_stride,
            )
            if len(windows) == 0:
                raise ValueError(
                    f"No training windows were generated for {config.model_type}"
                )
            if config.include_cycle_feature:
                windows = _append_cycle_channel(windows, metadata)

            validation_windows = None
            validation_targets = None
            if sequence_validation_frame is not None:
                scaled_validation = transform_features(
                    sequence_validation_frame, feature_columns, scaler
                )
                validation_windows, validation_targets, validation_metadata = (
                    make_windows(
                        scaled_validation,
                        feature_columns,
                        window_size=config.window_size,
                        target_column="rul",
                        stride=training_stride,
                    )
                )
                if len(validation_windows) == 0:
                    raise ValueError(
                        f"No validation windows were generated for {config.model_type}"
                    )
                if config.include_cycle_feature:
                    validation_windows = _append_cycle_channel(
                        validation_windows, validation_metadata
                    )

            if is_lstm:
                model: Any = NumpyLSTMRULModel(
                    in_features=windows.shape[2],
                    hidden_size=config.lstm_hidden_size,
                    seed=config.random_state,
                )
                model.fit(
                    windows,
                    targets,
                    X_val=validation_windows,
                    y_val=validation_targets,
                    max_epochs=config.lstm_epochs,
                    batch_size=config.lstm_batch_size,
                    lr=config.lstm_learning_rate,
                    patience=config.lstm_early_stopping_patience,
                )
            else:
                model = TorchSequenceRULModel(
                    architecture=config.model_type,
                    in_features=windows.shape[2],
                    window_size=config.window_size,
                    hidden_size=config.torch_hidden_size,
                    attention_heads=config.torch_attention_heads,
                    transformer_layers=config.torch_transformer_layers,
                    random_state=config.random_state,
                )
                model.fit(
                    windows,
                    targets,
                    X_val=validation_windows,
                    y_val=validation_targets,
                    max_epochs=config.torch_epochs,
                    batch_size=config.torch_batch_size,
                    lr=config.torch_learning_rate,
                    patience=config.torch_early_stopping_patience,
                )
            return cls(model, scaler, list(feature_columns), config)

        scaler = fit_feature_scaler(training_frame, feature_columns)
        scaled = transform_features(training_frame, feature_columns, scaler)
        if config.training_window_stride < 1:
            raise ValueError("training_window_stride must be positive")
        windows, targets, metadata = make_windows(
            scaled,
            feature_columns,
            window_size=config.window_size,
            target_column="rul",
            stride=config.training_window_stride,
        )
        if len(windows) == 0:
            raise ValueError("No training windows were generated")
        summary_features = summarize_windows(windows)
        if config.include_cycle_feature:
            cycle_feature = (
                metadata["end_cycle"].to_numpy(dtype=np.float32)[:, None] / 100.0
            )
            summary_features = np.concatenate(
                [summary_features, cycle_feature], axis=1
            )

        if config.model_type == "random_forest":
            model: Any = RandomForestRegressor(
                n_estimators=config.n_estimators,
                min_samples_leaf=config.min_samples_leaf,
                max_features=config.max_features,
                random_state=config.random_state,
                n_jobs=-1,
            )
            model.fit(summary_features, targets)
        elif config.model_type == "xgboost":
            try:
                from xgboost import XGBRegressor
            except ImportError as exc:
                raise RuntimeError(
                    "XGBoost is optional; install it with "
                    "python -m pip install -r requirements-model-comparison.txt"
                ) from exc
            model = XGBRegressor(
                n_estimators=config.xgb_n_estimators,
                max_depth=config.xgb_max_depth,
                learning_rate=config.xgb_learning_rate,
                subsample=config.xgb_subsample,
                colsample_bytree=config.xgb_colsample_bytree,
                objective="reg:squarederror",
                eval_metric="rmse",
                reg_lambda=1.0,
                random_state=config.random_state,
                n_jobs=4,
                verbosity=0,
            )
            model.fit(summary_features, targets)
        elif config.model_type == "ridge":
            model = ClassicalRidgeRULModel(random_state=config.random_state)
            model.fit(summary_features, targets)
        else:
            model = NaiveRULModel(default_rul=config.rul_reference)
            model.fit(targets)
        return cls(model, scaler, list(feature_columns), config)

    def calibrate_uncertainty(self, validation_frame: pd.DataFrame) -> Tuple[float, float]:
        """Calibrate terminal-engine intervals with finite-sample conformal quantiles.

        Each validation engine contributes one residual from a pseudo-current
        window. Validation engines are sampled across the configured RUL range,
        avoiding both correlated overlapping windows and the run-to-failure
        endpoint mismatch (whose RUL is always zero).
        """
        preds = self.predict_frame(validation_frame)
        if preds.empty or "rul" not in preds:
            return self.conformal_q90, self.conformal_q95
        calibration = select_rul_stratified_rows(
            preds,
            rul_reference=self.config.rul_reference,
            seed=self.config.random_state,
        )
        calibration_engine_count = int(calibration["unit_id"].nunique())
        if calibration_engine_count < 19:
            raise ValueError(
                "At least 19 calibration engines with valid windows are required "
                "for a finite 95% conformal quantile"
            )
        selected_reference_rul = calibration["rul"].to_numpy(dtype=float).tolist()
        actual = calibration["rul"].to_numpy(dtype=float)
        predicted = calibration["predicted_rul"].to_numpy(dtype=float)
        residuals = np.abs(actual - predicted)
        if not np.isfinite(residuals).all():
            raise ValueError("validation terminal residuals must be finite")

        def finite_sample_quantile(values: np.ndarray, level: float) -> float:
            # Split conformal uses the ceil((n + 1) * level)-th ordered value.
            # Selecting that order statistic directly avoids an interpolation
            # off-by-one caused by mapping rank/n through np.quantile(method="higher").
            order = ceil((len(values) + 1) * level)
            if order > len(values):
                raise ValueError(
                    f"{len(values)} calibration engines cannot provide a finite "
                    f"{level:.0%} conformal quantile"
                )
            return float(np.partition(values, order - 1)[order - 1])

        signed_overestimation = predicted - actual
        self.conformal_q90 = finite_sample_quantile(residuals, 0.90)
        self.conformal_q95 = finite_sample_quantile(residuals, 0.95)
        self.lower_bound_margin_95 = max(
            0.0, finite_sample_quantile(signed_overestimation, 0.95)
        )
        requested_cycles = tuple(
            getattr(self.config, "planning_margin_cycles", ())
        )
        if len(set(requested_cycles)) != len(requested_cycles):
            raise ValueError("planning_margin_cycles must not contain duplicates")
        if any(cycle < self.config.window_size for cycle in requested_cycles):
            raise ValueError(
                "planning_margin_cycles must be at least the prognostics window size"
            )
        self.planning_margins_by_cycle_ = {}
        self.planning_margin_engine_counts_by_cycle_ = {}
        for cycle in requested_cycles:
            cycle_rows = preds[preds["end_cycle"] == cycle]
            engine_count = int(cycle_rows["unit_id"].nunique())
            if engine_count < 19:
                raise ValueError(
                    f"At least 19 calibration engines with a cycle-{cycle} "
                    "prediction are required for a finite 95% planning margin"
                )
            cycle_overestimation = (
                cycle_rows["predicted_rul"].to_numpy(dtype=float)
                - cycle_rows["rul"].to_numpy(dtype=float)
            )
            if not np.isfinite(cycle_overestimation).all():
                raise ValueError(
                    f"cycle-{cycle} calibration residuals must be finite"
                )
            self.planning_margins_by_cycle_[int(cycle)] = max(
                0.0, finite_sample_quantile(cycle_overestimation, 0.95)
            )
            self.planning_margin_engine_counts_by_cycle_[int(cycle)] = engine_count
        self.n_calibration_engines_ = calibration_engine_count
        self.calibration_reference_rul_ = selected_reference_rul
        return self.conformal_q90, self.conformal_q95

    def _prepare(self, frame: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
        scaled = transform_features(frame, self.feature_columns, self.scaler)
        # Inference must work with telemetry-only input. A NaN target is added
        # solely because the shared window builder also returns target metadata;
        # it is discarded here and never reaches the model features.
        if "rul" not in scaled.columns:
            scaled = scaled.assign(rul=np.nan)
        windows, _, metadata = make_windows(
            scaled,
            self.feature_columns,
            window_size=self.config.window_size,
            target_column="rul",
        )
        if self.config.model_type in {
            "lstm",
            "pytorch_cnn",
            "pytorch_transformer",
        }:
            if getattr(self.config, "include_cycle_feature", False):
                windows = _append_cycle_channel(windows, metadata)
            return windows, metadata

        summary_features = summarize_windows(windows)
        if getattr(self.config, "include_cycle_feature", False):
            cycle_feature = (
                metadata["end_cycle"].to_numpy(dtype=np.float32)[:, None] / 100.0
            )
            summary_features = np.concatenate(
                [summary_features, cycle_feature], axis=1
            )
        return summary_features, metadata

    def predict_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Predict RUL for every valid window in a telemetry frame."""
        summary_features, metadata = self._prepare(frame)
        if len(summary_features) == 0:
            return metadata.assign(
                predicted_rul=np.array([], dtype=float),
                rul_uncertainty=np.array([], dtype=float),
                lower_rul=np.array([], dtype=float),
                upper_rul=np.array([], dtype=float),
                calibrated_lower_95=np.array([], dtype=float),
                calibrated_upper_95=np.array([], dtype=float),
                planning_rul_lower_95=np.array([], dtype=float),
                predicted_health_index=np.array([], dtype=float),
            )

        if hasattr(self.model, "estimators_"):
            tree_predictions = np.column_stack(
                [tree.predict(summary_features) for tree in self.model.estimators_]
            )
            predicted_rul = np.maximum(tree_predictions.mean(axis=1), 0.0)
            uncertainty = tree_predictions.std(axis=1)
        else:
            predicted_rul = np.maximum(self.model.predict(summary_features), 0.0)
            uncertainty = np.full(len(predicted_rul), self.conformal_q95 / 1.96)

        # Baseline heuristic dispersion interval
        lower_rul = np.maximum(predicted_rul - 1.96 * uncertainty, 0.0)
        upper_rul = predicted_rul + 1.96 * uncertainty

        # Calibrated conformal prediction interval
        cal_lower_95 = np.maximum(predicted_rul - self.conformal_q95, 0.0)
        cal_upper_95 = predicted_rul + self.conformal_q95
        # One-sided calibration uses only overestimation residuals. This is the
        # conservative RUL input used by the planner, separate from the
        # symmetric interval reported for uncertainty evaluation.
        # Older saved agents predate the one-sided margin. Their calibrated
        # symmetric margin remains a conservative compatible fallback.
        planning_margin = getattr(
            self, "lower_bound_margin_95", self.conformal_q95
        )
        cycle_margins = getattr(self, "planning_margins_by_cycle_", {})
        margins = np.asarray(
            [
                cycle_margins.get(int(cycle), planning_margin)
                for cycle in metadata["end_cycle"]
            ],
            dtype=float,
        )
        planning_lower_95 = np.maximum(predicted_rul - margins, 0.0)

        result = metadata.copy()
        result["predicted_rul"] = predicted_rul
        result["rul_uncertainty"] = uncertainty
        result["lower_rul"] = lower_rul
        result["upper_rul"] = upper_rul
        result["calibrated_lower_95"] = cal_lower_95
        result["calibrated_upper_95"] = cal_upper_95
        result["planning_rul_lower_95"] = planning_lower_95
        result["predicted_health_index"] = np.clip(
            1.0 - predicted_rul / self.config.rul_reference, 0.0, 1.0
        )
        return result

    def predict_current(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return the latest prediction for each engine in a telemetry frame."""
        predictions = self.predict_frame(frame)
        if predictions.empty:
            return predictions
        return predictions.sort_values(["unit_id", "end_cycle"]).groupby(
            "unit_id", sort=True
        ).tail(1)

    def predict_current_available_history(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return a terminal prediction for every unit using only observed rows.

        Classical window-summary models have a fixed summary-vector width even
        when the input history is shorter than the training window. For such a
        unit, summarize its available observations directly rather than
        dropping the unit or fabricating padded telemetry.
        """

        if self.config.model_type in {
            "lstm",
            "pytorch_cnn",
            "pytorch_transformer",
        }:
            raise ValueError(
                "Available-history terminal inference is supported only for "
                "classical window-summary models"
            )
        if not {"unit_id", "cycle"}.issubset(frame.columns):
            raise KeyError("Terminal inference requires unit_id and cycle metadata")
        if frame.empty:
            return frame.iloc[0:0].copy()

        terminal = self.predict_current(frame)
        included_units = set(terminal["unit_id"].astype(int)) if not terminal.empty else set()
        short_rows: list[dict[str, Any]] = []
        for unit_id, unit in frame.groupby("unit_id", sort=True):
            if int(unit_id) in included_units:
                continue
            unit = unit.sort_values("cycle")
            if unit.empty:
                continue
            scaled = transform_features(unit, self.feature_columns, self.scaler)
            values = scaled[self.feature_columns].to_numpy(dtype=np.float32)[
                -self.config.window_size :
            ]
            if len(values) >= 2:
                summary = summarize_windows(values[None, :, :])
            else:
                # Match summarize_windows' [last, mean, std, min, max, slope]
                # layout for a single observed sample.
                summary = np.concatenate(
                    [
                        values[-1],
                        values.mean(axis=0),
                        np.zeros(values.shape[1], dtype=np.float32),
                        values.min(axis=0),
                        values.max(axis=0),
                        np.zeros(values.shape[1], dtype=np.float32),
                    ]
                )[None, :]
            if self.config.include_cycle_feature:
                summary = np.concatenate(
                    [
                        summary,
                        np.asarray(
                            [[float(unit["cycle"].iloc[-1]) / 100.0]],
                            dtype=np.float32,
                        ),
                    ],
                    axis=1,
                )

            if hasattr(self.model, "estimators_"):
                tree_predictions = np.asarray(
                    [tree.predict(summary)[0] for tree in self.model.estimators_],
                    dtype=np.float64,
                )
                predicted_rul = max(0.0, float(tree_predictions.mean()))
                uncertainty = float(tree_predictions.std())
            else:
                predicted_rul = max(0.0, float(self.model.predict(summary)[0]))
                uncertainty = self.conformal_q95 / 1.96

            end_cycle = int(unit["cycle"].iloc[-1])
            if "rul" in unit:
                target_rul = float(unit["rul"].iloc[-1])
            else:
                target_rul = float("nan")
            cycle_margin = getattr(self, "planning_margins_by_cycle_", {}).get(
                end_cycle,
                getattr(self, "lower_bound_margin_95", self.conformal_q95),
            )
            short_rows.append(
                {
                    "unit_id": int(unit_id),
                    "end_cycle": end_cycle,
                    "rul": target_rul,
                    "predicted_rul": predicted_rul,
                    "rul_uncertainty": uncertainty,
                    "lower_rul": max(0.0, predicted_rul - 1.96 * uncertainty),
                    "upper_rul": predicted_rul + 1.96 * uncertainty,
                    "calibrated_lower_95": max(
                        0.0, predicted_rul - self.conformal_q95
                    ),
                    "calibrated_upper_95": predicted_rul + self.conformal_q95,
                    "planning_rul_lower_95": max(
                        0.0, predicted_rul - float(cycle_margin)
                    ),
                    "predicted_health_index": float(
                        np.clip(
                            1.0 - predicted_rul / self.config.rul_reference,
                            0.0,
                            1.0,
                        )
                    ),
                }
            )

        if short_rows:
            terminal = pd.concat(
                [terminal, pd.DataFrame(short_rows)], ignore_index=True
            )
        return terminal.sort_values(["unit_id", "end_cycle"]).reset_index(drop=True)

    def save(self, path: Path | str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path | str) -> "PrognosticsAgent":
        agent = joblib.load(path)
        if not isinstance(agent, cls):
            raise TypeError(f"Expected PrognosticsAgent, received {type(agent).__name__}")
        return agent


def benchmark_all_prognostics(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    feature_columns: list[str],
    window_size: int = 30,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train and compare Naive, Linear, Random Forest, and LSTM sequence prognostics models."""
    scaler = fit_feature_scaler(train_frame, feature_columns)
    tr_scaled = transform_features(train_frame, feature_columns, scaler)
    va_scaled = transform_features(val_frame, feature_columns, scaler)
    te_scaled = transform_features(test_frame, feature_columns, scaler)

    tr_win, tr_y, tr_meta = make_windows(tr_scaled, feature_columns, window_size, "rul")
    va_win, va_y, va_meta = make_windows(va_scaled, feature_columns, window_size, "rul")
    te_win, te_y, te_meta = make_windows(te_scaled, feature_columns, window_size, "rul")

    tr_sum = summarize_windows(tr_win)
    va_sum = summarize_windows(va_win)
    te_sum = summarize_windows(te_win)

    results: Dict[str, Any] = {}

    # 1. Naive Baseline
    naive = NaiveRULModel().fit(tr_y)
    va_naive_pred = naive.predict(va_sum)
    te_naive_pred = naive.predict(te_sum)
    results["naive_baseline"] = {
        "validation": compute_rul_metrics(va_y, va_naive_pred),
        "test": compute_rul_metrics(te_y, te_naive_pred),
    }

    # 2. Classical Linear Baseline (Ridge)
    ridge = ClassicalRidgeRULModel(alpha=10.0, random_state=seed).fit(tr_sum, tr_y)
    va_ridge_pred = ridge.predict(va_sum)
    te_ridge_pred = ridge.predict(te_sum)
    results["classical_ridge"] = {
        "validation": compute_rul_metrics(va_y, va_ridge_pred),
        "test": compute_rul_metrics(te_y, te_ridge_pred),
    }

    # 3. Random Forest Baseline (preserved existing)
    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=2, max_features=0.7, random_state=seed, n_jobs=-1)
    rf.fit(tr_sum, tr_y)
    va_rf_pred = rf.predict(va_sum)
    te_rf_pred = rf.predict(te_sum)
    results["random_forest_baseline"] = {
        "validation": compute_rul_metrics(va_y, va_rf_pred),
        "test": compute_rul_metrics(te_y, te_rf_pred),
    }

    # 4. LSTM Sequence Regressor
    lstm_rul = NumpyLSTMRULModel(in_features=len(feature_columns), hidden_size=32, seed=seed)
    lstm_rul.fit(tr_win, tr_y, va_win, va_y, max_epochs=8, batch_size=128, lr=1e-3)
    va_lstm_pred = lstm_rul.predict(va_win)
    te_lstm_pred = lstm_rul.predict(te_win)
    results["lstm_sequence_regressor"] = {
        "validation": compute_rul_metrics(va_y, va_lstm_pred),
        "test": compute_rul_metrics(te_y, te_lstm_pred),
    }

    # Terminal Engine Metrics for Test Set
    te_meta = te_meta.copy()
    terminal_idx = te_meta.groupby("unit_id", sort=True).tail(1).index
    t_y = te_y[terminal_idx]
    results["test_terminal_metrics"] = {
        "naive": compute_rul_metrics(t_y, te_naive_pred[terminal_idx]),
        "ridge": compute_rul_metrics(t_y, te_ridge_pred[terminal_idx]),
        "random_forest": compute_rul_metrics(t_y, te_rf_pred[terminal_idx]),
        "lstm": compute_rul_metrics(t_y, te_lstm_pred[terminal_idx]),
    }

    # Early-, mid-, and late-life breakdown (boundaries are explicit).
    early_mask = te_y > 125
    mid_mask = (te_y > 50) & (te_y <= 125)
    late_mask = te_y <= 50
    results["test_life_stage_mae"] = {
        "early_life_rul_gt_125": {
            "naive": float(mean_absolute_error(te_y[early_mask], te_naive_pred[early_mask])),
            "ridge": float(mean_absolute_error(te_y[early_mask], te_ridge_pred[early_mask])),
            "random_forest": float(mean_absolute_error(te_y[early_mask], te_rf_pred[early_mask])),
            "lstm": float(mean_absolute_error(te_y[early_mask], te_lstm_pred[early_mask])),
        },
        "mid_life_50_lt_rul_le_125": {
            "naive": float(mean_absolute_error(te_y[mid_mask], te_naive_pred[mid_mask])),
            "ridge": float(mean_absolute_error(te_y[mid_mask], te_ridge_pred[mid_mask])),
            "random_forest": float(mean_absolute_error(te_y[mid_mask], te_rf_pred[mid_mask])),
            "lstm": float(mean_absolute_error(te_y[mid_mask], te_lstm_pred[mid_mask])),
        },
        "late_life_rul_le_50": {
            "naive": float(mean_absolute_error(te_y[late_mask], te_naive_pred[late_mask])),
            "ridge": float(mean_absolute_error(te_y[late_mask], te_ridge_pred[late_mask])),
            "random_forest": float(mean_absolute_error(te_y[late_mask], te_rf_pred[late_mask])),
            "lstm": float(mean_absolute_error(te_y[late_mask], te_lstm_pred[late_mask])),
        },
    }

    # Keep this separate benchmark's endpoint-only calibration explicit. The
    # deployed PrognosticsAgent uses RUL-stratified pseudo-current calibration.
    rf_tree_preds = np.column_stack([t.predict(te_sum) for t in rf.estimators_])
    rf_std = rf_tree_preds.std(axis=1)
    val_terminal_idx = va_meta.groupby("unit_id", sort=True).tail(1).index
    test_terminal_idx = te_meta.groupby("unit_id", sort=True).tail(1).index
    dispersion_lower = te_rf_pred[test_terminal_idx] - 1.96 * rf_std[test_terminal_idx]
    dispersion_upper = te_rf_pred[test_terminal_idx] + 1.96 * rf_std[test_terminal_idx]
    dispersion_coverage = float((
        (te_y[test_terminal_idx] >= dispersion_lower)
        & (te_y[test_terminal_idx] <= dispersion_upper)
    ).mean())
    va_terminal_res = np.abs(va_y[val_terminal_idx] - va_rf_pred[val_terminal_idx])
    conformal_level = min(1.0, ceil((len(va_terminal_res) + 1) * 0.95) / len(va_terminal_res))
    q95 = float(np.quantile(va_terminal_res, conformal_level, method="higher"))
    conformal_lower = np.maximum(te_rf_pred[test_terminal_idx] - q95, 0.0)
    conformal_upper = te_rf_pred[test_terminal_idx] + q95
    conformal_coverage = float((
        (te_y[test_terminal_idx] >= conformal_lower)
        & (te_y[test_terminal_idx] <= conformal_upper)
    ).mean())

    results["uncertainty_calibration"] = {
        "scope": "separate comparative Random Forest benchmark; not the deployed PrognosticsAgent intervals",
        "calibration_point_selection": "last available window per validation engine (failure endpoint)",
        "evaluation_point_selection": "last available window per official test engine",
        "rf_tree_dispersion_coverage_test": dispersion_coverage,
        "rf_tree_dispersion_mean_interval_width_cycles": float(
            (dispersion_upper - dispersion_lower).mean()
        ),
        "rf_tree_dispersion_coverage_error": abs(dispersion_coverage - 0.95),
        "conformal_q95_test_coverage": conformal_coverage,
        "conformal_q95_margin_cycles": q95,
        "conformal_q95_mean_interval_width_cycles": float(
            (conformal_upper - conformal_lower).mean()
        ),
        "conformal_q95_coverage_error": abs(conformal_coverage - 0.95),
        "conformal_calibration_engines": int(len(va_terminal_res)),
        "conformal_evaluation_engines": int(len(test_terminal_idx)),
        "calibration_verified": bool(conformal_coverage >= 0.90),
    }

    return results
