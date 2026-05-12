"""
probe.py — Phase 1 ensemble probe (3 sub-probes, uniform weights).

Routes the two feature blocks from ``aggregation.py`` to dedicated
sub-probes, then averages their probabilities (uniform 1/3 each):

  * ``LDA-shrinkage`` on Block A (L24, 896-d) — calibrated linear, no memo
  * ``MLP-256`` on Block A — non-linear, more AUROC, memorizes
  * ``LDA-shrinkage`` on Block C (13-d geometric) — different signal type

Each block has its own ``StandardScaler`` so feature scales don't bleed
between blocks. Threshold is tuned on validation for accuracy and the
chosen value is printed each fold so it can be averaged and hardcoded
into ``__init__`` (the final submission probe in ``solution.py`` calls
``fit()`` only — never ``fit_hyperparameters()`` — so the hardcoded
default is what predictions.csv actually uses).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from aggregation import BLOCK_A_DIM, FEATURE_DIM


class HallucinationProbe(nn.Module):
    """Ensemble of three sub-probes routed by feature block."""

    def __init__(self) -> None:
        super().__init__()
        self._scaler_a = StandardScaler()
        self._scaler_c = StandardScaler()

        self._lda_a = LinearDiscriminantAnalysis(
            solver="lsqr", shrinkage="auto"
        )
        self._lda_c = LinearDiscriminantAnalysis(
            solver="lsqr", shrinkage="auto"
        )
        self._mlp: nn.Sequential | None = None

        # TODO: replace with average of per-fold thresholds printed by
        # fit_hyperparameters() once measured.
        self._threshold: float = 0.5

    def _split_blocks(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return X[:, :BLOCK_A_DIM], X[:, BLOCK_A_DIM:]

    def _build_mlp(self, input_dim: int) -> None:
        self._mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def _train_mlp(self, X_scaled: np.ndarray, y: np.ndarray) -> None:
        torch.manual_seed(0)
        self._build_mlp(X_scaled.shape[1])

        X_t = torch.from_numpy(X_scaled).float()
        y_t = torch.from_numpy(y.astype(np.float32))

        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(self._mlp.parameters(), lr=1e-3)

        self._mlp.train()
        for _ in range(200):
            optimizer.zero_grad()
            logits = self._mlp(X_t).squeeze(-1)
            loss = criterion(logits, y_t)
            loss.backward()
            optimizer.step()
        self._mlp.eval()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        if X.shape[1] != FEATURE_DIM:
            raise ValueError(
                f"Expected {FEATURE_DIM} features, got {X.shape[1]}. "
                "Check aggregation.py block dims."
            )

        X_a, X_c = self._split_blocks(X)
        X_a_scaled = self._scaler_a.fit_transform(X_a)
        X_c_scaled = self._scaler_c.fit_transform(X_c)

        self._lda_a.fit(X_a_scaled, y)
        self._lda_c.fit(X_c_scaled, y)
        self._train_mlp(X_a_scaled, y)
        return self

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        """Tune the decision threshold on a validation set to maximise accuracy."""
        probs = self.predict_proba(X_val)[:, 1]
        candidates = np.unique(np.concatenate([probs, np.linspace(0.0, 1.0, 101)]))

        best_threshold = 0.5
        best_acc = -1.0
        for t in candidates:
            y_pred_t = (probs >= t).astype(int)
            score = accuracy_score(y_val, y_pred_t)
            if score > best_acc:
                best_acc = score
                best_threshold = float(t)

        self._threshold = best_threshold
        # Per-fold threshold logging — collect these across all folds and
        # average them into the __init__ default to fix the eval/submission
        # mismatch (solution.py never calls fit_hyperparameters).
        print(f"  [probe] tuned threshold = {best_threshold:.4f}  (val acc = {best_acc:.4f})")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_a, X_c = self._split_blocks(X)
        X_a_scaled = self._scaler_a.transform(X_a)
        X_c_scaled = self._scaler_c.transform(X_c)

        lda_a_proba = self._lda_a.predict_proba(X_a_scaled)
        lda_c_proba = self._lda_c.predict_proba(X_c_scaled)

        X_t = torch.from_numpy(X_a_scaled).float()
        with torch.no_grad():
            mlp_logits = self._mlp(X_t).squeeze(-1)
            mlp_pos = torch.sigmoid(mlp_logits).numpy()
        mlp_proba = np.stack([1.0 - mlp_pos, mlp_pos], axis=1)

        return (lda_a_proba + mlp_proba + lda_c_proba) / 3.0
