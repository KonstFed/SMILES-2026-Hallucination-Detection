"""
splitting.py — Train / validation / test split utilities (student-implementable).

``split_data`` receives the label array ``y`` and, optionally, the full
DataFrame ``df`` (for group-aware splits).  It must return a list of
``(idx_train, idx_val, idx_test)`` tuples of integer index arrays.

Contract
--------
* ``idx_train``, ``idx_val``, ``idx_test`` are 1-D NumPy arrays of integer
  indices into the full dataset.
* ``idx_val`` may be ``None`` if no separate validation fold is needed.
* All indices must be non-overlapping; together they must cover every sample.
* Return a **list** — one element for a single split, K elements for k-fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    """Split dataset indices into train, validation, and test subsets.

    Reserves a fixed stratified test set (~15 %), then applies 5-fold
    stratified CV to the remaining samples so model selection is less noisy.

    Args:
        y:            Label array of shape ``(N,)`` with values in ``{0, 1}``.
        df:           Unused; kept for interface compatibility.
        test_size:    Fraction of samples reserved for the held-out test set.
        val_size:     Unused; fold size is determined by n_splits.
        random_state: Random seed for reproducibility.

    Returns:
        A list of 5 ``(idx_train, idx_val, idx_test)`` tuples.
    """
    idx = np.arange(len(y))

    idx_trainval, idx_test = train_test_split(
        idx,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    splits = []
    for rel_train, rel_val in kf.split(idx_trainval, y[idx_trainval]):
        idx_train = idx_trainval[rel_train]
        idx_val = idx_trainval[rel_val]
        splits.append((idx_train, idx_val, idx_test))

    return splits
