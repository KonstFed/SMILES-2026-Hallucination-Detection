"""
aggregation.py — Token aggregation strategy and feature extraction
               (student-implemented).

Converts per-token, per-layer hidden states from the extraction loop in
``solution.py`` into flat feature vectors for the probe classifier.

Two stages can be customised independently:

  1. ``aggregate`` — select layers and token positions, pool into a vector.
  2. ``extract_geometric_features`` — optional hand-crafted features
     (enabled by setting ``USE_GEOMETRIC = True`` in ``solution.py``).

Both stages are combined by ``aggregation_and_feature_extraction``, the
single entry point called from the notebook.
"""

from __future__ import annotations

import torch


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Convert per-token hidden states into a single feature vector.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
                        Layer index 0 is the token embedding; index -1 is the
                        final transformer layer.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D feature tensor of shape ``(hidden_dim,)`` or
        ``(k * hidden_dim,)`` if multiple layers are concatenated.

    Student task:
        Replace or extend the skeleton below with alternative layer selection,
        token pooling (mean, max, weighted), or multi-layer fusion strategies.
    """
    # ------------------------------------------------------------------
    # STUDENT: Replace or extend the aggregation below.
    # ------------------------------------------------------------------

    # Last real token of the final transformer layer.
    layer = hidden_states[-1]                                 # (seq_len, hidden_dim)
    real_positions = attention_mask.nonzero(as_tuple=False)   # (n_real, 1)
    last_pos = int(real_positions[-1].item())
    feature = layer[last_pos]                                  # (hidden_dim,)

    return feature
    # ------------------------------------------------------------------


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Extract hand-crafted geometric / statistical features from hidden states.

    Called only when ``USE_GEOMETRIC = True`` in ``solution.ipynb``.  The
    returned tensor is concatenated with the output of ``aggregate``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D float tensor of shape ``(n_geometric_features,)``.  The length
        must be the same for every sample.

    Student task:
        Replace the stub below.  Possible features: layer-wise activation
        norms, inter-layer cosine similarity (representation drift), or
        sequence length.
    """
    # ------------------------------------------------------------------
    # STUDENT: Replace or extend the geometric feature extraction below.
    # ------------------------------------------------------------------

    # 10 hand-crafted features, all computed at the last real token:
    #   1  — sequence length (count of real tokens)
    #   4  — last-token L2 norms at layers {6, 12, 18, 24}
    #   4  — cosine(h[k], h[k+1]) at last token for k ∈ {6, 12, 18, 22}
    #   1  — cosine(h[0], h[24]) at last token (overall transformation)
    real_positions = attention_mask.nonzero(as_tuple=False)
    last_pos = int(real_positions[-1].item())
    n_real = float(attention_mask.sum().item())

    device = hidden_states.device
    dtype = hidden_states.dtype
    eps = 1e-8

    def _cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a * b).sum() / (a.norm() * b.norm() + eps)

    feats: list[torch.Tensor] = []

    # 1. Sequence length (longer responses sometimes correlate with bullshit).
    feats.append(torch.tensor([n_real], dtype=dtype, device=device))

    # 2. Per-layer last-token norms (activation magnitude across depth).
    for k in (6, 12, 18, 24):
        feats.append(hidden_states[k][last_pos].norm().unsqueeze(0))

    # 3. Inter-layer drift: cosine between consecutive layers' last tokens.
    for k in (6, 12, 18, 22):
        a = hidden_states[k][last_pos]
        b = hidden_states[k + 1][last_pos]
        feats.append(_cos(a, b).unsqueeze(0))

    # 4. Total transformation: cosine between embedding and final layer.
    a = hidden_states[0][last_pos]
    b = hidden_states[24][last_pos]
    feats.append(_cos(a, b).unsqueeze(0))

    return torch.cat(feats, dim=0).float()  # (10,)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    """Aggregate hidden states and optionally append geometric features.

    Main entry point called from ``solution.ipynb`` for each sample.
    Concatenates the output of ``aggregate`` with that of
    ``extract_geometric_features`` when ``use_geometric=True``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``
                        for a single sample.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.
        use_geometric:  Whether to append geometric features.  Controlled by
                        the ``USE_GEOMETRIC`` flag in ``solution.ipynb``.

    Returns:
        A 1-D float tensor of shape ``(feature_dim,)`` where
        ``feature_dim = hidden_dim`` (or larger for multi-layer or geometric
        concatenations).
    """
    agg_features = aggregate(hidden_states, attention_mask)  # (feature_dim,)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features
