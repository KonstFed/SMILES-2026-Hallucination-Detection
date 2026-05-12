"""
aggregation.py — Phase 1 ensemble feature extraction.

Outputs two concatenated feature blocks consumed by the multi-probe
ensemble in ``probe.py``:

  * Block A (896-d) — last-token activation at layer 24. Strongest single
    layer per the layer sweep; main signal source for the linear and MLP
    sub-probes.

  * Block C (13-d) — geometric signal:
      - 9 trajectory features (step norms, step cosines, straightness,
        late-layer convergence, embedding/final magnitudes)
      - 1 EigenScore over response tokens at layer 24
      - 3 response-token norm stats at layer 24 (mean / std / max)
    Standalone trajectory features showed only a 7pt train/test gap
    (vs 25pt for activation-MLP), so this block has a fundamentally
    different failure mode and can contribute to ensemble diversity.

Block layout: ``X = [block_a (896) | block_c (13)]``. Probe slices on
``BLOCK_A_DIM`` and routes each block to its dedicated sub-probe.

The prompt/response boundary (needed for response-only stats) is found
without changing ``solution.py`` by fingerprinting the ``<|im_start|>``
embedding from the first real token of every ChatML prompt.
"""

from __future__ import annotations

import torch


LAYER = 24                       # strongest single layer per the layer sweep
HIDDEN_DIM = 896                 # Qwen2.5-0.5B
ROLE_PREFIX_LEN = 3              # tokens for "<|im_start|>assistant\n"
EIGEN_ALPHA = 1e-3               # ridge term for log det stability

BLOCK_A_DIM = HIDDEN_DIM
BLOCK_C_DIM = 13
FEATURE_DIM = BLOCK_A_DIM + BLOCK_C_DIM


def _response_span(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[int, int]:
    """Return ``(response_start, last_real)`` token positions."""
    real_positions = attention_mask.nonzero(as_tuple=False).squeeze(-1)
    first_real = int(real_positions[0].item())
    last_real = int(real_positions[-1].item())

    emb = hidden_states[0]
    fingerprint = emb[first_real]
    sims = (emb @ fingerprint) / (
        emb.norm(dim=-1) * fingerprint.norm() + 1e-8
    )

    real_mask = torch.zeros_like(sims, dtype=torch.bool)
    real_mask[real_positions] = True
    matches = ((sims > 0.99) & real_mask).nonzero(as_tuple=False).squeeze(-1)

    if matches.numel() == 0:
        response_start = first_real
    else:
        last_im_start = int(matches[-1].item())
        response_start = min(last_im_start + ROLE_PREFIX_LEN, last_real)

    return response_start, last_real


def _eigen_score(H: torch.Tensor, alpha: float = EIGEN_ALPHA) -> torch.Tensor:
    """``(1/T) * log det(H Hᵀ + αI)`` — dispersion of the token manifold."""
    T = H.shape[0]
    if T == 0:
        return torch.zeros((), dtype=H.dtype)

    gram = H @ H.T
    gram = gram + alpha * torch.eye(T, dtype=gram.dtype, device=gram.device)
    sign, logdet = torch.slogdet(gram)

    if sign <= 0 or torch.isnan(logdet) or torch.isinf(logdet):
        return torch.log(gram.diagonal().sum() + 1e-8) / T

    return logdet / T


def _block_a(hidden_states: torch.Tensor, last_real: int) -> torch.Tensor:
    """Last real token of layer 24 → 896-d."""
    return hidden_states[LAYER][last_real]


def _block_c(
    hidden_states: torch.Tensor,
    response_start: int,
    last_real: int,
) -> torch.Tensor:
    """Trajectory (9) + EigenScore (1) + response norm stats (3) → 13-d."""
    eps = 1e-8

    # ── Trajectory features at last real token across all 25 layers ────
    n_layers = hidden_states.shape[0]
    u = torch.stack(
        [hidden_states[k][last_real] for k in range(n_layers)], dim=0
    )

    steps = u[1:] - u[:-1]
    step_norms = steps.norm(dim=-1)

    a = steps[1:]
    b = steps[:-1]
    step_cos = (a * b).sum(dim=-1) / (
        a.norm(dim=-1) * b.norm(dim=-1) + eps
    )

    net_disp = (u[-1] - u[0]).norm()
    total_arc = step_norms.sum()
    straightness = net_disp / (total_arc + eps)

    quartile = max(1, n_layers // 4)
    a_late = u[-quartile - 1 : -1]
    b_late = u[-quartile:]
    late_cos = (a_late * b_late).sum(dim=-1) / (
        a_late.norm(dim=-1) * b_late.norm(dim=-1) + eps
    )

    traj = torch.stack([
        step_norms.mean(),
        step_norms.max(),
        step_norms.std(unbiased=False),
        step_cos.mean(),
        step_cos.min(),
        straightness,
        late_cos.mean(),
        u[0].norm(),
        u[-1].norm(),
    ])  # (9,)

    # ── Response-only features at layer 24 ─────────────────────────────
    h = hidden_states[LAYER]
    response_slice = h[response_start : last_real + 1]
    eigen = _eigen_score(response_slice).reshape(1)

    if response_slice.shape[0] == 0:
        norm_stats = torch.zeros(3, dtype=h.dtype)
    else:
        token_norms = response_slice.norm(dim=-1)
        norm_stats = torch.stack([
            token_norms.mean(),
            token_norms.std(unbiased=False),
            token_norms.max(),
        ])

    return torch.cat(
        [traj, eigen.to(traj.dtype), norm_stats.to(traj.dtype)], dim=0
    )


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Concatenate Block A (last token L24) + Block C (geometric)."""
    response_start, last_real = _response_span(hidden_states, attention_mask)

    block_a = _block_a(hidden_states, last_real)
    block_c = _block_c(hidden_states, response_start, last_real)

    return torch.cat([block_a, block_c.to(block_a.dtype)], dim=0)


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Optional geometric features. Block C is now baked into ``aggregate``."""
    return torch.zeros(0, dtype=torch.float32)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    agg_features = aggregate(hidden_states, attention_mask)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features
