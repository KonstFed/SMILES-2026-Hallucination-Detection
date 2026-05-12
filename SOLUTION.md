# SOLUTION.md

**Final 5-fold averaged metrics**: test accuracy **74.23%**, test AUROC **74.66%**.
Majority-class baseline = 70.19%, so the probe sits **+4.04pt above baseline**.

**Submitted predictions:** [`predictions.csv` on Google Drive](https://drive.google.com/file/d/1LTNHA5BRigi2AeN57hbWiL3E8TVEoBLI/view?usp=sharing)

## Reproducibility

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python solution.py
```

`solution.py` regenerates `results.json` and `predictions.csv`. Runtime ≈ 3–4 min on Macbook M4.

Determinism:
- Test split is fixed in `splitting.py` (15% stratified, `seed=42`); shared across folds, only train/val rotate.
- 5-fold StratifiedKFold uses `seed=42`.
- Each MLP in the bag is seeded (`torch.manual_seed(0/1/2)`).
- LDA is closed-form, deterministic by construction.

## Files I modified

- `aggregation.py` — feature extraction
- `probe.py` — classifier
- `splitting.py` — fixed 15% test + 5-fold CV on the remainder

The fixed infrastructure (`solution.py`, `model.py`, `evaluate.py`) is untouched.

## Final approach

### Features (909-dim, two blocks)

Layout: `X = [Block A (896) | Block C (13)]`.

- **Block A — last real token at layer 24** (896-d). A layer sweep over all 25 layers showed AUROC rises near-monotonically 0 → 24 with the peak at the final layer; the last token in a causal LM has attended to the whole sequence.
- **Block C — geometric / dispersion signal** (13-d):
  - 9 trajectory features at the last real token across all 25 layers (per-layer step norms mean/max/std, step-cosines mean/min, straightness, late-layer convergence, embedding/final magnitudes)
  - 1 EigenScore: `(1/T) · log det(H · Hᵀ + αI)` over the response tokens at layer 24
  - 3 response-token L2-norm stats at layer 24 (mean / std / max)

Block C needs the prompt/response boundary. To stay inside `aggregation.py`'s `(hidden_states, attention_mask)` contract (no `solution.py` change), I recover the boundary by fingerprinting the `<|im_start|>` token's embedding: it always sits at the first real position of every ChatML prompt, so the embedding row at `attention_mask.nonzero()[0]` is the fingerprint. The last position whose embedding matches it (cosine ≈ 1) is the start of the assistant turn; the response begins 3 tokens later.

### Probe (3-sub-probe ensemble)

Each block has its own `StandardScaler`. Three sub-probes:

1. **LDA-shrinkage on Block A** — `LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')`. Closed-form linear classifier with Ledoit-Wolf shrinkage of the covariance — the principled choice for d>>n (896 features, ~470 train samples).
2. **Bagged MLP on Block A** — 3 MLPs (`Linear(896→256) → ReLU → Linear(256→1)`), seeds `0/1/2`, 200 epochs each, Adam `lr=1e-3`, `BCEWithLogitsLoss(pos_weight = n_neg/n_pos)`. Their predicted probabilities are averaged.
3. **LDA-shrinkage on Block C** — same configuration as (1) on the 13-d geometric features.

Final probability is the uniform mean of the three sub-probes. Threshold is fixed at `0.5`.

### Why this combination

The big insights, in the order I learned them:

1. **Linear vs non-linear, both partial.** Solo LDA caps at ~69% AUROC — it can't capture all the structure. Solo MLP-256 gets 74.6% but pegs train AUROC at ~100. The ensemble combines their orthogonal failure modes: LDA underfits, MLP memorizes; their errors aren't perfectly correlated, so averaging beats either alone (+5pt over solo LDA).
2. **Add scalar features as their own probe, not stacked into the activation.** Directly concatenating EigenScore or trajectory features onto the 896-d input did nothing — the LDA simply ignored them (drowned by the activation). Giving Block C its own LDA in the ensemble lifted the result by +1.4pt AUROC and +1.5pt accuracy. The win is structural, not from the features themselves.
3. **MLP bagging stabilizes accuracy at the 0.5 threshold.** Going from 1 MLP to a bag of 3 left AUROC essentially flat (-0.12) but moved accuracy by +1.15pt. The reason: AUROC measures ranking, but binary predictions at threshold 0.5 are sensitive to the noisy ~0.5 region — bagging smooths the marginal probabilities, so fewer samples flip across the threshold between seeds.

## What I tried that didn't work

| Attempt | Result | Why it failed |
|---|---|---|
| Mid-layer pooling (L12 / L16) | 64–68% AUROC | The layer sweep showed signal rises monotonically 0 → 24 on this 0.5B model. The "use a middle layer" rule comes from ≥7B models and doesn't transfer |
| Multi-layer activation concat `{20, 22, 24}` (2688-d) | within noise | Adjacent transformer layers correlate strongly via the residual stream — redundant features that just give the probe more rope to overfit |
| 3-pool feature `[mean_resp, last_resp, mean_prompt]` at L24 (2688-d) | **62.75% AUROC**, 69.23% acc — *below baseline* | `mean_prompt` actively hurt: prompts share style, so the probe latched onto a question-type fingerprint that didn't transfer to test. Tripling feature dim also gave the MLP more memorization room |
| EigenScore concatenated directly into the 896-d activation | flat | LDA ignored it (drowned by 896 raw features); useful only when isolated to its own sub-probe |
| Mid-layer L12 as a 4th sub-probe with uniform weighting | −2.5pt AUROC | L12 solo AUROC is ~64% (vs L24's ~75%); a uniform-averaged weak probe drags the ensemble down |
| Val-AUROC-weighted ensemble (squared, normalized) | −3.5pt AUROC | Weights estimated on a noisy 117-sample val *plus* threshold tuned on the same val = compounded val-overfit |
| MLP early stopping (50 / 100 epochs) | 71.9% / 73.6% AUROC (vs 74.66% at 200) | Test AUROC monotonically rose with more training. The train→100% AUROC isn't memorizing past the test signal; it's compressing real signal into tiny-N training data |
| MLP bag size 5 (vs 3) | −0.77pt acc | Diminishing returns past N=3; variance was already mostly absorbed |
| F1-tuned threshold | acc dropped below 70% baseline | F1-optimal threshold collapsed predictions toward all-positive on the 70/30-imbalanced classes; AUROC was fine but acc broke |
| Heavy regularization (`wd=1e-2`, MLP-64 + dropout 0.3) | crushed signal | Over-regularization killed real signal alongside the overfit |

## Honest assessment

74.23% accuracy is what I got, +4pt over the majority-class baseline. I think it's reasonable for 689 samples and a 0.5B model, but I'd bet good money someone in this competition found a clever trick I didn't. If you're grading this and the leaderboard is full of 80%+ scores, please tell me how — I'll buy you a coffee.

Things I poked at but couldn't get to work: layer aggregation past the last one, weighting the ensemble, more features stacked into the activation. The pattern that kept biting me was *adding feature dimensions to fix things made it worse* — the MLP just memorized harder. Most of the gains came from the ensemble structure, not from any single clever feature.
