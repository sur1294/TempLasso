# Provenance of `models/alter/`

## Source

| field | value |
|---|---|
| Repository | https://github.com/yushuowiki/ALTER |
| Commit | `ad5c5a95d54aa2a0a2a275c008a692d73af293f7` |
| Commit date | 2025-09-13 |
| Retrieved | 2026-09-10 |
| License | **none published** — the upstream repository contains no LICENSE file. The code is reproduced here for research reproduction with explicit attribution and with the original docstrings, comments and third-party notices preserved. |
| Paper | ALTER: Adaptive Long-range aware TransformER for brain graphs. Model key `lrbgt`; class name in code `BrainNetworkTransformer`. |
| Third-party | `ptdec/` carries the upstream note "From https://github.com/vlukiyanov/pt-dec". |

### Why vendored rather than referenced

The collaborator pointed at `https://anonymous.4open.science/r/EDGEQUADD-3A6E`, whose
`zip_all_7_models.sh` lists seven baseline folders (`ALTER`, `BioBGT`, `BQN-corrected`,
`brainetcnn`, `gcn_gps_experiments`, `graphtransformer`, `QUETT`). Only `QUETT/` and
`gcn_gps_experiments/` were actually uploaded (47 files, fully enumerated 2026-09-10), and
`gcn_gps_experiments/main.py` imports a `models` package that is not in the archive. There is
no ALTER source in the anonymous repository, so the public upstream release is the only
available reference implementation.

### Upstream file checksums (SHA-256 of the retrieved files)

```
bdd3e1d9770420b1dbd511773e87bfe0bb4bf3e87cf5ea7d81aa7967d5bd529a  alter/models/LRBGT/lrbgt.py
283e41edad502a4a4f366a596a649ce328d732391cdcfd52ff2716ca8549f5e1  alter/models/LRBGT/ptdec/dec.py
218c73f6edecda5e49d0045ffd22dc15fc4d05372d0ae289c86a17c81dd63c9d  alter/models/LRBGT/ptdec/cluster.py
883ff1f46a1ca58ca4639926cbb233b031af259f95967696450150d2ea800612  alter/models/LRBGT/components/transformer_encoder.py
9fe4d46d6ddab194e5761ce86892170b99c39aed091724d369974b7936c061c7  alter/conf/model/lrbgt.yaml
78bcb980a85b76febf401e110ed1b379babed53dbcd3524ad1a723fca8dd8711  alter/models/brainnetcnn.py   (not vendored; kept for a later increment)
```

A verbatim copy of the retrieved upstream tree is kept outside this repository at
`~/temporal-brain-network/experiments/alter_reference/`.

## Mapping

| upstream | here |
|---|---|
| `alter/models/LRBGT/lrbgt.py` :: `add_full_rrwp` + `add_every_rrwp` | `alter_model.py` :: `rrwp_positional_encoding` |
| `alter/models/LRBGT/components/transformer_encoder.py` :: `InterpretableTransformerEncoder` | `alter_model.py` :: `InterpretableTransformerEncoder` |
| `alter/models/LRBGT/lrbgt.py` :: `TransPoolingEncoder` | `alter_model.py` :: `TransPoolingEncoder` |
| `alter/models/LRBGT/lrbgt.py` :: `BrainNetworkTransformer` | `alter_model.py` :: `StaticALTER` |
| `alter/models/LRBGT/ptdec/dec.py` :: `DEC` | `ptdec/dec.py` :: `DEC` (verbatim + one line) |
| `alter/models/LRBGT/ptdec/cluster.py` :: `ClusterAssignment` | `ptdec/cluster.py` :: `ClusterAssignment` (verbatim) |
| `alter/conf/model/lrbgt.yaml` + `alter/conf/dataset/ABIDE.yaml` | `alter_model.py` :: `ALTERConfig` defaults |

## Complete list of changes

Every item below is behaviour-preserving **except D on torch >= 2.0** (see D). `tests/test_alter_reference.py` transcribes the
released classes (with only the two torch-compatibility fixes, D and E, which are required for
the released code to execute at all on torch >= 2.0) and asserts **bit-for-bit identical**
logits against this port — in eval and train mode, for all three positional encodings, and
additionally asserts that both constructions produce an identical `state_dict` from the same
`torch.manual_seed`.

**A. Dense batched RRWP instead of per-subject sparse COO.**
Upstream builds a `torch.sparse_coo_tensor` per subject inside a Python loop and immediately
calls `.to_dense()`. Since `torch.where(data > 0)` yields unique indices, that is exactly
`data * 1[data > 0]`, so the dense form is identical, and batching the matrix powers with
`torch.bmm` is identical too (verified bitwise for B in {1,3,4,8,16}, R in {20,50,100},
walk length in {8,16,32}, including a subject with an isolated node and an all-positive
matrix). Motivation: the dense path is device-agnostic and is safe under
`torch.use_deterministic_algorithms(True, warn_only=False)`, which is this repository's
production setting since `main@e97aabe`.

**B. `.cuda()` removed.** Upstream's `add_every_rrwp` ends with `pe = torch.stack(pe_list, dim=-1).cuda()`,
which hard-fails on a CPU-only host. The encoding now follows the device and dtype of its input.

**C. `sizes` list replaced by `n_layers` + `cluster_num`.**
`lrbgt.yaml` declares `sizes: [360, 100]`, but the constructor's first statement is
`sizes[0] = config.dataset.node_sz`, so the effective schedule on ABIDE is `[100, 100]` and the
`360` never reaches a layer. This port derives the same schedule
(`sizes = [n_rois]*(L-1) + [cluster_num]`, `pooling = [False]*(L-1) + [True]`) rather than
carrying a list whose first entry is discarded. For the default `n_layers=2, cluster_num=100`
the constructed modules are identical to upstream.

**D. `InterpretableTransformerEncoder.__init__` passes `device`/`dtype` by keyword — and
this one is NOT behaviour-preserving on torch >= 2.0.** Upstream forwards them positionally.
`nn.TransformerEncoderLayer` gained a `bias` argument before `device` in torch 2.0, so the
positional call binds `device=None` into `bias`, and every `Linear`/`MultiheadAttention`/
`LayerNorm` inside the layer is built **without bias**. Consequences:

* On the torch 1.x the paper used, the released code builds biased layers (the paper's model).
* On any torch >= 2.0, the *unmodified* released code silently builds bias-free layers
  (1,767,290 instead of 1,771,186 parameters). The collaborator's harness — which added
  `is_causal` to `_sa_block` and therefore runs on torch >= 2.0 — trains that bias-free variant.

Both are reproducible here on purpose: `ALTERConfig.transformer_bias=True` (default) is the
paper's model; `transformer_bias=False` is the released-code-on-modern-torch variant.
`tests/test_alter_harness_variant.py` pins the latter bit-for-bit against a literal transcription
of the collaborator's files.

**E. `_sa_block` accepts `is_causal`.**
torch >= 2.0 calls `self._sa_block(..., is_causal=is_causal)`. The released override does not
accept it, so the released file raises `TypeError` on any modern torch. The argument is
accepted and ignored (ALTER never uses a causal mask).

**F. `nn.KLDivLoss(size_average=False)` → `nn.KLDivLoss(reduction='sum')`.**
`size_average=False` is *defined* as `reduction='sum'`; the keyword has been deprecated since
torch 1.0. The loss is never called (see H).

**G. Head renamed `fc` → `classifier`.**
Required so the module can be reused as the pipeline head by
`unified_bn.pipelines.vanilla.VanillaPipeline`, matching `StaticBQN.classifier`; also avoids
colliding with `fc` = functional connectivity, which is what `fc` means everywhere else in this
repository. State-dict keys change from `fc.N.*` to `classifier.N.*` and nothing else does.

**H. `forward(time_seires, node_feature)` → `forward(corr, ...)`.**
The released signature's first argument is ROI time series, which the ALTER forward never
reads. Dropping it matches `StaticBQN` and keeps the true data flow visible. The
`return_embedding` / `return_aux` keywords are added to match the `StaticBQN` calling
convention that `unified_bn` backbones rely on.

**I. `nhead`, `dim_feedforward`, `dropout` promoted to configuration.**
Hard-coded upstream as `4`, `1024`, and torch's `0.1` default. Their defaults here are those
values, so the default model is unchanged.

**J. Input validation added** (`_validate_corr`), mirroring `StaticBQN._validate_corr`. Rejects
non-`[B,R,R]`, non-floating-point, and non-finite input instead of failing deep inside attention.

**K. `edge_threshold` configuration field, default `0.0`.**
The walk edge rule stays exactly upstream's `data > 0.0` at the default. The field exists
because the collaborator's own harness (`gcn_gps_experiments/main.py`) comments
`threshold = 0.3  # Same as ALTER for adjacency matrix conversion`, which the released ALTER
code does not implement. Making the rule explicit means switching to `0.3` later is a visible
one-line configuration change with a distinct candidate fingerprint, not a silent edit.

## Behaviour worth knowing (present upstream, deliberately preserved)

* **`need_weights=True` changes the attention kernel.** It keeps torch on the math attention
  path; the fused path differs by ~1e-6 in float32. Preserved because it is what the released
  model trains with. Consequence: on modern torch, eval-mode forwards with no mask take a fused
  fast path that bypasses `_sa_block` entirely, so `attention_weights` is only refreshed during
  training-mode forwards.
* **Cluster-center row 0 is not unit-norm.** The released Gram-Schmidt assigns
  `orthogonal_cluster_centers[0] = initial_cluster_centers[0]` without normalising, then
  normalises rows 1..K-1. All rows are mutually orthogonal; only row 0 has an arbitrary norm.
* **The soft assignment divides by `‖C‖₂`, not `‖C‖₂²`.** Sibling re-implementations of the same
  pt-dec code (including the one in this repository's `models/bqn/bqn_model.py::DECReadout` and
  the one in the EdgeQuad `QUETT/model/cluster_pooling.py`) use the square. ALTER does not.
  Combined with the previous point this is numerically observable.
* **The DEC KL auxiliary loss is never used.** ALTER's released training loop optimises
  cross-entropy only. `StaticALTER.dec_kl_loss` is preserved for fidelity and is not called
  anywhere in this repository.
* **Upstream trains with continuous mixup** (`conf/preprocess/mixup.yaml`, applied in
  `alter/training/training.py`). `unified_bn` has no mixup and neither did the Static-BQN and
  TempLasso runs, so ALTER is run here without it for protocol parity. This is a *protocol*
  difference, not a model difference, and is recorded in the PR rather than in this file.
