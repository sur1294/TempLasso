"""
K-class extension (PPMI: Control / PD / SWEDD / Prodromal, macro one-vs-rest AUC).

Pins: (1) K=2 is byte-for-byte the historical protocol -- identical fingerprints,
identical metrics; (2) K>2 widens every head, switches the accumulator to macro
OvR AUC / macro recall / macro specificity, lets the L1 selector take the union
over one-vs-rest rows, and makes the data layer accept labels 0..K-1 only when
asked to.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score

from unified_bn.data.common import normalize_binary_labels, normalize_labels
from unified_bn.selectors.lasso import LassoEdgeSelector, LassoEdgeSelectorConfig
from unified_bn.temporal.windowing import WindowPolicyConfig
from unified_bn.training.config import RunIdentity
from unified_bn.training.metrics import (
    BinaryMetricAccumulator, BinaryMetricError, UndefinedBinaryMetricError,
)
from unified_bn.training.model_factory import build_model_from_candidate
from unified_bn.training.search_space import (
    ALTERSearchSpace, BQNSearchSpace, BiGRUSearchSpace, BrainNetCNNSearchSpace,
    candidate_fingerprint, generate_temporal_candidates, generate_vanilla_candidates,
)

R = 12
SPACES = {
    "bqn": BQNSearchSpace(n_layers=(1,), activations=("elu",), dropouts=(0.1,), n_rois=R),
    "alter": ALTERSearchSpace(n_layers=(1,), n_rois=R, pos_embed_dim=4, nhead=4, ffn_dim=16, cluster_num=3),
    "brainnetcnn": BrainNetCNNSearchSpace(dropouts=(0.3,), n_rois=R),
}


def _fc(batch: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.tanh(torch.randn(batch, R, R, generator=g)); x = 0.5 * (x + x.transpose(-1, -2))
    x[:, torch.arange(R), torch.arange(R)] = 1.0
    return x


# ---------------------------------------------------------------- identity
def test_num_classes_two_is_not_serialised_so_fingerprints_are_unchanged() -> None:
    a = generate_vanilla_candidates(seed=42, split_identity="sha256:s", backbone_space=SPACES["bqn"])[0]
    b = generate_vanilla_candidates(seed=42, split_identity="sha256:s", backbone_space=SPACES["bqn"], num_classes=2)[0]
    assert candidate_fingerprint(a) == candidate_fingerprint(b)
    assert "num_classes" not in a.identity.to_dict()
    c = generate_vanilla_candidates(seed=42, split_identity="sha256:s", backbone_space=SPACES["bqn"], num_classes=4)[0]
    assert c.identity.to_dict()["num_classes"] == 4
    assert candidate_fingerprint(c) != candidate_fingerprint(a)


def test_identity_rejects_bad_num_classes() -> None:
    with pytest.raises(ValueError):
        RunIdentity(mode="vanilla", seed=1, split_identity="s", backbone_config={"name": "bqn"}, num_classes=1)
    with pytest.raises(TypeError):
        RunIdentity(mode="vanilla", seed=1, split_identity="s", backbone_config={"name": "bqn"}, num_classes=4.0)


# ---------------------------------------------------------------- heads
@pytest.mark.parametrize("name", list(SPACES))
def test_vanilla_head_width_follows_num_classes(name: str) -> None:
    torch.manual_seed(0)
    cand = generate_vanilla_candidates(seed=42, split_identity="sha256:s", backbone_space=SPACES[name], num_classes=4)[0]
    model = build_model_from_candidate(cand).eval()
    with torch.no_grad():
        logits = model(_fc(3, 1)).logits
    assert logits.shape == (3, 4)
    # The vanilla path still reuses the reference head (no extra parameters).
    assert model.classifier is model.backbone.reference_model.classifier
    two = build_model_from_candidate(generate_vanilla_candidates(
        seed=42, split_identity="sha256:s", backbone_space=SPACES[name])[0]).eval()
    with torch.no_grad():
        assert two(_fc(3, 1)).logits.shape == (3, 2)


def test_temporal_head_width_follows_num_classes() -> None:
    torch.manual_seed(0)
    cand = generate_temporal_candidates(
        seed=42, split_identity="sha256:s", window_policy=WindowPolicyConfig(policy="native"),
        temporal_space=BiGRUSearchSpace(hidden_dims=(8,), num_layers=(1,), poolings=("final",)),
        backbone_space=SPACES["bqn"], num_classes=4,
    )[0]
    model = build_model_from_candidate(cand)
    assert model.num_classes == 4 and model.classifier.out_features == 4


# ---------------------------------------------------------------- metrics
def test_binary_metrics_are_unchanged_for_two_classes() -> None:
    acc = BinaryMetricAccumulator()
    logits = torch.tensor([[2.0, -1.0], [0.3, 0.1], [-1.0, 1.5], [0.5, 0.2]])
    labels = torch.tensor([0, 1, 1, 0])
    acc.update(logits=logits, labels=labels, loss_sum=2.0)
    m = acc.compute()
    p = torch.softmax(logits, 1)[:, 1].numpy()
    assert m.roc_auc == pytest.approx(roc_auc_score(labels.numpy(), p))
    assert m.sensitivity == 0.5 and m.specificity == 1.0 and m.accuracy == 0.75 and m.loss == 0.5


def test_macro_metrics_for_four_classes() -> None:
    g = torch.Generator().manual_seed(3)
    logits = torch.randn(40, 4, generator=g)
    labels = torch.arange(40) % 4
    acc = BinaryMetricAccumulator(); acc.update(logits=logits, labels=labels, loss_sum=40.0)
    m = acc.compute()
    p = torch.softmax(logits, 1).numpy()
    assert m.roc_auc == pytest.approx(roc_auc_score(labels.numpy(), p, multi_class="ovr", average="macro"))
    pred = logits.argmax(1)
    recalls = [((pred == c) & (labels == c)).sum().item() / (labels == c).sum().item() for c in range(4)]
    assert m.sensitivity == pytest.approx(float(np.mean(recalls)))
    assert m.accuracy == pytest.approx((pred == labels).float().mean().item())
    assert 0.0 <= m.specificity <= 1.0 and m.n_samples == 40


def test_macro_metrics_require_every_class_and_reject_out_of_range_labels() -> None:
    acc = BinaryMetricAccumulator()
    acc.update(logits=torch.zeros(6, 4), labels=torch.tensor([0, 1, 2, 0, 1, 2]), loss_sum=1.0)
    with pytest.raises(UndefinedBinaryMetricError, match="every class"):
        acc.compute()
    with pytest.raises(BinaryMetricError, match="0..3"):
        BinaryMetricAccumulator().update(logits=torch.zeros(2, 4), labels=torch.tensor([0, 4]), loss_sum=1.0)


# ---------------------------------------------------------------- selector
def test_lasso_selector_takes_the_union_over_one_vs_rest_rows() -> None:
    rng = np.random.default_rng(0)
    n, e = 120, 60
    x = rng.normal(size=(n, e))
    y = np.repeat([0, 1, 2], n // 3)
    # class 1 lives on edge 3, class 2 on edge 7: both must be kept.
    x[y == 1, 3] += 3.0
    x[y == 2, 7] += 3.0
    sel = LassoEdgeSelector(LassoEdgeSelectorConfig(C=0.05, n_rois=None, min_edges=1, max_iter=2000, random_state=0)) \
        if "n_rois" in LassoEdgeSelectorConfig.__dataclass_fields__ else \
        LassoEdgeSelector(LassoEdgeSelectorConfig(C=0.05, min_edges=1, max_iter=2000, random_state=0))
    sel.fit(torch.tensor(x), torch.tensor(y))
    kept = set(sel.selected_indices.tolist())
    assert {3, 7} <= kept


# ---------------------------------------------------------------- data layer
def test_label_normalisation_is_binary_unless_told_otherwise() -> None:
    assert normalize_binary_labels(np.array([0, 1, 1])).tolist() == [0, 1, 1]
    with pytest.raises(ValueError, match="binary"):
        normalize_binary_labels(np.array([0, 1, 3]))
    assert normalize_labels(np.array([0, 1, 3, 2]), num_classes=4).tolist() == [0, 1, 3, 2]
    with pytest.raises(ValueError, match="0..3"):
        normalize_labels(np.array([0, 4]), num_classes=4)
    with pytest.raises(ValueError):
        normalize_labels(np.array([0, 1]), num_classes=1)
