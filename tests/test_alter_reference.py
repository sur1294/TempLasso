"""
Bit-for-bit equivalence between ``models.alter`` and the released ALTER code.

The block marked UPSTREAM TRANSCRIPTION below is the released implementation
(https://github.com/yushuowiki/ALTER @ ad5c5a95d54aa2a0a2a275c008a692d73af293f7,
files ``alter/models/LRBGT/lrbgt.py``, ``alter/models/LRBGT/ptdec/{dec,cluster}.py``
and ``alter/models/LRBGT/components/transformer_encoder.py``) copied verbatim,
with exactly two mechanical torch-compatibility fixes, each marked ``# FIX``:

  * ``_sa_block`` accepts ``is_causal`` (torch >= 2.0 passes it, so the released
    override raises ``TypeError`` on any modern torch and cannot run at all);
  * ``nn.KLDivLoss(size_average=False)`` -> ``reduction='sum'``, which is the
    definition of the removed keyword;

plus the removal of a hard-coded ``.cuda()`` so the reference can run on a
CPU-only host. Nothing else is touched -- in particular ``need_weights=True``,
the unnormalised cluster-center row 0, and the division by ``||C||`` rather
than ``||C||^2`` are all preserved, because those are exactly the details a
"faithful port" claim has to be tested against.

If these tests fail, the port has drifted from the released model and any
ALTER number produced by this repository stops being attributable to ALTER.
"""

from __future__ import annotations

import ast
import importlib
import inspect
from types import SimpleNamespace
from typing import Optional

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch.nn.functional import softmax

from models.alter.alter_model import (
    ALTERConfig,
    StaticALTER,
    rrwp_positional_encoding,
)


# =====================================================================
# UPSTREAM TRANSCRIPTION - do not "clean up"
# =====================================================================

class UpClusterAssignment(nn.Module):
    def __init__(self, cluster_number, embedding_dimension, alpha=1.0,
                 cluster_centers=None, orthogonal=True, freeze_center=True,
                 project_assignment=True):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha
        self.project_assignment = project_assignment
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(
                self.cluster_number, self.embedding_dimension, dtype=torch.float
            )
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        if orthogonal:
            orthogonal_cluster_centers = torch.zeros(
                cluster_number, embedding_dimension, dtype=torch.float
            )
            orthogonal_cluster_centers[0] = initial_cluster_centers[0]
            for i in range(1, cluster_number):
                project = 0
                for j in range(i):
                    project += self.project(
                        initial_cluster_centers[j], initial_cluster_centers[i]
                    )
                initial_cluster_centers[i] -= project
                orthogonal_cluster_centers[i] = initial_cluster_centers[i] / \
                    torch.norm(initial_cluster_centers[i], p=2)
            initial_cluster_centers = orthogonal_cluster_centers
        self.cluster_centers = Parameter(
            initial_cluster_centers, requires_grad=(not freeze_center)
        )

    @staticmethod
    def project(u, v):
        return (torch.dot(u, v) / torch.dot(u, u)) * u

    def forward(self, batch):
        if self.project_assignment:
            assignment = batch @ self.cluster_centers.T
            assignment = torch.pow(assignment, 2)
            norm = torch.norm(self.cluster_centers, p=2, dim=-1)
            soft_assign = assignment / norm
            return softmax(soft_assign, dim=-1)
        norm_squared = torch.sum(
            (batch.unsqueeze(1) - self.cluster_centers) ** 2, 2
        )
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        numerator = numerator ** (float(self.alpha + 1) / 2)
        return numerator / torch.sum(numerator, dim=1, keepdim=True)


class UpDEC(nn.Module):
    def __init__(self, cluster_number, hidden_dimension, encoder, alpha=1.0,
                 orthogonal=True, freeze_center=True, project_assignment=True):
        super().__init__()
        self.encoder = encoder
        self.hidden_dimension = hidden_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha
        self.assignment = UpClusterAssignment(
            cluster_number, hidden_dimension, alpha, orthogonal=orthogonal,
            freeze_center=freeze_center, project_assignment=project_assignment,
        )
        self.loss_fn = nn.KLDivLoss(reduction='sum')  # FIX: size_average=False

    def forward(self, batch):
        node_num = batch.size(1)
        batch_size = batch.size(0)
        flattened_batch = batch.view(batch_size, -1)
        encoded = self.encoder(flattened_batch)
        encoded = encoded.view(batch_size * node_num, -1)
        assignment = self.assignment(encoded)
        assignment = assignment.view(batch_size, node_num, -1)
        encoded = encoded.view(batch_size, node_num, -1)
        node_repr = torch.bmm(assignment.transpose(1, 2), encoded)
        return node_repr, assignment


class UpInterpretableTransformerEncoder(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation=F.relu, layer_norm_eps=1e-5, batch_first=False,
                 norm_first=False, device=None, dtype=None) -> None:
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation,
                         layer_norm_eps, batch_first, norm_first,
                         device=device, dtype=dtype)
        self.attention_weights: Optional[Tensor] = None

    def _sa_block(self, x, attn_mask, key_padding_mask,
                  is_causal=False):  # FIX: is_causal added
        x, weights = self.self_attn(x, x, x, attn_mask=attn_mask,
                                    key_padding_mask=key_padding_mask,
                                    need_weights=True)
        self.attention_weights = weights
        return self.dropout1(x)


def up_add_full_rrwp(data, walk_length):
    pes = []
    for ids in range(data.shape[0]):
        dt = data[ids].squeeze()
        pe = up_add_every_rrwp(dt, walk_length)
        pes.append(pe)
    return torch.stack(pes)


def up_add_every_rrwp(data, walk_length=8, add_identity=True, spd=False, **kwargs):
    edge_index = torch.column_stack(torch.where(data > 0.0)).T.contiguous()
    device = edge_index.device
    num_nodes = data.shape[0]
    edge_weight = data[edge_index[0], edge_index[1]]
    adj = torch.sparse_coo_tensor(
        edge_index, edge_weight, (num_nodes, num_nodes)
    )
    adj = adj.to_dense()
    deg = adj.sum(dim=1)
    deg_inv = 1.0 / adj.sum(dim=1)
    deg_inv[deg_inv == float('inf')] = 0
    adj = adj * deg_inv.view(-1, 1)
    adj = adj.to_dense()

    pe_list = []
    i = 0
    if add_identity:
        pe_list.append(
            torch.eye(num_nodes, dtype=torch.float, device=device)
        )
        i = i + 1
    out = adj
    pe_list.append(adj)
    if walk_length > 2:
        for j in range(i + 1, walk_length):
            out = out @ adj
            pe_list.append(out)
    pe = torch.stack(pe_list, dim=-1)  # FIX: trailing .cuda() removed
    abs_pe = pe.diagonal().transpose(0, 1)
    return abs_pe


class UpTransPoolingEncoder(nn.Module):
    def __init__(self, input_feature_size, input_node_num, hidden_size,
                 output_node_num, pooling=True, orthogonal=True,
                 freeze_center=False, project_assignment=True):
        super().__init__()
        self.transformer = UpInterpretableTransformerEncoder(
            d_model=input_feature_size, nhead=4,
            dim_feedforward=hidden_size, batch_first=True,
        )
        self.pooling = pooling
        if pooling:
            encoder_hidden_size = 32
            self.encoder = nn.Sequential(
                nn.Linear(input_feature_size * input_node_num,
                          encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size,
                          input_feature_size * input_node_num),
            )
            self.dec = UpDEC(
                cluster_number=output_node_num,
                hidden_dimension=input_feature_size, encoder=self.encoder,
                orthogonal=orthogonal, freeze_center=freeze_center,
                project_assignment=project_assignment,
            )

    def is_pooling_enabled(self):
        return self.pooling

    def forward(self, x):
        x = self.transformer(x)
        if self.pooling:
            x, assignment = self.dec(x)
            return x, assignment
        return x, None


class UpBrainNetworkTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_list = nn.ModuleList()
        forward_dim = config.dataset.node_sz
        self.pos_encoding = config.model.pos_encoding
        self.pos_embed_dim = config.model.pos_embed_dim
        if self.pos_encoding == 'identity':
            self.node_identity = nn.Parameter(torch.zeros(
                config.dataset.node_sz, config.model.pos_embed_dim),
                requires_grad=True)
            forward_dim = config.dataset.node_sz + config.model.pos_embed_dim
            nn.init.kaiming_normal_(self.node_identity)
        if self.pos_encoding == 'rrwp':
            forward_dim = config.dataset.node_sz + config.model.pos_embed_dim

        sizes = config.model.sizes
        sizes[0] = config.dataset.node_sz
        in_sizes = [config.dataset.node_sz] + sizes[:-1]
        do_pooling = config.model.pooling
        self.do_pooling = do_pooling
        for index, size in enumerate(sizes):
            self.attention_list.append(
                UpTransPoolingEncoder(
                    input_feature_size=forward_dim,
                    input_node_num=in_sizes[index],
                    hidden_size=1024,
                    output_node_num=size,
                    pooling=do_pooling[index],
                    orthogonal=config.model.orthogonal,
                    freeze_center=config.model.freeze_center,
                    project_assignment=config.model.project_assignment))

        self.dim_reduction = nn.Sequential(
            nn.Linear(forward_dim, 8), nn.LeakyReLU())
        self.fc = nn.Sequential(
            nn.Linear(8 * sizes[-1], 256), nn.LeakyReLU(),
            nn.Linear(256, 32), nn.LeakyReLU(), nn.Linear(32, 2))

    def forward(self, time_seires, node_feature):
        bz, _, _, = node_feature.shape
        if self.pos_encoding == 'identity':
            pos_emb = self.node_identity.expand(bz, *self.node_identity.shape)
            node_feature = torch.cat([node_feature, pos_emb], dim=-1)
        if self.pos_encoding == 'rrwp':
            pos_emb = up_add_full_rrwp(node_feature, self.pos_embed_dim)
            node_feature = torch.cat([node_feature, pos_emb], dim=-1)
        assignments = []
        for atten in self.attention_list:
            node_feature, assignment = atten(node_feature)
            assignments.append(assignment)
        node_feature = self.dim_reduction(node_feature)
        node_feature = node_feature.reshape((bz, -1))
        return self.fc(node_feature)


def _upstream_config(node_sz: int = 100, pos_encoding: str = "rrwp"):
    """The released lrbgt.yaml + ABIDE.yaml, as a plain attribute holder."""
    return SimpleNamespace(
        dataset=SimpleNamespace(node_sz=node_sz),
        model=SimpleNamespace(
            # sizes[0] is overwritten by the constructor; 360 never survives.
            sizes=[360, node_sz],
            pooling=[False, True],
            pos_encoding=pos_encoding,
            pos_embed_dim=32,
            orthogonal=True,
            freeze_center=True,
            project_assignment=True,
        ),
    )


# =====================================================================
# helpers
# =====================================================================

R = 100
POS_ENCODINGS = ("rrwp", "identity", "none")


def _correlation_like(batch_size: int, n_rois: int = R, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(batch_size, n_rois, n_rois, generator=generator)
    x = 0.5 * (x + x.transpose(-1, -2))
    x = torch.tanh(x)
    diagonal = torch.arange(n_rois)
    x[:, diagonal, diagonal] = 1.0
    return x


def _upstream_key_to_port(key: str) -> str:
    # Only deviation G: the head is called ``fc`` upstream.
    return key.replace("fc.", "classifier.", 1) if key.startswith("fc.") else key


def _paired_models(pos_encoding: str, seed: int = 1234):
    torch.manual_seed(seed)
    upstream = UpBrainNetworkTransformer(_upstream_config(R, pos_encoding))
    torch.manual_seed(seed)
    port = StaticALTER(ALTERConfig(pos_encoding=pos_encoding))
    return upstream, port


def _copy_upstream_weights(upstream: nn.Module, port: nn.Module) -> None:
    remapped = {
        _upstream_key_to_port(key): value
        for key, value in upstream.state_dict().items()
    }
    missing, unexpected = port.load_state_dict(remapped, strict=True)
    assert not missing and not unexpected, (missing, unexpected)


# =====================================================================
# RRWP parity
# =====================================================================

@pytest.mark.parametrize(
    "batch_size,n_rois,walk_length,seed",
    [
        (4, 100, 32, 1),
        (1, 100, 32, 2),
        (16, 100, 32, 3),
        (3, 50, 8, 4),
        (8, 20, 32, 5),
    ],
)
def test_dense_batched_rrwp_is_bitwise_upstream(
    batch_size, n_rois, walk_length, seed
) -> None:
    fc = _correlation_like(batch_size, n_rois, seed)
    expected = up_add_full_rrwp(fc, walk_length)
    observed = rrwp_positional_encoding(fc, walk_length=walk_length)
    assert torch.equal(expected, observed)


def test_rrwp_matches_upstream_for_a_node_with_no_positive_edges() -> None:
    # Upstream maps 1/0 -> inf -> 0, leaving that row of the transition matrix
    # at zero. The dense port has to reproduce that, not produce NaN.
    fc = _correlation_like(2, 20, 9)
    fc[0, 3, :] = -0.5
    expected = up_add_full_rrwp(fc, 8)
    observed = rrwp_positional_encoding(fc, walk_length=8)
    assert torch.equal(expected, observed)
    assert torch.isfinite(observed).all()


def test_rrwp_matches_upstream_for_an_all_positive_matrix() -> None:
    fc = _correlation_like(2, 30, 11).abs()
    expected = up_add_full_rrwp(fc, 16)
    observed = rrwp_positional_encoding(fc, walk_length=16)
    assert torch.equal(expected, observed)


def test_rrwp_shape_and_identity_column() -> None:
    fc = _correlation_like(3, R, 13)
    pe = rrwp_positional_encoding(fc, walk_length=32)
    assert pe.shape == (3, R, 32)
    # Column 0 is diag(P^0) = diag(I).
    assert torch.equal(pe[:, :, 0], torch.ones(3, R))
    # Return probabilities of a row-stochastic walk.
    assert float(pe.min()) >= 0.0
    assert float(pe.max()) <= 1.0


def test_rrwp_edge_threshold_default_is_the_released_rule() -> None:
    fc = _correlation_like(2, 20, 17)
    assert torch.equal(
        rrwp_positional_encoding(fc, walk_length=8, edge_threshold=0.0),
        up_add_full_rrwp(fc, 8),
    )
    # A higher threshold must actually change the walk, otherwise the knob
    # would silently do nothing.
    thresholded = rrwp_positional_encoding(fc, walk_length=8, edge_threshold=0.3)
    assert not torch.equal(thresholded, up_add_full_rrwp(fc, 8))


def test_rrwp_rejects_bad_arguments() -> None:
    fc = _correlation_like(2, 10, 19)
    with pytest.raises(ValueError):
        rrwp_positional_encoding(fc, walk_length=1)
    with pytest.raises(ValueError):
        rrwp_positional_encoding(torch.randn(2, 10, 11), walk_length=4)
    with pytest.raises(TypeError):
        rrwp_positional_encoding([[0.0]], walk_length=4)


# =====================================================================
# whole-model parity
# =====================================================================

@pytest.mark.parametrize("pos_encoding", POS_ENCODINGS)
def test_port_matches_upstream_logits_in_eval_mode(pos_encoding) -> None:
    upstream, port = _paired_models(pos_encoding)
    _copy_upstream_weights(upstream, port)
    upstream.eval()
    port.eval()

    fc = _correlation_like(5, R, 21)
    with torch.no_grad():
        expected = upstream(None, fc)
        observed = port(fc)

    assert torch.equal(expected, observed)


@pytest.mark.parametrize("pos_encoding", POS_ENCODINGS)
def test_port_matches_upstream_logits_in_train_mode(pos_encoding) -> None:
    # Train mode matters on its own: it is the only mode in which torch runs
    # the ``_sa_block`` override, i.e. the ``need_weights=True`` math attention
    # path that the released model actually trains with.
    upstream, port = _paired_models(pos_encoding)
    _copy_upstream_weights(upstream, port)
    upstream.train()
    port.train()

    fc = _correlation_like(5, R, 23)
    torch.manual_seed(77)
    expected = upstream(None, fc)
    torch.manual_seed(77)
    observed = port(fc)

    assert torch.equal(expected, observed)


def test_port_and_upstream_initialise_identically_from_one_seed() -> None:
    # No weight copying here: this pins the *order* in which parameters are
    # drawn, so a future refactor that reorders module construction is caught
    # even though it would leave the architecture nominally unchanged.
    upstream, port = _paired_models("rrwp", seed=999)
    upstream_state = upstream.state_dict()
    port_state = port.state_dict()

    assert sorted(
        _upstream_key_to_port(key) for key in upstream_state
    ) == sorted(port_state)
    for key, value in upstream_state.items():
        assert torch.equal(value, port_state[_upstream_key_to_port(key)])


def test_only_the_head_is_renamed() -> None:
    upstream, port = _paired_models("rrwp")
    renamed = {
        key
        for key in upstream.state_dict()
        if _upstream_key_to_port(key) != key
    }
    assert all(key.startswith("fc.") for key in renamed)
    assert renamed, "expected the head keys to be renamed"


# =====================================================================
# architecture invariants
# =====================================================================

def test_default_configuration_is_the_released_abide_one() -> None:
    config = ALTERConfig()
    assert config.n_rois == 100
    assert config.n_layers == 2
    assert config.pos_encoding == "rrwp"
    assert config.pos_embed_dim == 32
    assert config.nhead == 4
    assert config.ffn_dim == 1024
    assert config.dropout == pytest.approx(0.1)
    assert config.cluster_num == 100
    assert config.edge_threshold == pytest.approx(0.0)
    assert config.d_model == 132
    assert config.embedding_dim == 800


def test_effective_layer_schedule_pools_only_in_the_last_block() -> None:
    model = StaticALTER(ALTERConfig(n_layers=3, cluster_num=7))
    pooling = [block.is_pooling_enabled() for block in model.attention_list]
    assert pooling == [False, False, True]
    assert model.embedding_dim == 8 * 7


def test_embedding_and_logit_shapes() -> None:
    torch.manual_seed(3)
    model = StaticALTER(ALTERConfig()).eval()
    fc = _correlation_like(4, R, 29)
    with torch.no_grad():
        embedding = model(fc, return_embedding=True)
        logits = model(fc)
        logits_aux, aux = model(fc, return_aux=True)

    assert embedding.shape == (4, 800)
    assert logits.shape == (4, 2)
    assert torch.equal(logits, logits_aux)
    assert aux["assignment"].shape == (4, R, R)
    assert aux["pos_enc"].shape == (4, R, 32)
    assert torch.equal(aux["embedding"], embedding)


def test_cluster_assignment_rows_are_a_distribution() -> None:
    torch.manual_seed(5)
    model = StaticALTER(ALTERConfig()).eval()
    with torch.no_grad():
        _, aux = model(_correlation_like(3, R, 31), return_aux=True)
    assignment = aux["assignment"]
    assert torch.allclose(
        assignment.sum(dim=-1), torch.ones(3, R), atol=1e-5
    )
    assert float(assignment.min()) >= 0.0


def test_cluster_centers_are_frozen_orthogonal_and_row_zero_is_unnormalised() -> None:
    # The last clause is not a nitpick: the released Gram-Schmidt writes row 0
    # through without normalising, and the soft assignment then divides by
    # ||C||, so "helpfully" normalising row 0 would silently change outputs.
    torch.manual_seed(7)
    model = StaticALTER(ALTERConfig())
    centers = model.get_cluster_centers()

    assert centers.requires_grad is False
    assert centers.shape == (100, 132)

    norms = centers.norm(dim=-1)
    assert torch.allclose(norms[1:], torch.ones(99), atol=1e-4)
    assert abs(float(norms[0]) - 1.0) > 1e-3

    gram = centers @ centers.T
    off_diagonal = gram - torch.diag(torch.diag(gram))
    assert float(off_diagonal.abs().max()) < 1e-4


def test_every_trainable_parameter_receives_a_gradient() -> None:
    torch.manual_seed(11)
    model = StaticALTER(ALTERConfig(n_rois=20, pos_embed_dim=4, cluster_num=3,
                                    ffn_dim=16, n_layers=2))
    model.train()
    model(_correlation_like(4, 20, 37)).sum().backward()

    without_grad = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert without_grad == []

    frozen = [
        name
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    ]
    # Exactly the DEC centers, which the released config freezes.
    assert frozen == ["attention_list.1.dec.assignment.cluster_centers"]


def test_attention_weights_are_captured_during_training_forwards() -> None:
    torch.manual_seed(13)
    model = StaticALTER(ALTERConfig(n_rois=20, pos_embed_dim=4, cluster_num=3,
                                    ffn_dim=16))
    model.train()
    model(_correlation_like(2, 20, 41))
    weights = model.get_attention_weights()
    assert len(weights) == 2
    assert all(w is not None and w.shape == (2, 20, 20) for w in weights)


def test_construction_and_forward_are_deterministic() -> None:
    fc = _correlation_like(3, 20, 43)
    config = ALTERConfig(n_rois=20, pos_embed_dim=4, cluster_num=3, ffn_dim=16)

    torch.manual_seed(101)
    first = StaticALTER(config).eval()
    torch.manual_seed(101)
    second = StaticALTER(config).eval()

    for key, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[key])
    with torch.no_grad():
        assert torch.equal(first(fc), second(fc))


def test_input_validation_rejects_malformed_fc() -> None:
    model = StaticALTER(ALTERConfig(n_rois=20, pos_embed_dim=4, cluster_num=3,
                                    ffn_dim=16))
    with pytest.raises(ValueError):
        model(torch.zeros(2, 20))
    with pytest.raises(ValueError):
        model(torch.zeros(2, 19, 19))
    with pytest.raises(TypeError):
        model(torch.zeros(2, 20, 20, dtype=torch.int64))
    with pytest.raises(ValueError):
        model(torch.full((2, 20, 20), float("nan")))
    with pytest.raises(TypeError):
        model([[0.0]])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_rois": 0},
        {"n_rois": 1},
        {"n_layers": 0},
        {"nhead": 7},                      # 132 % 7 != 0
        {"pos_encoding": "spd"},
        {"pos_embed_dim": 1},              # rrwp needs I and P
        {"dropout": 1.0},
        {"dropout": -0.1},
        {"edge_threshold": -0.1},
        {"cluster_num": 0},
        {"ffn_dim": 0},
    ],
)
def test_configuration_fails_closed(kwargs) -> None:
    with pytest.raises((TypeError, ValueError)):
        ALTERConfig(**kwargs)


def test_parameter_summary_accounts_for_every_parameter() -> None:
    torch.manual_seed(17)
    model = StaticALTER(ALTERConfig())
    summary = model.parameter_summary()

    assert summary["total"] == sum(p.numel() for p in model.parameters())
    assert (
        summary["attention_blocks"]
        + summary["dim_reduction"]
        + summary["classifier"]
        == summary["total"]
    )
    # Frozen centers are the only non-trainable parameters.
    assert summary["total"] - summary["trainable"] == 100 * 132


# =====================================================================
# vendoring hygiene
# =====================================================================

def _executable_source(module) -> str:
    """Module source with comments and docstrings removed."""
    tree = ast.parse(inspect.getsource(module))
    for node in ast.walk(tree):
        if isinstance(
            node,
            (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            body = getattr(node, "body", None)
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                body.pop(0)
    return ast.unparse(tree)


@pytest.mark.parametrize(
    "module_name",
    [
        "models.alter.alter_model",
        "models.alter.ptdec.dec",
        "models.alter.ptdec.cluster",
    ],
)
def test_vendored_modules_are_device_agnostic(module_name) -> None:
    # The released add_every_rrwp ends in .cuda(); a reintroduced hard-coded
    # device would break every CPU test and every non-CUDA cluster node.
    module = importlib.import_module(module_name)
    assert "cuda" not in _executable_source(module)


def test_dec_kl_loss_exists_but_is_not_used_by_this_repository() -> None:
    import subprocess
    from pathlib import Path

    torch.manual_seed(19)
    model = StaticALTER(ALTERConfig(n_rois=20, pos_embed_dim=4, cluster_num=3,
                                    ffn_dim=16))
    model.train()
    _, aux = model(_correlation_like(2, 20, 47), return_aux=True)
    loss = model.dec_kl_loss(aux["assignments"])
    assert loss is not None and torch.isfinite(loss)

    # ...and nothing in the training stack calls it.
    repo = Path(__file__).resolve().parents[1]
    hits = subprocess.run(
        ["git", "grep", "-n", "dec_kl_loss", "--", "unified_bn"],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    assert hits.stdout.strip() == "", hits.stdout
