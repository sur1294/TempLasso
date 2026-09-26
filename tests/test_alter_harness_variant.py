"""
Bit-for-bit equivalence between ``StaticALTER(transformer_bias=False,
nhead=1, edge_threshold=0.3)`` and the ALTER the collaborator actually runs.

The collaborator's ALTER folder is a clone of https://github.com/yushuowiki/ALTER
at commit ``92cb82e`` -- five commits BEHIND the ``ad5c5a9`` this repository
vendors -- restructured into a flat ``models/lrbgt/`` package. Relative to the
vendored release it differs in exactly three functional ways:

  1. ``rrwp.py`` keeps the pre-``ad5c5a9`` walk-edge rule ``corr > 0.3``
     (upstream changed it to ``> 0.0`` in its final commit);
  2. ``TransPoolingEncoder`` hard-codes ``nhead=1`` instead of ``4``;
  3. ``transformer_encoder.py`` forwards ``device, dtype`` positionally to
     ``nn.TransformerEncoderLayer``. On torch >= 2.0 that puts ``device=None``
     in the ``bias`` slot, so every attention/FFN/LayerNorm inside the encoder
     is built WITHOUT bias. Her file also adds ``is_causal`` to ``_sa_block``,
     which only torch >= 2.0 requires -- i.e. that is the torch she runs on,
     so the bias-free layer is the one her numbers come from.

``cluster.py`` / ``dec.py`` are byte-identical to upstream, the head FLOPs
bookkeeping she added lives in ``__init__`` and never touches the forward.

The block marked COLLABORATOR TRANSCRIPTION is her ``rrwp.py``,
``transformer_encoder.py`` and the model classes of ``model.py`` copied
verbatim (FLOPs bookkeeping and the unused ``rrwp_config`` lookup dropped, as
neither affects the computation). The positional ``super().__init__`` call is
kept exactly as she wrote it, so this test only means what it claims on a
torch whose ``TransformerEncoderLayer`` has the ``bias`` parameter; it skips
otherwise rather than pretend.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Optional

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.alter.alter_model import ALTERConfig, StaticALTER
from models.alter.ptdec.dec import DEC  # byte-identical to hers


_HAS_BIAS_SLOT = "bias" in inspect.signature(
    nn.TransformerEncoderLayer.__init__
).parameters

pytestmark = pytest.mark.skipif(
    not _HAS_BIAS_SLOT,
    reason="collaborator's positional device/dtype call only drops the bias "
           "on a torch whose TransformerEncoderLayer has a `bias` parameter",
)


# =====================================================================
# COLLABORATOR TRANSCRIPTION - do not "clean up"
# =====================================================================

def her_add_full_rrwp(data: torch.Tensor, walk_length: int) -> torch.Tensor:
    batch_size, num_nodes, _ = data.shape
    device = data.device
    pes = []
    for batch_idx in range(batch_size):
        adj_matrix = data[batch_idx]
        pe = her_add_every_rrwp(adj_matrix, walk_length)
        pes.append(pe)
    return torch.stack(pes, dim=0)


def her_add_every_rrwp(data: torch.Tensor,
                       walk_length: int = 8,
                       threshold: float = 0.3,
                       add_identity: bool = True,
                       spd: bool = False,
                       **kwargs) -> torch.Tensor:
    device = data.device
    num_nodes = data.shape[0]
    edge_mask = data > threshold
    edge_index = torch.column_stack(torch.where(edge_mask)).T.contiguous()
    if edge_index.size(0) == 0:
        return torch.zeros(num_nodes, walk_length, device=device)
    edge_weights = data[edge_index[0], edge_index[1]]
    adj_sparse = torch.sparse_coo_tensor(
        edge_index, edge_weights, (num_nodes, num_nodes)
    )
    adj = adj_sparse.to_dense()
    degree = adj.sum(dim=1)
    degree_inv = torch.where(degree > 0, 1.0 / degree, 0.0)
    adj_normalized = adj * degree_inv.view(-1, 1)
    pe_list = []
    if add_identity:
        pe_list.append(torch.eye(num_nodes, dtype=torch.float, device=device))
    pe_list.append(adj_normalized)
    current_power = adj_normalized
    for _ in range(2, walk_length):
        current_power = current_power @ adj_normalized
        pe_list.append(current_power)
    pe_tensor = torch.stack(pe_list, dim=-1)
    abs_pe = pe_tensor.diagonal().transpose(0, 1)
    return abs_pe


class HerInterpretableTransformerEncoder(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation=F.relu, layer_norm_eps=1e-5, batch_first=False,
                 norm_first=False, device=None, dtype=None) -> None:
        # Positional, exactly as in her file: on torch >= 2.0 ``device``
        # lands in ``bias``.
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation,
                         layer_norm_eps, batch_first, norm_first, device, dtype)
        self.attention_weights: Optional[Tensor] = None

    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor],
                  key_padding_mask: Optional[Tensor],
                  is_causal: bool = False) -> Tensor:
        x, weights = self.self_attn(x, x, x, attn_mask=attn_mask,
                                    key_padding_mask=key_padding_mask,
                                    need_weights=True)
        self.attention_weights = weights
        return self.dropout1(x)


class HerTransPoolingEncoder(nn.Module):
    def __init__(self, input_feature_size, input_node_num, hidden_size,
                 output_node_num, pooling=True, orthogonal=True,
                 freeze_center=False, project_assignment=True):
        super().__init__()
        self.transformer = HerInterpretableTransformerEncoder(
            d_model=input_feature_size, nhead=1,
            dim_feedforward=hidden_size, batch_first=True)
        self.pooling = pooling
        if pooling:
            encoder_hidden_size = 32
            self.encoder = nn.Sequential(
                nn.Linear(input_feature_size * input_node_num, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size, input_feature_size * input_node_num),
            )
            self.dec = DEC(cluster_number=output_node_num,
                           hidden_dimension=input_feature_size,
                           encoder=self.encoder, orthogonal=orthogonal,
                           freeze_center=freeze_center,
                           project_assignment=project_assignment)

    def forward(self, x):
        x = self.transformer(x)
        if self.pooling:
            x, assignment = self.dec(x)
            return x, assignment
        return x, None


class HerBrainNetworkTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
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
                HerTransPoolingEncoder(
                    input_feature_size=forward_dim,
                    input_node_num=in_sizes[index],
                    hidden_size=1024,
                    output_node_num=size,
                    pooling=do_pooling[index],
                    orthogonal=config.model.orthogonal,
                    freeze_center=config.model.freeze_center,
                    project_assignment=config.model.project_assignment))
        self.dim_reduction = nn.Sequential(nn.Linear(forward_dim, 8), nn.LeakyReLU())
        self.fc = nn.Sequential(
            nn.Linear(8 * sizes[-1], 256), nn.LeakyReLU(),
            nn.Linear(256, 32), nn.LeakyReLU(), nn.Linear(32, 2))

    def forward(self, time_seires, node_feature):
        bz, _, _, = node_feature.shape
        if self.pos_encoding == 'identity':
            pos_emb = self.node_identity.expand(bz, *self.node_identity.shape)
            node_feature = torch.cat([node_feature, pos_emb], dim=-1)
        if self.pos_encoding == 'rrwp':
            # her forward reads config.model.rrwp.threshold and then does not
            # pass it on; add_full_rrwp therefore runs with rrwp.py's default
            # threshold=0.3 and walk_length=pos_embed_dim.
            pos_emb = her_add_full_rrwp(node_feature, self.pos_embed_dim)
            node_feature = torch.cat([node_feature, pos_emb], dim=-1)
        assignments = []
        for atten in self.attention_list:
            node_feature, assignment = atten(node_feature)
            assignments.append(assignment)
        node_feature = self.dim_reduction(node_feature)
        node_feature = node_feature.reshape((bz, -1))
        return self.fc(node_feature)


def _her_config(node_sz: int = 100):
    """Her main.py ModelConfig / DatasetConfig for ABIDE, verbatim values."""
    return SimpleNamespace(
        dataset=SimpleNamespace(node_sz=node_sz),
        model=SimpleNamespace(
            sizes=[360, node_sz],
            pooling=[False, True],
            pos_encoding="rrwp",
            pos_embed_dim=32,
            rrwp=SimpleNamespace(walk_length=8, threshold=0.3, add_identity=True),
            orthogonal=True,
            freeze_center=True,
            project_assignment=True,
        ),
    )


# =====================================================================
# helpers
# =====================================================================

R = 100

HARNESS = ALTERConfig(nhead=1, edge_threshold=0.3, transformer_bias=False)


def _correlation_like(batch_size: int, n_rois: int = R, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(batch_size, n_rois, n_rois, generator=generator)
    x = 0.5 * (x + x.transpose(-1, -2))
    x = torch.tanh(x)
    diagonal = torch.arange(n_rois)
    x[:, diagonal, diagonal] = 1.0
    return x


def _to_port_key(key: str) -> str:
    return key.replace("fc.", "classifier.", 1) if key.startswith("fc.") else key


def _paired(seed: int = 1234):
    torch.manual_seed(seed)
    hers = HerBrainNetworkTransformer(_her_config())
    torch.manual_seed(seed)
    ours = StaticALTER(HARNESS)
    return hers, ours


# =====================================================================
# tests
# =====================================================================

def test_her_layers_really_are_bias_free_on_this_torch() -> None:
    hers, _ = _paired()
    names = [n for n, _ in hers.named_parameters()]
    assert not any(".transformer." in n and n.endswith(".bias") for n in names)
    assert sum(p.numel() for p in hers.parameters()) == 1_767_290


def test_state_dicts_match_from_one_seed_without_weight_copy() -> None:
    hers, ours = _paired(seed=999)
    her_state, our_state = hers.state_dict(), ours.state_dict()
    assert sorted(_to_port_key(k) for k in her_state) == sorted(our_state)
    for key, value in her_state.items():
        assert torch.equal(value, our_state[_to_port_key(key)]), key


def test_rrwp_with_threshold_matches_her_routine_bitwise() -> None:
    from models.alter.alter_model import rrwp_positional_encoding

    for seed in (1, 2, 3):
        fc = _correlation_like(4, R, seed)
        assert torch.equal(
            her_add_full_rrwp(fc, 32),
            rrwp_positional_encoding(fc, walk_length=32, edge_threshold=0.3),
        )
    # and it is genuinely different from the vendored release's rule
    fc = _correlation_like(2, R, 4)
    assert not torch.equal(
        rrwp_positional_encoding(fc, walk_length=32, edge_threshold=0.3),
        rrwp_positional_encoding(fc, walk_length=32, edge_threshold=0.0),
    )


def test_logits_match_hers_bitwise_in_eval_mode() -> None:
    hers, ours = _paired()
    hers.eval()
    ours.eval()
    fc = _correlation_like(5, R, 21)
    with torch.no_grad():
        assert torch.equal(hers(None, fc), ours(fc))


def test_logits_match_hers_bitwise_in_train_mode() -> None:
    hers, ours = _paired()
    hers.train()
    ours.train()
    fc = _correlation_like(5, R, 23)
    torch.manual_seed(77)
    expected = hers(None, fc)
    torch.manual_seed(77)
    observed = ours(fc)
    assert torch.equal(expected, observed)


def test_gradients_match_hers_bitwise() -> None:
    hers, ours = _paired()
    hers.train()
    ours.train()
    fc = _correlation_like(3, R, 29)
    torch.manual_seed(5)
    hers(None, fc).sum().backward()
    torch.manual_seed(5)
    ours(fc).sum().backward()
    her_grads = {n: p.grad for n, p in hers.named_parameters() if p.grad is not None}
    our_grads = {n: p.grad for n, p in ours.named_parameters() if p.grad is not None}
    assert set(_to_port_key(k) for k in her_grads) == set(our_grads)
    for key, grad in her_grads.items():
        assert torch.equal(grad, our_grads[_to_port_key(key)]), key


def test_paper_variant_and_harness_variant_are_distinct_models() -> None:
    torch.manual_seed(3)
    paper = StaticALTER(ALTERConfig())
    torch.manual_seed(3)
    harness = StaticALTER(HARNESS)
    assert paper.config.nhead == 4 and harness.config.nhead == 1
    assert sum(p.numel() for p in paper.parameters()) == 1_771_186
    assert sum(p.numel() for p in harness.parameters()) == 1_767_290
