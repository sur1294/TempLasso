"""
DEC pooling block used by ALTER's TransPoolingEncoder.

PROVENANCE
----------
Vendored from the official ALTER release:

    https://github.com/yushuowiki/ALTER
    commit ad5c5a95d54aa2a0a2a275c008a692d73af293f7
    path   alter/models/LRBGT/ptdec/dec.py

which itself carries the upstream note "From https://github.com/vlukiyanov/pt-dec".

ONLY MODIFICATION
-----------------
``nn.KLDivLoss(size_average=False)`` -> ``nn.KLDivLoss(reduction='sum')``.
``size_average`` has been deprecated since torch 1.0 and ``size_average=False``
is defined to mean ``reduction='sum'``, so this is a rename, not a semantic
change. It is required because newer torch releases warn (and eventually
error) on the legacy keyword.

NOTE ON ``loss``
----------------
``DEC.loss`` implements the DEC KL auxiliary objective. The released ALTER
training loop never adds it -- ``loss = CrossEntropy(predict, label)`` is the
whole objective -- so it is preserved here for architectural fidelity and is
never called by this repository either. See
``tests/test_alter_reference.py::test_dec_kl_loss_is_never_used_by_training``.
"""

import torch
import torch.nn as nn
from typing import Tuple

from models.alter.ptdec.cluster import ClusterAssignment


class DEC(nn.Module):
    def __init__(
        self,
        cluster_number: int,
        hidden_dimension: int,
        encoder: torch.nn.Module,
        alpha: float = 1.0,
        orthogonal=True,
        freeze_center=True, project_assignment=True
    ):
        """
        Module which holds all the moving parts of the DEC algorithm, as described in
        Xie/Girshick/Farhadi; this includes the AutoEncoder stage and the ClusterAssignment stage.

        :param cluster_number: number of clusters
        :param hidden_dimension: hidden dimension, output of the encoder
        :param encoder: encoder to use
        :param alpha: parameter representing the degrees of freedom in the t-distribution, default 1.0
        """
        super(DEC, self).__init__()
        self.encoder = encoder
        self.hidden_dimension = hidden_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha
        self.assignment = ClusterAssignment(
            cluster_number, self.hidden_dimension, alpha, orthogonal=orthogonal, freeze_center=freeze_center, project_assignment=project_assignment
        )

        # Upstream: nn.KLDivLoss(size_average=False). Same semantics.
        self.loss_fn = nn.KLDivLoss(reduction='sum')

    def forward(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the cluster assignment using the ClusterAssignment after running the batch
        through the encoder part of the associated AutoEncoder module.

        :param batch: [batch size, embedding dimension] FloatTensor
        :return: [batch size, number of clusters] FloatTensor
        """
        node_num = batch.size(1)
        batch_size = batch.size(0)

        # [batch size, embedding dimension]
        flattened_batch = batch.view(batch_size, -1)
        encoded = self.encoder(flattened_batch)
        # [batch size * node_num, hidden dimension]
        encoded = encoded.view(batch_size * node_num, -1)
        # [batch size * node_num, cluster_number]
        assignment = self.assignment(encoded)
        # [batch size, node_num, cluster_number]
        assignment = assignment.view(batch_size, node_num, -1)
        # [batch size, node_num, hidden dimension]
        encoded = encoded.view(batch_size, node_num, -1)
        # Multiply the encoded vectors by the cluster assignment to get the final node representations
        # [batch size, cluster_number, hidden dimension]
        node_repr = torch.bmm(assignment.transpose(1, 2), encoded)
        return node_repr, assignment

    def target_distribution(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Compute the target distribution p_ij, given the batch (q_ij), as in 3.1.3 Equation 3 of
        Xie/Girshick/Farhadi; this is used the KL-divergence loss function.

        :param batch: [batch size, number of clusters] Tensor of dtype float
        :return: [batch size, number of clusters] Tensor of dtype float
        """
        weight = (batch ** 2) / torch.sum(batch, 0)
        return (weight.t() / torch.sum(weight, 1)).t()

    def loss(self, assignment):
        flattened_assignment = assignment.view(-1, assignment.size(-1))
        target = self.target_distribution(flattened_assignment).detach()
        return self.loss_fn(flattened_assignment.log(), target) / flattened_assignment.size(0)

    def get_cluster_centers(self) -> torch.Tensor:
        """
        Get the cluster centers, as computed by the encoder.

        :return: [number of clusters, hidden dimension] Tensor of dtype float
        """
        return self.assignment.get_cluster_centers()


__all__ = ["DEC"]
