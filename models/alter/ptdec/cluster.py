"""
Cluster assignment for the DEC readout used by ALTER.

PROVENANCE
----------
Vendored from the official ALTER release:

    https://github.com/yushuowiki/ALTER
    commit ad5c5a95d54aa2a0a2a275c008a692d73af293f7
    path   alter/models/LRBGT/ptdec/cluster.py

which itself carries the upstream note "From https://github.com/vlukiyanov/pt-dec".

The class body below is a verbatim copy of the released code. Nothing about
the cluster-center initialisation or the soft-assignment mathematics has been
changed, because the whole point of vendoring is that ALTER numbers produced
in this repository are attributable to the released ALTER implementation.

Two properties of the released code are load-bearing and are therefore
recorded here rather than "fixed":

1.  With ``orthogonal=True`` the Gram-Schmidt pass writes row 0 through
    UNNORMALISED (``orthogonal_cluster_centers[0] = initial_cluster_centers[0]``)
    while rows 1..K-1 are normalised to unit length. The rows are mutually
    orthogonal either way, but ``||C_0|| != 1`` in general.

2.  ``forward`` with ``project_assignment=True`` divides the squared
    similarity by ``||C||_2`` (the first power of the norm), not by
    ``||C||_2^2``. Sibling re-implementations of this same pt-dec code in
    other brain-network projects divide by the square; ALTER does not, and
    combined with (1) that is numerically observable.
"""

import torch
import torch.nn as nn
from torch.nn import Parameter
from typing import Optional
from torch.nn.functional import softmax


class ClusterAssignment(nn.Module):
    def __init__(
        self,
        cluster_number: int,
        embedding_dimension: int,
        alpha: float = 1.0,
        cluster_centers: Optional[torch.Tensor] = None,
        orthogonal=True,
        freeze_center=True,
        project_assignment=True
    ) -> None:
        """
        Module to handle the soft assignment, for a description see in 3.1.1. in Xie/Girshick/Farhadi,
        where the Student's t-distribution is used measure similarity between feature vector and each
        cluster centroid.

        :param cluster_number: number of clusters
        :param embedding_dimension: embedding dimension of feature vectors
        :param alpha: parameter representing the degrees of freedom in the t-distribution, default 1.0
        :param cluster_centers: clusters centers to initialise, if None then use Xavier uniform
        """
        super(ClusterAssignment, self).__init__()
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
                self.cluster_number, self.embedding_dimension, dtype=torch.float
            )
            orthogonal_cluster_centers[0] = initial_cluster_centers[0]
            for i in range(1, cluster_number):
                project = 0
                for j in range(i):
                    project += self.project(
                        initial_cluster_centers[j], initial_cluster_centers[i])
                initial_cluster_centers[i] -= project
                orthogonal_cluster_centers[i] = initial_cluster_centers[i] / \
                    torch.norm(initial_cluster_centers[i], p=2)

            initial_cluster_centers = orthogonal_cluster_centers

        self.cluster_centers = Parameter(
            initial_cluster_centers, requires_grad=(not freeze_center))

    @staticmethod
    def project(u, v):
        return (torch.dot(u, v)/torch.dot(u, u))*u

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Compute the soft assignment for a batch of feature vectors, returning a batch of assignments
        for each cluster.

        :param batch: FloatTensor of [batch size, embedding dimension]
        :return: FloatTensor [batch size, number of clusters]
        """

        if self.project_assignment:

            assignment = batch@self.cluster_centers.T
            # prove
            assignment = torch.pow(assignment, 2)

            norm = torch.norm(self.cluster_centers, p=2, dim=-1)
            soft_assign = assignment/norm
            return softmax(soft_assign, dim=-1)

        else:

            norm_squared = torch.sum(
                (batch.unsqueeze(1) - self.cluster_centers) ** 2, 2)
            numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
            power = float(self.alpha + 1) / 2
            numerator = numerator ** power
            return numerator / torch.sum(numerator, dim=1, keepdim=True)

    def get_cluster_centers(self) -> torch.Tensor:
        """
        Get the cluster centers.

        :return: FloatTensor [number of clusters, embedding dimension]
        """
        return self.cluster_centers


__all__ = ["ClusterAssignment"]
