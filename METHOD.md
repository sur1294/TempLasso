# GCN baseline for FC graph classification

This is a pure-PyTorch implementation of the normalized GCN operator of
[Kipf and Welling (ICLR 2017)](https://arxiv.org/abs/1609.02907), adapted to
subject-level brain graph classification. It is not a reproduction of their
node-classification experiments. No PyTorch Geometric installation is needed.

## Graph construction

Input `R` is a symmetric correlation matrix with unit diagonal, in canonical
ROI order. Fisher-z values must be reconstructed into correlations by the
existing temporal loader before they reach this model.

- Node features: `H_0 = R`. Each ROI's signed connectivity profile is a feature
  vector. Negative correlations are retained here.
- Weighted adjacency: positive off-diagonal correlations, with exactly one
  unit self-loop at each node. Negative correlations are excluded from message
  passing weights. This positive-edge convention is a baseline design choice,
  not a claim that negative connectivity is biologically irrelevant.
- Normalization is performed separately on every subject/window:

```text
A_tilde[i,j] = max(R[i,j], 0) if i != j, otherwise 1
D[i,i]      = sum_j A_tilde[i,j]
A_hat       = D^(-1/2) A_tilde D^(-1/2)
H_(l+1)     = dropout(ReLU(A_hat H_l W_l + b_l))
embedding   = mean_ROI(H_L)
logits      = linear(embedding)
```

Bias is added after aggregation, matching the standard
[GCNConv operator](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GCNConv.html).
No second identity is added to an already self-looped adjacency. Unit loops
keep isolated nodes well-defined. The input FC tensor is not modified.

Default: two layers, width 64, dropout 0.1, spatial mean readout. There is no
adjacency threshold/density search, fitted cohort graph, global normalization,
graph cache, demographic feature, or site feature. With connectivity profiles
as node features, the same canonical ROI ordering is required for every record.

## Temporal and fusion use

The mean readout above averages **ROIs within one window**, not windows over
time. TempX independently builds the graph for each valid FC window, shares
GCN weights across windows, and sends the ordered embedding sequence to the
existing BiGRU. Padding is excluded before encoding. No sequence averaging or
shuffling is introduced.

X+TempX uses independent static and temporal GCN parameters, then the existing
concatenation fusion and classifier. X+TempLasso uses GCN only on the static
branch; its temporal branch is the existing selected-edge sequence and BiGRU.
Standalone TempLasso has no GCN and does not need another table entry per
backbone.

## Scientific scope

Report this as a **positive-adjacency GCN with signed FC node features**.
Different graph conventions (absolute edges, density thresholds, signed GCNs)
would be distinct baselines. Do not tune that convention on held-out results.

Keep frozen subject splits, dataset labels, preprocessing, validation selection,
and test protocol consistent with the other backbones. An identical candidate
count is a useful budget constraint, but does not by itself prove all model
families are equally optimized. This addition has no measured patient-data
accuracy or AUC yet.
