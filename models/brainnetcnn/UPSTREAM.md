# BrainNetCNN provenance

`brainnetcnn_model.py` is a transcription of the collaborator's harness copy
`ALTER/models/brainnetcnn/brainnetcnn.py` (shared 2026-09; identical to the
BrainNetworkTransformer repository's `BrainNetCNN` baseline), which implements
Kawahara et al., *BrainNetCNN: Convolutional neural networks for brain networks;
towards predicting neurodevelopment*, NeuroImage 2017.

Reference structure kept verbatim
---------------------------------
* `E2EBlock(in, out, R)`: `Conv2d(in,out,(1,R))` and `Conv2d(in,out,(R,1))`,
  row/column results broadcast by `torch.cat([a]*R, 3) + torch.cat([b]*R, 2)`.
* Layer stack and widths: E2E 1->32, E2E 32->64, E2N `Conv2d(64,1,(1,R))`,
  N2G `Conv2d(1,256,(R,1))`, dense 256->128->30->2.
* LeakyReLU with `negative_slope=0.33` after every layer **including the last
  dense layer** (the reference quirk; `final_activation=True` by default).
* Dropout `p=0.5` after N2G, dense1 and dense2.
* Module creation order (parameter draw under a shared seed) unchanged.

Deviations
----------
A. **Input.** The reference `forward(time_series, node_feature)` ignores the
   time series; this class takes only the FC batch `[B,R,R]` and validates it.
B. **Evaluation-time dropout fixed.** The reference calls
   `F.dropout(x, p=0.5)` without `training=self.training`, so dropout stays
   active under `model.eval()` and validation/test predictions are stochastic.
   This transcription uses `nn.Dropout` modules, which are disabled by
   `eval()`. Numbers produced by the collaborator's harness for BrainNetCNN
   are therefore *not* bit-comparable to this model (theirs are noisier).
C. **Configurable widths / dropout.** `BrainNetCNNConfig` exposes the channel
   widths, dense dims, dropout and slope; defaults reproduce the reference.
   The harness' `--dropout` / `--num_layers` flags never reached its
   BrainNetCNN (hard-coded 0.5), so its grids over those flags were no-ops.
D. **Embedding / head split.** The 256-d N2G output is exposed as the
   pre-classifier embedding (`return_embedding=True`); the dense head is
   grouped under `self.classifier` so the vanilla pipeline can reuse the
   reference head exactly, mirroring `StaticBQN` / `StaticALTER`.
E. Head-FLOPs bookkeeping present in the harness copy is omitted.
