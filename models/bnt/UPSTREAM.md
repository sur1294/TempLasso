# Provenance of the `bnt` backbone

The user requested the transformer under `ALTER/models/transformer` from
`ALTER_code_only.zip`. That file defines **GraphTransformer**. The framework
uses the requested short registry name **`bnt`**, while recording the original
class here. This is a port of that exact supplied model, not a claim that it
reproduces a separately published Brain Network Transformer architecture.

The archive's different `models/lrbgt/model.py::BrainNetworkTransformer` is
the ALTER/LRBGT path with positional encoding and cluster pooling. It is not
the source of this new backbone. Existing ALTER remains unchanged.

- Source member: `ALTER/models/transformer/transformer.py`
- Source file SHA-256:
  `7477f8d6d4f96c135b7d0f6c80ad60ff688f846cee7312ec43c90495c5aa679c`
- Source archive SHA-256:
  `f42e384619a2c0cabe518e5343ab60a12c776397352189711539ebec8268f0bb`
- An exact byte copy is retained as `GraphTransformer.source.txt` for reference
  equality tests. It is not imported by the production model.

## Preserved computation

Each FC row is a token, so Schaefer100 input is `[B,100,100]`. The supplied
default configuration uses two separately initialized TransformerEncoderLayer
blocks, four heads, FFN dimension 1024, and PyTorch's default ReLU, post-layer
normalization, bias, and dropout 0.1. Mean readout gives a 100-dimensional
embedding, followed by `100 -> 256 -> 32 -> classes` with LeakyReLU between
linear layers. There is no added positional encoding, DEC, or learned edge mask.

All four source readouts are preserved:

- mean: average ROI embeddings within one FC matrix; dimension R;
- max: maximum across ROI embeddings; dimension R;
- sum: sum across ROIs followed by BatchNorm1d; dimension R;
- concat: linear R->8 with LeakyReLU per ROI, then flatten; dimension 8R.

This spatial readout does not average temporal windows. TempBNT still processes
the complete ordered sequence of valid windows through the shared BNT and BiGRU.

Module creation order and state-dict names (`attention_list`, `dim_reduction`,
`norm`, `fc`) match the source. Tests compare seeded initial weights, forward
outputs, parameter/input gradients, and strict state loading. The default
binary Schaefer100 model has 527,594 parameters.

## Interface changes

1. Accept a typed dataclass and one FC argument; the original ignored its
   `time_series` argument. No OmegaConf dependency or CUDA-only factory is copied.
2. Expose the embedding before the original MLP for the shared framework.
   `classifier` is a property alias of `fc`, so static mode reuses the same head.
3. Parameterize output classes (2 for ABIDE/ADNI, 4 for the project's PPMI task).
   Layers, dropout, readout, head count, and FFN dimension are explicit identity
   fields. Defaults match the supplied model; grid variations are recorded.
4. Validate shapes/configurations early. The source sum readout's BatchNorm
   still requires at least two training examples; an explicit error explains
   this case. The default mean readout and supplied search grids do not use BN.
5. Do not carry over the broken source `get_attention_weights()` method:
   ordinary TransformerEncoderLayer has no such accessor. No attention maps
   are invented or claimed by this adapter.

The source harness's `num_heads`, `ffn_hidden_dim`, and `dropout` settings do
not reach its GraphTransformer: that file hard-codes four heads and FFN 1024
and inherits dropout 0.1. This port uses those actual defaults and explicitly
applies any newly requested dropout value, rather than reproducing a no-op flag.
