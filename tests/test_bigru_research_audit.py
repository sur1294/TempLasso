from __future__ import annotations

import torch
from torch.nn.utils.rnn import pack_padded_sequence

from unified_bn.temporal.bigru import BiGRUAggregator, BiGRUConfig
from unified_bn.temporal.pooling import lengths_to_mask


def _make_sequence() -> tuple[torch.Tensor, torch.Tensor]:
    """Return an intentionally unsorted variable-length batch."""
    torch.manual_seed(1201)
    sequence = torch.randn(4, 6, 5)
    lengths = torch.tensor([3, 6, 1, 4], dtype=torch.long)
    return sequence, lengths


def test_final_hidden_matches_explicit_top_layer_direction_layout() -> None:
    """
    Audit the bidirectional hidden-state interpretation used by the aggregator.

    PyTorch flattens [layer, direction] into the first dimension of h_n.  This
    test independently reshapes that tensor and verifies that the production
    embedding is exactly the top-layer forward state concatenated with the
    top-layer backward state.
    """
    torch.manual_seed(101)
    model = BiGRUAggregator(
        BiGRUConfig(
            input_dim=5,
            hidden_dim=4,
            num_layers=2,
            dropout=0.0,
            pooling="final",
        )
    )
    model.eval()

    sequence, lengths = _make_sequence()

    with torch.no_grad():
        output = model(sequence, lengths)

        packed = pack_padded_sequence(
            sequence,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, h_n = model.gru(packed)

    batch_size = sequence.shape[0]
    h_by_layer_direction = h_n.reshape(
        model.config.num_layers,
        2,
        batch_size,
        model.config.hidden_dim,
    )
    expected = torch.cat(
        [
            h_by_layer_direction[-1, 0],
            h_by_layer_direction[-1, 1],
        ],
        dim=-1,
    )

    torch.testing.assert_close(
        output.aux["final_hidden"],
        expected,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        output.embedding,
        expected,
        rtol=0.0,
        atol=0.0,
    )


def test_packed_batch_matches_independent_trimmed_subject_runs() -> None:
    """
    Verify that unsorted packed batching preserves each subject's true sequence.

    The reference calculation runs the same GRU separately on every subject
    after physically trimming padding, so it does not depend on packed-batch
    indexing or sorting restoration.
    """
    torch.manual_seed(202)
    model = BiGRUAggregator(
        BiGRUConfig(
            input_dim=5,
            hidden_dim=3,
            num_layers=2,
            dropout=0.0,
            pooling="final",
        )
    )
    model.eval()

    sequence, lengths = _make_sequence()

    with torch.no_grad():
        packed_batch_embedding = model(sequence, lengths).embedding

        expected_subjects = []
        for subject_index, length in enumerate(lengths.tolist()):
            trimmed = sequence[subject_index : subject_index + 1, :length]
            _, h_n = model.gru(trimmed)
            h_n = h_n.reshape(
                model.config.num_layers,
                2,
                1,
                model.config.hidden_dim,
            )
            expected_subjects.append(
                torch.cat([h_n[-1, 0, 0], h_n[-1, 1, 0]], dim=-1)
            )

    expected = torch.stack(expected_subjects, dim=0)
    torch.testing.assert_close(
        packed_batch_embedding,
        expected,
        rtol=1e-6,
        atol=1e-7,
    )


def test_batch_permutation_is_equivariant() -> None:
    """Reordering subjects must only reorder the corresponding embeddings."""
    torch.manual_seed(303)
    model = BiGRUAggregator(
        BiGRUConfig(
            input_dim=5,
            hidden_dim=4,
            num_layers=2,
            dropout=0.0,
            pooling="final",
        )
    )
    model.eval()

    sequence, lengths = _make_sequence()
    permutation = torch.tensor([2, 0, 3, 1], dtype=torch.long)
    inverse = torch.argsort(permutation)

    with torch.no_grad():
        original = model(sequence, lengths).embedding
        permuted = model(
            sequence[permutation],
            lengths[permutation],
        ).embedding

    torch.testing.assert_close(
        original,
        permuted[inverse],
        rtol=1e-6,
        atol=1e-7,
    )


def test_extreme_padding_is_invariant_for_every_pooling_mode() -> None:
    """Padded values must not influence final, mean, or attention embeddings."""
    sequence, lengths = _make_sequence()

    altered = sequence.clone()
    for subject_index, length in enumerate(lengths.tolist()):
        if length < sequence.shape[1]:
            altered[subject_index, length:] = (
                1e8 * torch.randn_like(altered[subject_index, length:])
            )

    for pooling in ("final", "mean", "attention"):
        torch.manual_seed(404)
        model = BiGRUAggregator(
            BiGRUConfig(
                input_dim=5,
                hidden_dim=4,
                pooling=pooling,
                attention_hidden_dim=7,
                attention_dropout=0.0,
            )
        )
        model.eval()

        with torch.no_grad():
            original = model(sequence, lengths).embedding
            changed = model(altered, lengths).embedding

        torch.testing.assert_close(
            original,
            changed,
            rtol=0.0,
            atol=0.0,
        )


def test_masked_mean_pooling_matches_manual_valid_step_mean() -> None:
    """Mean pooling must average recurrent outputs over valid steps only."""
    torch.manual_seed(505)
    model = BiGRUAggregator(
        BiGRUConfig(
            input_dim=5,
            hidden_dim=4,
            pooling="mean",
        )
    )
    model.eval()

    sequence, lengths = _make_sequence()

    with torch.no_grad():
        output = model(sequence, lengths)

    recurrent_sequence = output.aux["sequence_output"]
    mask = lengths_to_mask(
        lengths,
        max_steps=recurrent_sequence.shape[1],
        device=recurrent_sequence.device,
    )
    expected = (
        recurrent_sequence * mask.unsqueeze(-1).to(recurrent_sequence.dtype)
    ).sum(dim=1) / lengths.to(recurrent_sequence.dtype).unsqueeze(-1)

    torch.testing.assert_close(
        output.embedding,
        expected,
        rtol=0.0,
        atol=0.0,
    )


def test_attention_pooling_matches_manual_masked_softmax_and_weighted_sum() -> None:
    """
    Audit attention masking, normalization, and the subject-level weighted sum.
    """
    torch.manual_seed(606)
    model = BiGRUAggregator(
        BiGRUConfig(
            input_dim=5,
            hidden_dim=4,
            pooling="attention",
            attention_hidden_dim=6,
            attention_dropout=0.0,
        )
    )
    model.eval()

    sequence, lengths = _make_sequence()

    with torch.no_grad():
        output = model(sequence, lengths)
        recurrent_sequence = output.aux["sequence_output"]

        mask = lengths_to_mask(
            lengths,
            max_steps=recurrent_sequence.shape[1],
            device=recurrent_sequence.device,
        )
        manual_scores = model.pooler.scorer(recurrent_sequence).squeeze(-1)
        manual_scores = manual_scores.masked_fill(~mask, float("-inf"))
        manual_weights = torch.softmax(manual_scores, dim=1)
        manual_weights = manual_weights.masked_fill(~mask, 0.0)
        manual_embedding = torch.bmm(
            manual_weights.unsqueeze(1),
            recurrent_sequence,
        ).squeeze(1)

    torch.testing.assert_close(
        output.aux["attention_weights"],
        manual_weights,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        output.embedding,
        manual_embedding,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        output.aux["attention_weights"].sum(dim=1),
        torch.ones(sequence.shape[0]),
        rtol=1e-6,
        atol=1e-6,
    )
    assert torch.equal(
        output.aux["attention_weights"][2, 1:],
        torch.zeros(sequence.shape[1] - 1),
    )


def test_single_window_subject_matches_direct_one_step_gru() -> None:
    """A K=1 subject must have a well-defined bidirectional representation."""
    torch.manual_seed(707)
    model = BiGRUAggregator(
        BiGRUConfig(
            input_dim=5,
            hidden_dim=3,
            num_layers=1,
            dropout=0.0,
            pooling="final",
        )
    )
    model.eval()

    sequence = torch.randn(2, 4, 5)
    lengths = torch.tensor([1, 4], dtype=torch.long)

    with torch.no_grad():
        batch_output = model(sequence, lengths).embedding[0]
        _, h_n = model.gru(sequence[0:1, :1])
        h_n = h_n.reshape(1, 2, 1, 3)
        expected = torch.cat([h_n[0, 0, 0], h_n[0, 1, 0]], dim=-1)

    torch.testing.assert_close(
        batch_output,
        expected,
        rtol=1e-6,
        atol=1e-7,
    )


def test_single_layer_configured_gru_dropout_is_effectively_disabled() -> None:
    """
    Document PyTorch semantics: GRU dropout is inter-layer only.

    This is intentionally an audit finding.  With one recurrent layer the
    effective nn.GRU dropout must be zero, regardless of the configured value.
    """
    model = BiGRUAggregator(
        BiGRUConfig(
            input_dim=5,
            hidden_dim=4,
            num_layers=1,
            dropout=0.7,
            pooling="final",
        )
    )

    assert model.config.dropout == 0.7
    assert model.gru.dropout == 0.0


def test_multilayer_gru_dropout_is_stochastic_only_in_training_mode() -> None:
    """For stacked GRUs, dropout should affect train mode but not eval mode."""
    torch.manual_seed(808)
    model = BiGRUAggregator(
        BiGRUConfig(
            input_dim=5,
            hidden_dim=6,
            num_layers=2,
            dropout=0.5,
            pooling="final",
        )
    )
    sequence, lengths = _make_sequence()

    model.train()
    torch.manual_seed(11)
    train_a = model(sequence, lengths).embedding.detach()
    torch.manual_seed(12)
    train_b = model(sequence, lengths).embedding.detach()
    assert not torch.equal(train_a, train_b)

    model.eval()
    with torch.no_grad():
        torch.manual_seed(21)
        eval_a = model(sequence, lengths).embedding
        torch.manual_seed(22)
        eval_b = model(sequence, lengths).embedding

    torch.testing.assert_close(eval_a, eval_b, rtol=0.0, atol=0.0)


def test_gradients_reach_both_directions_and_top_recurrent_layer() -> None:
    """A temporal loss must train both directions and all stacked GRU layers."""
    torch.manual_seed(909)
    model = BiGRUAggregator(
        BiGRUConfig(
            input_dim=5,
            hidden_dim=4,
            num_layers=2,
            dropout=0.0,
            pooling="final",
        )
    )

    sequence, lengths = _make_sequence()
    sequence = sequence.clone().requires_grad_(True)

    loss = model(sequence, lengths).embedding.square().mean()
    loss.backward()

    parameters = {
        "weight_ih_l0": model.gru.weight_ih_l0,
        "weight_ih_l0_reverse": model.gru.weight_ih_l0_reverse,
        "weight_ih_l1": model.gru.weight_ih_l1,
        "weight_ih_l1_reverse": model.gru.weight_ih_l1_reverse,
    }

    assert sequence.grad is not None
    assert torch.isfinite(sequence.grad).all()
    assert torch.count_nonzero(sequence.grad).item() > 0

    for name, parameter in parameters.items():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert torch.count_nonzero(parameter.grad).item() > 0, name


def test_projection_and_layernorm_equal_explicit_post_pooling_transform() -> None:
    """Projection/normalization must be a pure post-pooling transformation."""
    torch.manual_seed(1001)
    model = BiGRUAggregator(
        BiGRUConfig(
            input_dim=5,
            hidden_dim=4,
            pooling="final",
            projection_dim=7,
            layer_norm=True,
        )
    )
    model.eval()

    sequence, lengths = _make_sequence()

    with torch.no_grad():
        output = model(sequence, lengths)
        expected = model.normalization(
            model.projection(output.aux["final_hidden"])
        )

    torch.testing.assert_close(
        output.embedding,
        expected,
        rtol=0.0,
        atol=0.0,
    )


def test_eval_forward_is_repeatable() -> None:
    """Evaluation must not contain hidden stochasticity."""
    torch.manual_seed(1101)
    model = BiGRUAggregator(
        BiGRUConfig(
            input_dim=5,
            hidden_dim=4,
            num_layers=2,
            dropout=0.4,
            pooling="attention",
            attention_hidden_dim=7,
            attention_dropout=0.4,
        )
    )
    model.eval()

    sequence, lengths = _make_sequence()

    with torch.no_grad():
        torch.manual_seed(1)
        first = model(sequence, lengths).embedding
        torch.manual_seed(9999)
        second = model(sequence, lengths).embedding

    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
