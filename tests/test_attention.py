import numpy as np

from modules.attention import causal_mask, run_attention, softmax, tokenize_words


def test_adjacent_emojis_are_separate_tokens():
    assert tokenize_words("😀😃😄") == ["😀", "😃", "😄"]
    assert tokenize_words("hello 😀😃 world") == ["hello", "😀", "😃", "world"]


def test_softmax_rows_sum_to_one():
    values = softmax(np.array([[1.0, 2.0], [1000.0, 999.0]]), axis=1)
    assert np.allclose(values.sum(axis=1), 1.0)


def test_causal_mask_values():
    assert np.array_equal(causal_mask(3), np.array([[0.0, -1e9, -1e9], [0.0, 0.0, -1e9], [0.0, 0.0, 0.0]]))


def test_future_weights_are_zero_and_rows_sum_to_one():
    result = run_attention("one two three", dimension=8)
    assert np.allclose(result.weights.sum(axis=1), 1.0)
    assert np.all(result.weights[np.triu_indices(3, k=1)] == 0.0)


def test_unmasked_reference_differs_from_causal_attention():
    causal = run_attention("one two", dimension=8)
    unmasked = run_attention("one two", dimension=8, causal=False)
    assert np.array_equal(causal.raw_scores, unmasked.raw_scores)
    assert not np.array_equal(causal.weights, unmasked.weights)


def test_scaling_changes_attention_values():
    scaled = run_attention("one two", dimension=8, scale=True)
    unscaled = run_attention("one two", dimension=8, scale=False)
    assert not np.array_equal(scaled.scaled_scores, unscaled.scaled_scores)
    assert not np.array_equal(scaled.weights, unscaled.weights)


def test_output_shape():
    result = run_attention(["one", "two"], dimension=16)
    assert result.output.shape == (2, 16)


def test_attention_is_deterministic():
    first = run_attention("same input", dimension=8, seed=12)
    second = run_attention("same input", dimension=8, seed=12)
    assert np.array_equal(first.weights, second.weights)
    assert np.array_equal(first.output, second.output)


def test_hand_worked_two_token_attention_shape_and_first_row():
    result = run_attention(["a", "b"], dimension=8, seed=7)
    assert np.allclose(result.weights[0], np.array([1.0, 0.0]))
    assert np.isfinite(result.output).all()
