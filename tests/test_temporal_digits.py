import numpy as np
import torch

from src.temporal_digits import VARIANT_NAMES, build_temporal_digit_set, sample_sequence_batch


def test_variants_preserve_labels_and_source_split() -> None:
    images = np.zeros((4, 8, 8), dtype=np.float32)
    images[:, 3:5, 3:5] = 1
    labels = np.array([1, 2, 3, 4])
    train = build_temporal_digit_set(images[:2], labels[:2], source_indices=np.array([0, 1]))
    test = build_temporal_digit_set(images[2:], labels[2:], source_indices=np.array([2, 3]))
    assert train.frames.shape == (2 * len(VARIANT_NAMES), 8, 8, 8)
    assert set(train.source_indices).isdisjoint(test.source_indices)
    assert set(train.labels) == {1, 2}
    assert set(test.labels) == {3, 4}


def test_sequence_sampler_has_expected_shape_and_finite_values() -> None:
    frames = torch.rand((2, 8, 8, 8))
    xy = np.array([[-1, -1], [0, 0], [1, 1]], dtype=np.float32)
    sampled = sample_sequence_batch(frames, xy)
    assert sampled.shape == (2, 8, 3)
    assert torch.isfinite(sampled).all()
