import numpy as np
from PIL import Image
from sklearn.datasets import load_digits

from src.image_encoder import bilinear_sample, prepare_image, sample_digit_batch, sample_image


def test_coordinate_sampling_has_expected_range_and_size() -> None:
    image = Image.fromarray(np.full((10, 10), 128, dtype=np.uint8))
    xy = np.array([[-1, -1], [0, 0], [1, 1]], dtype=np.float32)
    _, samples = sample_image(image, xy)
    assert samples.shape == (3,)
    assert np.all((0 <= samples) & (samples <= 1))


def test_bilinear_sampling_hits_corners() -> None:
    image = np.array([[0, 1], [1, 0]], dtype=np.float32)
    samples = bilinear_sample(image, np.array([[-1, -1], [1, 1]], dtype=np.float32))
    np.testing.assert_allclose(samples, [0, 0])


def test_vectorized_digit_sampling_matches_single_image_sampling() -> None:
    images = np.arange(32, dtype=np.float32).reshape(2, 4, 4)
    xy = np.array([[-1, -1], [0, 0], [1, 1]], dtype=np.float32)
    expected = np.stack([bilinear_sample(image / images.max(), xy) for image in images])
    np.testing.assert_allclose(sample_digit_batch(images, xy), expected)


def test_prepare_image_recovers_hand_drawn_digit_off_center() -> None:
    # Simulate a canvas drawing: invert sklearn's bright-on-black digit to black ink on
    # white paper, blow it up, and paste it off-center on a larger blank canvas.
    digit = load_digits().images[0].astype(np.float32)
    ink = 255 - (digit / digit.max() * 255)
    ink_image = Image.fromarray(ink.astype(np.uint8), mode="L").resize((240, 240), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (400, 400), color=(255, 255, 255))
    canvas.paste(ink_image, (90, 130))

    prepared = prepare_image(canvas, size=8)
    expected = digit / digit.max()

    assert prepared.shape == (8, 8)
    correlation = np.corrcoef(prepared.ravel(), expected.ravel())[0, 1]
    assert correlation > 0.9


def test_prepare_image_returns_zeros_for_blank_canvas() -> None:
    blank = Image.new("RGB", (280, 280), color=(255, 255, 255))
    prepared = prepare_image(blank, size=8)
    assert prepared.shape == (8, 8)
    assert not prepared.any()
