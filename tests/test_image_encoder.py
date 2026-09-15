import numpy as np
from PIL import Image

from src.image_encoder import bilinear_sample, sample_image


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
