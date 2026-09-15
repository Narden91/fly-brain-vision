"""Image preprocessing and sampling at real MaleCNS optic-column coordinates."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps


def prepare_image(image: Image.Image, size: int = 64) -> np.ndarray:
    """Return a centered grayscale image with bright digit foreground in [0, 1]."""
    gray = ImageOps.autocontrast(image.convert("L"))
    resized = ImageOps.contain(gray, (size, size), method=Image.Resampling.BILINEAR)
    canvas = Image.new("L", (size, size), color=0)
    canvas.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
    values = np.asarray(canvas, dtype=np.float32) / 255.0
    if float(values.mean()) > 0.5:
        values = 1.0 - values
    return values


def bilinear_sample(image: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Sample a 2-D image at normalized ``[-1, 1]`` x/y column coordinates."""
    image = np.asarray(image, dtype=np.float32)
    xy = np.asarray(xy, dtype=np.float32)
    if image.ndim != 2 or xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("Expected a 2-D image and coordinates shaped (n, 2).")
    if not np.isfinite(image).all() or not np.isfinite(xy).all():
        raise ValueError("Image and coordinates must be finite.")
    x0, x1, y0, y1, dx, dy = _bilinear_coordinates(xy, image.shape)
    return ((1 - dx) * (1 - dy) * image[y0, x0] + dx * (1 - dy) * image[y0, x1]
            + (1 - dx) * dy * image[y1, x0] + dx * dy * image[y1, x1]).astype(np.float32)


def sample_image(image: Image.Image, xy: np.ndarray, size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    prepared = prepare_image(image, size=size)
    return prepared, bilinear_sample(prepared, xy)


def _bilinear_coordinates(xy: np.ndarray, image_shape: tuple[int, int]) -> tuple[np.ndarray, ...]:
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError("Expected finite coordinates shaped (n, 2).")
    height, width = image_shape
    x = np.clip((xy[:, 0] + 1.0) * 0.5 * (width - 1), 0, width - 1)
    y = np.clip((xy[:, 1] + 1.0) * 0.5 * (height - 1), 0, height - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    return x0, np.minimum(x0 + 1, width - 1), y0, np.minimum(y0 + 1, height - 1), x - x0, y - y0


def sample_digit_batch(images: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Sample scikit-learn digit images. They are already bright foreground on black."""
    images = np.asarray(images, dtype=np.float32)
    if images.ndim != 3:
        raise ValueError("Expected digit images shaped (batch, height, width).")
    if not np.isfinite(images).all():
        raise ValueError("Digit images must be finite.")
    scale = float(images.max())
    if scale > 1:
        images = images / scale
    xy = np.asarray(xy, dtype=np.float32)
    x0, x1, y0, y1, dx, dy = _bilinear_coordinates(xy, images.shape[1:])
    return (
        (1 - dx) * (1 - dy) * images[:, y0, x0]
        + dx * (1 - dy) * images[:, y0, x1]
        + (1 - dx) * dy * images[:, y1, x0]
        + dx * dy * images[:, y1, x1]
    ).astype(np.float32)


def plot_samples(ax, xy: np.ndarray, values: np.ndarray) -> None:
    ax.scatter(xy[:, 0], xy[:, 1], c=values, s=24, cmap="gray", vmin=0, vmax=1)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")
