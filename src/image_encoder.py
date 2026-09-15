"""Image preprocessing and sampling at real MaleCNS optic-column coordinates."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps


def prepare_image(image: Image.Image, size: int = 8) -> np.ndarray:
    """Return a digit cropped to its ink, centered, and downsampled to ``size``x``size`` in [0, 1].

    Matches how scikit-learn's digits dataset is framed (ink fills the frame), since the
    classifier is trained on that dataset. Bright foreground on a dark background.
    """
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        composited = Image.new("RGBA", image.size, (255, 255, 255, 255))
        composited.paste(image.convert("RGBA"), mask=image.convert("RGBA"))
        image = composited.convert("RGB")
    gray = ImageOps.autocontrast(image.convert("L"))
    values = np.asarray(gray, dtype=np.float32) / 255.0
    if float(values.mean()) > 0.5:
        values = 1.0 - values

    ink = Image.fromarray((values * 255).astype(np.uint8))
    bbox = ink.point(lambda p: 255 if p > 0.1 * 255 else 0).getbbox()
    if bbox is None:
        return np.zeros((size, size), dtype=np.float32)

    cropped = ink.crop(bbox)
    side = max(cropped.width, cropped.height)
    square = Image.new("L", (side, side), color=0)
    square.paste(cropped, ((side - cropped.width) // 2, (side - cropped.height) // 2))
    resized = square.resize((size, size), resample=Image.Resampling.BOX)

    out = np.asarray(resized, dtype=np.float32)
    peak = float(out.max())
    return out / peak if peak > 0 else out


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


def sample_image(image: Image.Image, xy: np.ndarray, size: int = 8) -> tuple[np.ndarray, np.ndarray]:
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
