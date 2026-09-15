"""Deterministic static and temporal variants for the sklearn digits task."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.ndimage import rotate, shift


VARIANT_NAMES = (
    "static",
    "translated-left",
    "translated-right",
    "rotated-clockwise",
    "rotated-counterclockwise",
    "motion-left",
    "motion-right",
    "motion-up",
    "motion-down",
)


@dataclass(frozen=True)
class TemporalDigitSet:
    frames: np.ndarray
    labels: np.ndarray
    source_indices: np.ndarray
    variants: np.ndarray


def _transform(image: np.ndarray, *, x_shift: float = 0, y_shift: float = 0, angle: float = 0) -> np.ndarray:
    transformed = image
    if angle:
        transformed = rotate(transformed, angle, reshape=False, order=1, mode="constant", cval=0.0, prefilter=False)
    if x_shift or y_shift:
        transformed = shift(transformed, (y_shift, x_shift), order=1, mode="constant", cval=0.0, prefilter=False)
    return np.clip(transformed, 0, 1).astype(np.float32, copy=False)


def build_temporal_digit_set(images: np.ndarray, labels: np.ndarray, *, frames: int = 8, source_indices: np.ndarray | None = None) -> TemporalDigitSet:
    """Create deterministic augmentations after callers split original digit IDs."""
    images = np.asarray(images, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    if images.ndim != 3 or images.shape[1:] != (8, 8) or len(images) != len(labels) or frames < 2:
        raise ValueError("Expected equally sized 8x8 images and labels with at least two frames.")
    scale = float(images.max())
    if scale > 1:
        images = images / scale
    source_indices = np.arange(len(images), dtype=np.int64) if source_indices is None else np.asarray(source_indices, dtype=np.int64)
    if len(source_indices) != len(images):
        raise ValueError("source_indices must have one item per image.")

    sequences: list[np.ndarray] = []
    expanded_labels: list[np.ndarray] = []
    expanded_sources: list[np.ndarray] = []
    variants: list[np.ndarray] = []
    motion = np.linspace(-1.5, 1.5, frames, dtype=np.float32)
    for variant_index, variant in enumerate(VARIANT_NAMES):
        if variant == "static":
            sequence = np.repeat(images[:, None], frames, axis=1)
        elif variant == "translated-left":
            sequence = np.repeat(np.stack([_transform(image, x_shift=-1.25) for image in images])[:, None], frames, axis=1)
        elif variant == "translated-right":
            sequence = np.repeat(np.stack([_transform(image, x_shift=1.25) for image in images])[:, None], frames, axis=1)
        elif variant == "rotated-clockwise":
            sequence = np.repeat(np.stack([_transform(image, angle=-12) for image in images])[:, None], frames, axis=1)
        elif variant == "rotated-counterclockwise":
            sequence = np.repeat(np.stack([_transform(image, angle=12) for image in images])[:, None], frames, axis=1)
        else:
            axis = 0 if variant in {"motion-up", "motion-down"} else 1
            direction = -1 if variant in {"motion-left", "motion-up"} else 1
            sequence = np.stack(
                [
                    np.stack(
                        [_transform(image, **({"y_shift": direction * amount} if axis == 0 else {"x_shift": direction * amount})) for amount in motion]
                    )
                    for image in images
                ]
            )
        sequences.append(sequence)
        expanded_labels.append(labels)
        expanded_sources.append(source_indices)
        variants.append(np.full(len(images), variant_index, dtype=np.int8))
    return TemporalDigitSet(
        frames=np.concatenate(sequences).astype(np.float32),
        labels=np.concatenate(expanded_labels),
        source_indices=np.concatenate(expanded_sources),
        variants=np.concatenate(variants),
    )


def sample_sequence_batch(images: torch.Tensor, xy: np.ndarray) -> torch.Tensor:
    """Bilinearly sample ``(batch, frames, 8, 8)`` grids at optic-column positions."""
    if images.ndim != 4 or images.shape[-2:] != (8, 8):
        raise ValueError("Expected image batches shaped (batch, frames, 8, 8).")
    xy = np.asarray(xy, dtype=np.float32)
    x = np.clip((xy[:, 0] + 1) * 3.5, 0, 7)
    y = np.clip((xy[:, 1] + 1) * 3.5, 0, 7)
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    x1, y1 = np.minimum(x0 + 1, 7), np.minimum(y0 + 1, 7)
    flat = images.reshape(-1, 8, 8)
    device = images.device
    x0_t, x1_t = torch.as_tensor(x0, device=device), torch.as_tensor(x1, device=device)
    y0_t, y1_t = torch.as_tensor(y0, device=device), torch.as_tensor(y1, device=device)
    dx = torch.as_tensor(x - x0, dtype=images.dtype, device=device)
    dy = torch.as_tensor(y - y0, dtype=images.dtype, device=device)
    sampled = (
        (1 - dx) * (1 - dy) * flat[:, y0_t, x0_t]
        + dx * (1 - dy) * flat[:, y0_t, x1_t]
        + (1 - dx) * dy * flat[:, y1_t, x0_t]
        + dx * dy * flat[:, y1_t, x1_t]
    )
    return sampled.reshape(*images.shape[:2], len(xy))
