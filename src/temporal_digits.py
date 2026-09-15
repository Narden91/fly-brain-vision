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

STATIC_TRANSFORMS = {
    "static": {},
    "translated-left": {"x_shift": -1.25},
    "translated-right": {"x_shift": 1.25},
    "rotated-clockwise": {"angle": -12},
    "rotated-counterclockwise": {"angle": 12},
}

MOTION_TRANSFORMS = {
    "motion-left": ("x_shift", -1),
    "motion-right": ("x_shift", 1),
    "motion-up": ("y_shift", -1),
    "motion-down": ("y_shift", 1),
}


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


def _static_sequence(images: np.ndarray, frames: int, transform: dict[str, float]) -> np.ndarray:
    if transform:
        images = np.stack([_transform(image, **transform) for image in images])
    return np.repeat(images[:, None], frames, axis=1)


def _motion_sequence(images: np.ndarray, frames: int, parameter: str, direction: int) -> np.ndarray:
    offsets = np.linspace(-1.5, 1.5, frames, dtype=np.float32)
    return np.stack(
        [
            np.stack([_transform(image, **{parameter: direction * offset}) for offset in offsets])
            for image in images
        ]
    )


def _variant_sequence(images: np.ndarray, variant: str, frames: int) -> np.ndarray:
    if variant in STATIC_TRANSFORMS:
        return _static_sequence(images, frames, STATIC_TRANSFORMS[variant])
    parameter, direction = MOTION_TRANSFORMS[variant]
    return _motion_sequence(images, frames, parameter, direction)


def build_temporal_digit_set(
    images: np.ndarray, labels: np.ndarray, *, frames: int = 8, source_indices: np.ndarray | None = None
) -> TemporalDigitSet:
    images = np.asarray(images, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    if images.ndim != 3 or images.shape[1:] != (8, 8) or len(images) != len(labels) or frames < 2:
        raise ValueError("Expected equally sized 8x8 images and labels with at least two frames.")
    scale = float(images.max())
    if scale > 1:
        images = images / scale
    if source_indices is None:
        source_indices = np.arange(len(images), dtype=np.int64)
    else:
        source_indices = np.asarray(source_indices, dtype=np.int64)
    if len(source_indices) != len(images):
        raise ValueError("source_indices must have one item per image.")

    sequences: list[np.ndarray] = []
    expanded_labels: list[np.ndarray] = []
    expanded_sources: list[np.ndarray] = []
    variants: list[np.ndarray] = []
    for variant_index, variant in enumerate(VARIANT_NAMES):
        sequences.append(_variant_sequence(images, variant, frames))
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
