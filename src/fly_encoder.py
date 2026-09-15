from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from PIL import Image, ImageOps

import flyvis
from flyvis.datasets.rendering import BoxEye
from flyvis.utils.hex_utils import get_hex_coords, hex_to_pixel


DT = 1 / 100
FRAMES = 20
HEX_EXTENT = 15
HEXALS = 721
MEDULLA_PREFIXES = ("Mi", "Tm", "C", "Mt")


@dataclass
class EncodingResult:
    features: np.ndarray
    retina: np.ndarray
    activity_names: list[str]
    activity_values: np.ndarray


class FlyEncoder:
    """Frozen FlyVis visual-system model used as an image feature extractor."""

    def __init__(self) -> None:
        self.eye = BoxEye(extent=HEX_EXTENT, kernel_size=13)
        network_view = flyvis.NetworkView(flyvis.results_dir / "flow/0000/000")
        self.network = network_view.init_network()
        self.network.eval()

        nodes = self.network.connectome.nodes
        self.cell_types = nodes.type[:].astype(str)
        self.u = nodes.u[:]
        self.v = nodes.v[:]

        unique = sorted(set(self.cell_types.tolist()))
        self.feature_types = [
            name for name in unique if name.startswith(MEDULLA_PREFIXES)
        ]
        if not self.feature_types:
            raise RuntimeError("No medulla/Tm feature cell types found in the FlyVis connectome.")

        hu, hv = get_hex_coords(HEX_EXTENT)
        self.hex_u = np.asarray(hu)
        self.hex_v = np.asarray(hv)
        hx, hy = hex_to_pixel(hu, hv)
        self.hex_x = np.asarray(hx)
        self.hex_y = np.asarray(hy)

        hex_key = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(hu, hv))}
        self.hex_col = np.asarray(
            [hex_key[(int(a), int(b))] for a, b in zip(self.u, self.v)],
            dtype=np.int64,
        )
        self.rows_by_type = {
            cell_type: np.where(self.cell_types == cell_type)[0]
            for cell_type in self.feature_types
        }

    @staticmethod
    def preprocess_image(image: Image.Image) -> np.ndarray:
        """Convert an uploaded image to an MNIST-like 64x64 float image."""
        image = ImageOps.autocontrast(image.convert("L"))
        image = image.resize((64, 64), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0

        # MNIST is bright foreground on a dark background. Invert common white-paper inputs.
        if float(array.mean()) > 0.5:
            array = 1.0 - array
        return np.clip(array, 0.0, 1.0)

    def _scatter_features(self, response_mean: np.ndarray) -> np.ndarray:
        batch = response_mean.shape[0]
        packed = np.zeros(
            (batch, len(self.feature_types), HEXALS), dtype=np.float32
        )
        for type_idx, cell_type in enumerate(self.feature_types):
            rows = self.rows_by_type[cell_type]
            packed[:, type_idx, self.hex_col[rows]] = response_mean[:, rows]
        return packed.reshape(batch, -1)

    def encode_batch(self, images: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Encode B grayscale images in [0, 1].

        Returns:
            features: (B, n_medulla_types * 721)
            retina: (B, 721), time-averaged photoreceptor input
            type_activity: (B, n_medulla_types), mean response per selected cell type
        """
        images = np.asarray(images, dtype=np.float32)
        if images.ndim != 3 or images.shape[1:] != (64, 64):
            raise ValueError(f"Expected (B, 64, 64), got {images.shape}")

        movie = np.repeat(images[:, None, :, :], FRAMES, axis=1)
        movie_t = torch.as_tensor(movie, dtype=torch.float32, device=flyvis.device)

        with torch.no_grad():
            rendered = self.eye(movie_t)  # (B, T, 1, 721)
            state = self.network.fade_in_state(1.0, DT, rendered[:, 0])
            responses = self.network.simulate(rendered, DT, initial_state=state)

        retina = rendered[:, :, 0].mean(dim=1).detach().cpu().numpy().astype(np.float32)
        response_mean = responses.mean(dim=1).detach().cpu().numpy().astype(np.float32)
        features = self._scatter_features(response_mean)

        type_activity = np.stack(
            [response_mean[:, self.rows_by_type[name]].mean(axis=1) for name in self.feature_types],
            axis=1,
        ).astype(np.float32)

        return features, retina, type_activity

    def encode_image(self, image: Image.Image) -> EncodingResult:
        array = self.preprocess_image(image)
        features, retina, activity = self.encode_batch(array[None])
        return EncodingResult(
            features=features[0],
            retina=retina[0],
            activity_names=self.feature_types,
            activity_values=activity[0],
        )
