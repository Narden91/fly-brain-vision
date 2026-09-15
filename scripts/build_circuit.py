"""Download MaleCNS inputs once and build a compact offline circuit artifact."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATASET = "male-cns:v1.0"
OPTIC_COLUMNS_URL = (
    "https://raw.githubusercontent.com/flyconnectome/2025malecns/main/"
    "supplemental_data/optic-column-type-assignments-v1.0.xlsx"
)
REQUIRED_EDGE_COLUMNS = {"bodyId_pre", "bodyId_post", "weight"}


def download_optic_columns(destination: Path) -> Path:
    """Cache the official spreadsheet; no neuPrint request happens here."""
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        response = requests.get(OPTIC_COLUMNS_URL, timeout=60)
        response.raise_for_status()
        destination.write_bytes(response.content)
    except requests.RequestException as exc:
        raise RuntimeError(
            "Could not download the official MaleCNS optic-column spreadsheet. "
            "Check network access and retry."
        ) from exc
    return destination


def _column_name(columns: list[str], needle: str) -> str | None:
    needle = needle.casefold()
    exact = [column for column in columns if column.casefold() == needle]
    return (exact or [column for column in columns if needle in column.casefold()] or [None])[0]


def _body_id(value: object) -> int | None:
    if pd.isna(value):
        return None
    try:
        body_id = int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None
    return body_id if body_id > 0 else None


def load_right_eye_l1(path: Path) -> list[dict[str, int | float | str]]:
    """Parse column labels defensively and return normalized right-eye L1 positions."""
    table = pd.read_excel(path)
    columns = [str(column) for column in table.columns]
    column_key = _column_name(columns, "column")
    l1_key = _column_name(columns, "l1")
    if column_key is None or l1_key is None:
        raise ValueError("Optic-column spreadsheet needs a column label and an L1 body-ID column.")

    parsed: list[dict[str, int | float | str]] = []
    for _, row in table.iterrows():
        column = str(row[column_key]).strip()
        match = re.search(r"_(-?\d+)_(-?\d+)$", column)
        body_id = _body_id(row[l1_key])
        if not match or body_id is None or "_R_" not in column.upper():
            continue
        parsed.append(
            {"body_id": body_id, "column": column, "x": int(match.group(1)), "y": int(match.group(2))}
        )
    if not parsed:
        raise ValueError("No valid right-eye L1 visual input neurons were found in the spreadsheet.")

    for axis in ("x", "y"):
        values = np.asarray([item[axis] for item in parsed], dtype=np.float32)
        span = float(values.max() - values.min())
        normalized = np.zeros_like(values) if span == 0 else 2 * (values - values.min()) / span - 1
        for item, value in zip(parsed, normalized, strict=True):
            item[axis] = float(value)
    return parsed


def _edges_or_error(edges: pd.DataFrame, label: str) -> pd.DataFrame:
    missing = REQUIRED_EDGE_COLUMNS.difference(edges.columns)
    if missing:
        raise ValueError(f"neuPrint {label} response is missing required columns: {', '.join(sorted(missing))}.")
    return edges.loc[:, ["bodyId_pre", "bodyId_post", "weight"]].copy()


def _fetch_annotations(selected_ids: list[int], client) -> pd.DataFrame:
    from neuprint import NeuronCriteria, fetch_neurons

    result = fetch_neurons(NeuronCriteria(bodyId=selected_ids), client=client)
    return result[0] if isinstance(result, tuple) else result


def _annotation_values(annotations: pd.DataFrame, selected_ids: list[int], field: str) -> list[str | None]:
    if "bodyId" not in annotations.columns or field not in annotations.columns:
        return [None] * len(selected_ids)
    lookup = annotations.set_index("bodyId")[field]
    return [None if pd.isna(lookup.get(body_id)) else str(lookup.get(body_id)) for body_id in selected_ids]


def build_artifact(
    inputs: list[dict[str, int | float | str]],
    *,
    client,
    hops: int,
    min_edge_weight: int,
    internal_min_edge_weight: int,
    max_neurons: int,
    matrix_path: Path,
    metadata_path: Path,
) -> dict:
    """Select two downstream hops, then save normalized ``W[post, pre]``."""
    from neuprint import fetch_adjacencies

    input_ids = [int(item["body_id"]) for item in inputs]
    if len(input_ids) > max_neurons:
        raise ValueError("MAX_NEURONS is smaller than the number of visual input neurons.")
    selected, current = set(input_ids), input_ids
    for hop in range(hops):
        if not current or len(selected) >= max_neurons:
            break
        _, edges = fetch_adjacencies(
            sources=current,
            targets=None,
            min_total_weight=min_edge_weight,
            omit_rois=True,
            weight_props=["weight"],
            client=client,
        )
        edges = _edges_or_error(edges, "hop")
        candidates = edges.groupby("bodyId_post", as_index=False)["weight"].sum().sort_values("weight", ascending=False)
        room = max_neurons - len(selected)
        remaining_hops = hops - hop
        limit = room if remaining_hops == 1 else max(1, room // remaining_hops)
        current = [int(body_id) for body_id in candidates["bodyId_post"] if int(body_id) not in selected][:limit]
        selected.update(current)

    selected_ids = sorted(selected)
    if len(selected_ids) > max_neurons:
        raise ValueError("Circuit selection exceeded MAX_NEURONS.")
    _, connections = fetch_adjacencies(
        sources=selected_ids,
        targets=selected_ids,
        min_total_weight=internal_min_edge_weight,
        omit_rois=True,
        weight_props=["weight"],
        client=client,
    )
    connections = _edges_or_error(connections, "internal-circuit")
    index = {body_id: i for i, body_id in enumerate(selected_ids)}
    rows = connections["bodyId_post"].map(index).to_numpy()
    cols = connections["bodyId_pre"].map(index).to_numpy()
    if pd.isna(rows).any() or pd.isna(cols).any():
        raise ValueError("neuPrint returned an edge outside the selected circuit.")
    strengths = np.log1p(connections["weight"].to_numpy(dtype=np.float32))
    W = sparse.csr_matrix((strengths, (rows.astype(int), cols.astype(int))), shape=(len(selected_ids), len(selected_ids)))
    W.sum_duplicates()
    row_total = np.asarray(np.abs(W).sum(axis=1)).ravel()
    W = sparse.diags(np.divide(1.0, np.maximum(row_total, 1e-12), where=row_total > 0, out=np.zeros_like(row_total))) @ W
    W = W.tocsr().astype(np.float32)

    try:
        annotations = _fetch_annotations(selected_ids, client)
    except Exception as exc:
        raise RuntimeError("Could not fetch annotations for selected MaleCNS neurons.") from exc
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(matrix_path, W)
    metadata = {
        "dataset": DATASET,
        "number_of_neurons": len(selected_ids),
        "number_of_edges": int(W.nnz),
        "input_neuron_count": len(input_ids),
        "input_body_ids": input_ids,
        "body_ids": selected_ids,
        "cell_types": _annotation_values(annotations, selected_ids, "type"),
        "superclasses": _annotation_values(annotations, selected_ids, "superclass"),
        "input_xy": [[item["x"], item["y"]] for item in inputs],
        "min_edge_weight": min_edge_weight,
        "internal_min_edge_weight": internal_min_edge_weight,
        "hops": hops,
        "normalization": "log1p(anatomical synapse count), then each postsynaptic row has L1 norm 1.",
        "generated_at": datetime.now(UTC).isoformat(),
        "scientific_note": "MaleCNS supplies anatomical connectivity and synapse counts. Dynamics, input encoding, normalization, and classifier are simplified computational assumptions.",
        "annotations": {
            field: _annotation_values(annotations, selected_ids, field)
            for field in ("instance", "consensusNt", "predictedNt", "celltypePredictedNt")
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a compact MaleCNS visual circuit for offline use.")
    parser.add_argument("--hops", type=int, default=2)
    parser.add_argument("--min-edge-weight", type=int, default=5)
    parser.add_argument("--internal-min-edge-weight", type=int, default=3)
    parser.add_argument("--max-neurons", type=int, default=3000)
    args = parser.parse_args()
    token = os.environ.get("NEUPRINT_APPLICATION_CREDENTIALS")
    if not token:
        raise SystemExit(
            "NEUPRINT_APPLICATION_CREDENTIALS is required. Create a free token at "
            "https://neuprint.janelia.org, export it, then run this command again."
        )
    try:
        from neuprint import Client

        inputs = load_right_eye_l1(download_optic_columns(ROOT / "data/raw/optic-column-type-assignments-v1.0.xlsx"))
        client = Client("neuprint.janelia.org", dataset=DATASET, token=token)
        metadata = build_artifact(
            inputs,
            client=client,
            hops=args.hops,
            min_edge_weight=args.min_edge_weight,
            internal_min_edge_weight=args.internal_min_edge_weight,
            max_neurons=args.max_neurons,
            matrix_path=ROOT / "data/malecns_circuit.npz",
            metadata_path=ROOT / "data/malecns_circuit_meta.json",
        )
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"Circuit build failed: {exc}") from exc
    print(json.dumps({key: metadata[key] for key in ("number_of_neurons", "number_of_edges", "input_neuron_count")}, indent=2))


if __name__ == "__main__":
    main()
