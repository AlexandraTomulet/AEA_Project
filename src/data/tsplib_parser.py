from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

from src.models.mtsp_instance import MTSPInstance
from src.utils.distance import build_distance_matrix


def _parse_header_value(line: str) -> tuple[str, str]:
    """
    Parsează o linie de forma:
    KEY: VALUE
    """
    key, value = line.split(":", 1)
    return key.strip().upper(), value.strip()


def _parse_node_coord_section(lines: List[str], dimension: int) -> np.ndarray:
    """
    Parsează secțiunea NODE_COORD_SECTION dintr-un fișier TSPLIB.
    Format așteptat pe linie:
    node_id x y
    """
    coordinates: List[Tuple[float, float]] = []

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.upper() == "EOF":
            continue

        parts = stripped.split()
        if len(parts) < 3:
            continue

        # Ignorăm node_id-ul din fișier și păstrăm coordonatele
        _, x, y = parts[:3]
        coordinates.append((float(x), float(y)))

        if len(coordinates) == dimension:
            break

    if len(coordinates) != dimension:
        raise ValueError(
            f"Numărul de coordonate citite ({len(coordinates)}) "
            f"nu corespunde cu DIMENSION ({dimension})."
        )

    return np.array(coordinates, dtype=float)


def parse_tsplib_file(file_path: str | Path, num_salesmen: int) -> MTSPInstance:
    """
    Parsează un fișier TSPLIB .tsp și îl transformă într-un MTSPInstance.
    Convenția folosită pentru mTSPLIB:
    - primul oraș devine depozit -> depot_index = 0
    - num_salesmen este furnizat de noi
    """
    file_path = Path(file_path)

    with file_path.open("r", encoding="utf-8") as f:
        raw_lines = [line.rstrip("\n") for line in f]

    header: Dict[str, str] = {}
    coord_start_idx = None

    for idx, line in enumerate(raw_lines):
        stripped = line.strip()

        if not stripped:
            continue

        if stripped.upper() == "NODE_COORD_SECTION":
            coord_start_idx = idx + 1
            break

        if ":" in stripped:
            key, value = _parse_header_value(stripped)
            header[key] = value

    if coord_start_idx is None:
        raise ValueError(f"Fișierul {file_path.name} nu conține NODE_COORD_SECTION.")

    if "DIMENSION" not in header:
        raise ValueError(f"Fișierul {file_path.name} nu conține DIMENSION.")

    dimension = int(header["DIMENSION"])
    coordinates = _parse_node_coord_section(raw_lines[coord_start_idx:], dimension)
    distance_matrix = build_distance_matrix(coordinates)

    instance_name = file_path.stem

    return MTSPInstance(
        name=f"{instance_name}-m{num_salesmen}",
        num_nodes=dimension,
        num_salesmen=num_salesmen,
        depot_index=0,
        coordinates=coordinates,
        distance_matrix=distance_matrix,
        best_known_cost=None,
    )