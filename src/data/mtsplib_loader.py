from __future__ import annotations

from pathlib import Path
from typing import List

from src.models.mtsp_instance import MTSPInstance
from src.data.tsplib_parser import parse_tsplib_file


BASE_TSPLIB_FILES = [
    "eil51.tsp",
    "berlin52.tsp",
    "eil76.tsp",
    "rat99.tsp",
]

M_VALUES = [2, 3, 5, 7]


def load_mtsplib_instances(base_folder: str = "data/raw/mtsplib") -> List[MTSPInstance]:
    """
    Încarcă benchmark-ul de tip mTSPLIB pornind de la fișierele TSPLIB de bază.
    Pentru fiecare fișier de bază se generează instanțe pentru m = 2, 3, 5, 7.
    """
    folder = Path(base_folder)

    if not folder.exists():
        raise FileNotFoundError(f"Folderul nu există: {folder}")

    instances: List[MTSPInstance] = []

    for file_name in BASE_TSPLIB_FILES:
        file_path = folder / file_name

        if not file_path.exists():
            raise FileNotFoundError(f"Lipsește fișierul necesar: {file_path}")

        for m in M_VALUES:
            instance = parse_tsplib_file(file_path=file_path, num_salesmen=m)
            instances.append(instance)

    return instances


def load_single_mtsplib_instance(
    base_name: str,
    num_salesmen: int,
    base_folder: str = "data/raw/mtsplib"
) -> MTSPInstance:
    """
    Încarcă o singură instanță derivată dintr-un fișier TSPLIB.
    Exemplu:
    load_single_mtsplib_instance("eil51", 3)
    """
    file_path = Path(base_folder) / f"{base_name}.tsp"

    if not file_path.exists():
        raise FileNotFoundError(f"Fișierul nu există: {file_path}")

    return parse_tsplib_file(file_path=file_path, num_salesmen=num_salesmen)