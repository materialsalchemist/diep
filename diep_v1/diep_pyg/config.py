"""Global configuration and default dtypes for the standalone PyG DIEP package.

Vendored from the DGL-era ``diep.config`` and ``diep.__init__`` so this package has no
parent package to import from. The dtype globals live here rather than in ``__init__``
so that submodules can ``from diep_pyg import config`` without a circular import.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import torch
from pymatgen.core.periodic_table import Element

# Default set of elements supported by universal DIEP models. Excludes radioactive and
# most artificial elements.
DEFAULT_ELEMENTS = tuple(el.symbol for el in Element if el.symbol not in ["Po", "At", "Rn", "Fr", "Ra"] and el.Z < 95)

# Default location of the cache, e.g. for storing downloaded models.
DIEP_CACHE = Path(os.path.expanduser("~")) / ".cache/diep"
os.makedirs(DIEP_CACHE, exist_ok=True)

# Download url for pre-trained models.
PRETRAINED_MODELS_BASE_URL = "http://github.com/materialsvirtuallab/diep/raw/main/pretrained_models/"

# Default datatype definitions. Modules read these as ``config.float_th`` rather than
# importing the names directly, so ``set_default_dtype`` is visible to every caller.
float_np = np.float32
float_th = torch.float32

int_np = np.int32
int_th = torch.int32


def set_default_dtype(type_: str = "float", size: int = 32) -> None:
    """Set the default dtype size (16, 32 or 64) for int or float used throughout the package.

    Args:
        type_: "float" or "int".
        size: 16, 32 or 64.
    """
    if size in (16, 32, 64):
        globals()[f"{type_}_th"] = getattr(torch, f"{type_}{size}")
        globals()[f"{type_}_np"] = getattr(np, f"{type_}{size}")
        torch.set_default_dtype(getattr(torch, f"float{size}"))
    else:
        raise ValueError("Invalid dtype size")
    if type_ == "float" and size == 16 and not torch.cuda.is_available():
        raise Exception(
            "torch.float16 is not supported on CPU because addmm_impl_cpu_ is not implemented"
            " for this floating precision. Please use size = 32, 64 or run with 'cuda' instead."
        )


def clear_cache(confirm: bool = True) -> None:
    """Delete all files in the cache, e.g. to clean out downloaded models.

    Args:
        confirm: Whether to ask for confirmation.
    """
    answer = "" if confirm else "y"
    while answer not in ("y", "n"):
        answer = input(f"Do you really want to delete everything in {DIEP_CACHE} (y|n)? ").lower().strip()
    if answer == "y":
        try:
            shutil.rmtree(DIEP_CACHE)
        except FileNotFoundError:
            print(f"cache dir {DIEP_CACHE!r} not found")
