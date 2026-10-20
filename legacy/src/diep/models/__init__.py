"""Package containing model implementations.

Imports are lazy (PEP 562) so that importing :mod:`diep.models` does not require DGL; the
DGL-backed :class:`DIEP` model still raises a clear ImportError if DGL is missing and you
ask for it. The PyG equivalent lives in :mod:`diep.pyg.models`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

_MODULE_BY_NAME = {
    "MatGLModel": "diep.models._core",
    "DIEP": "diep.models._diep",
    "TransformedTargetModel": "diep.models._wrappers",
}

__all__ = sorted(_MODULE_BY_NAME)


def __getattr__(name: str):
    try:
        module_name = _MODULE_BY_NAME[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    import importlib

    return getattr(importlib.import_module(module_name), name)


def __dir__():
    return __all__


if TYPE_CHECKING:
    from diep.models._core import MatGLModel
    from diep.models._diep import DIEP
    from diep.models._wrappers import TransformedTargetModel
