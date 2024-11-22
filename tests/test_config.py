from __future__ import annotations

import os.path

from diep.config import DIEP_CACHE, clear_cache


def test_clear_cache():
    clear_cache(False)
    assert not os.path.exists(DIEP_CACHE)
