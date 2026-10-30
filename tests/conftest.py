import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """Keep API/LLM caches out of ~/.cache, but reuse already-downloaded face models."""
    from sixdegrees import util

    real_models = util.cache_dir() / "models"
    cache = tmp_path / "cache"
    (cache).mkdir()
    if real_models.exists():
        (cache / "models").symlink_to(real_models, target_is_directory=True)
    monkeypatch.setenv("SIXDEGREES_CACHE", str(cache))
    return cache
