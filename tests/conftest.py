from __future__ import annotations

import pytest

from stereodic.dic.backends import cuda_info


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    info = cuda_info()
    if info["available"]:
        return
    skip = pytest.mark.skip(reason=f"CUDA unavailable: {info.get('reason')}")
    for item in items:
        if "cuda" in item.keywords:
            item.add_marker(skip)
