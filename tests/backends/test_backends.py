from __future__ import annotations

import numpy as np
import pytest

from stereodic.dic.backends import available_backends, cuda_info, resolve_backend


def test_resolve_backend():
    avail = available_backends()
    assert "numpy" in avail and "numba" in avail
    assert resolve_backend("auto") in avail
    assert resolve_backend("numpy") == "numpy"
    with pytest.raises(ValueError):
        resolve_backend("mps")
    if "cuda" not in avail:
        with pytest.raises(RuntimeError):
            resolve_backend("cuda")


@pytest.mark.cuda
def test_cuda_float64_and_spline_filter():
    import cupy as cp
    from cupyx.scipy.ndimage import spline_filter
    from scipy.ndimage import spline_filter as spline_filter_cpu

    info = cuda_info()
    assert info["available"], info
    img = np.random.default_rng(0).random((64, 64))
    gpu = cp.asnumpy(spline_filter(cp.asarray(img), order=3, mode="mirror"))
    cpu = spline_filter_cpu(img, order=3, mode="mirror")
    assert gpu.dtype == np.float64
    np.testing.assert_allclose(gpu, cpu, rtol=0, atol=1e-10)
