"""Import-step helpers (U1): readers, frame offset, probe, image quality, sync summary."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import tifffile

from stereodic.io.probe import numbering_gaps, probe_sequence
from stereodic.io.readers import readers
from stereodic.io.readers.imagedir import ImageDirSequence
from stereodic.io.sequence import OffsetSequence, open_sequence
from stereodic.io.sync import sync_summary, timing_summary
from stereodic.quality.image import effective_bits, frame_report, sharpness, speckle_size, trailing_zero_bits
from stereodic.synth import SensorModel, SpeckleTexture, render_frame


def _frames(d: Path, n: int, shift: int = 0, names=None, ext=".tif"):
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    out = []
    for i in range(n):
        img = (rng.integers(0, 4096, (24, 32)).astype(np.uint16) << shift).astype(np.uint16)
        name = names[i] if names else f"img{i:04d}{ext}"
        if ext == ".tif":
            tifffile.imwrite(d / name, img)
        else:
            cv2.imwrite(str(d / name), img)
        out.append(img)
    return out


def test_registry_png_and_offset(tmp_path: Path):
    assert [r.name for r in readers()][:2] == ["tiff", "imagedir"]
    frames = _frames(tmp_path / "png", 6, ext=".png")
    seq = open_sequence(tmp_path / "png", "*.png")
    assert isinstance(seq, ImageDirSequence) and seq.dtype == np.uint16
    np.testing.assert_array_equal(seq.read(3), frames[3])
    off = open_sequence(tmp_path / "png", "*.png", frame_offset=2)
    assert isinstance(off, OffsetSequence) and len(off) == 4
    np.testing.assert_array_equal(off.read(0), frames[2])
    assert off.frame_name(-1) == "img0005.png" and off.raw_index(1) == 3
    with pytest.raises(ValueError):
        open_sequence(tmp_path / "png", "*.png", frame_offset=6)
    (tmp_path / "jpg").mkdir()
    cv2.imwrite(str(tmp_path / "jpg" / "a.jpg"), np.zeros((8, 8), np.uint8))
    with pytest.raises(ValueError, match="unsupported"):
        open_sequence(tmp_path / "jpg", "*.jpg")


def test_probe_bit_depth_gaps_and_timestamps(tmp_path: Path):
    names = ["f_001.tif", "f_002.tif", "f_005.tif", "f_006.tif"]
    _frames(tmp_path / "cam", 4, shift=4, names=names)
    (tmp_path / "cam" / "timestamps.csv").write_text("0\n1\n4\n5\n")
    info = probe_sequence(tmp_path / "cam", "*.tif")
    assert info["reader"] == "tiff" and info["n_frames"] == 4 and info["shape"] == [24, 32]
    assert info["container_bits"] == 16 and info["msb_aligned_shift"] == 4 and info["data_bits"] == 12
    assert info["numbering"]["n_missing"] == 2 and info["numbering"]["gaps"] == [{"after": 2, "missing": 2}]
    assert any("MSB-aligned" in w for w in info["warnings"]) and any("missing" in w for w in info["warnings"])
    assert info["timestamp_candidates"][0].endswith("timestamps.csv")
    assert numbering_gaps(["a.tif", "b.tif"])["numbered"] is False


def test_image_quality_measures():
    assert effective_bits(4095) == 12 and effective_bits(4096) == 13
    assert trailing_zero_bits([np.array([16, 32, 48], np.uint16)]) == 4
    sizes, sharp = [], []
    for d in (3, 4, 6):
        tex = SpeckleTexture.random((-20, -20, 300, 300), diameter_px=d, seed=1)
        img = render_frame(tex, (256, 256), None, SensorModel(bits=12, noise_std=4), oversample=3, seed=0)
        sizes.append(speckle_size(img))
        sharp.append(sharpness(img))
        rep = frame_report(img, 4095)
        assert rep["saturation_fraction"] < 0.05 and sum(rep["histogram"]["counts"]) == img.size
    np.testing.assert_allclose(sizes, [3, 4, 6], rtol=0.2)  # calibrated estimator, +-20 %
    assert sharp[0] > sharp[1] > sharp[2]  # larger speckles / blur: lower gradient energy
    blurred = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 2.0)
    assert sharpness(blurred) < 0.6 * sharpness(img)


def test_sync_summary_detects_drops_and_offsets():
    fs = 1000.0
    t = np.arange(50) / fs
    ok = sync_summary(t, t + 2e-6, nominal_rate_hz=fs)
    assert ok["index_aligned"] and ok["n_pairs"] == 50 and ok["delta_us"]["max"] == pytest.approx(2.0)
    dropped = np.delete(t, 20)
    rep = sync_summary(t, dropped, max_delta_seconds=1e-4, nominal_rate_hz=fs)
    assert not rep["index_aligned"] and rep["n_unpaired_left"] == 1 and rep["unpaired_left"] == [20]
    tim = timing_summary(dropped, fs)
    assert tim["n_gaps"] == 1 and tim["gaps"][0]["missing"] == 1 and tim["rate_hz"] == pytest.approx(fs)
