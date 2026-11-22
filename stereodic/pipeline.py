"""Stage orchestration: calibrate -> stereo -> temporal -> reconstruct -> analyze (docs §12.2).

Scripts only parse arguments and call these functions. Each stage records its config
hash in the HDF5 `progress/` group; a finished stage with the same hash is skipped.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .core.config import ExperimentConfig, config_hash

log = logging.getLogger(__name__)


# ------------------------------------------------------------------- calibration
def run_calibration(cfg: ExperimentConfig, plot: bool = True):
    """Detect ChArUco corners in the calibration images, calibrate, write `calibration_file`."""
    from .calibration.calibrate import CalibrationOptions, calibrate_stereo
    from .calibration.detect import CharucoBoardSpec, detect_charuco
    from .calibration.report import plot_residuals, save_calibration
    from .io.sequence import open_sequence

    cc = cfg.calibration
    if cc is None:
        raise ValueError("experiment config has no `calibration` section")
    if cfg.calibration_file is None:
        raise ValueError("experiment config has no `calibration_file`")
    spec = CharucoBoardSpec(**cc.board.model_dump())
    seq_l = open_sequence(cfg.data_path(cc.left.path), cc.left.pattern)
    seq_r = open_sequence(cfg.data_path(cc.right.path), cc.right.pattern)
    if len(seq_l) != len(seq_r):
        raise ValueError(f"calibration views: {len(seq_l)} left vs {len(seq_r)} right images")
    origin = np.asarray(cc.roi_origin, dtype=np.float64)
    h, w = seq_l.frame_shape
    sensor = tuple(cc.sensor_size) if cc.sensor_size else (w + int(origin[0]), h + int(origin[1]))

    dets = {"left": [], "right": []}
    names = []
    for i in range(len(seq_l)):
        names.append((seq_l.frame_name(i), seq_r.frame_name(i)))
        for side, seq in (("left", seq_l), ("right", seq_r)):
            d = detect_charuco(seq.read(i), spec, min_corners=cc.min_corners)
            if d is not None:
                d.image_points = d.image_points + origin
            dets[side].append(d)
            log.info("%s %s: %s corners", side, seq.frame_name(i), 0 if d is None else len(d))
    opts = CalibrationOptions(
        distortion=cc.distortion,
        refine_stereo_intrinsics=cc.refine_stereo_intrinsics,
        outlier_factor=cc.outlier_factor,
        min_views=cc.min_views,
        min_common_corners=cc.min_corners,
    )
    result = calibrate_stereo(dets["left"], dets["right"], sensor, sensor, opts)
    out = cfg.data_path(cfg.calibration_file)
    save_calibration(
        out,
        result,
        extra={
            "views": [list(n) for n in names],
            "board": cc.board.model_dump(),
            "options": cc.model_dump(mode="json"),
            "config_hash": config_hash(cc),
        },
    )
    if plot:
        plot_residuals(result, Path(out).with_suffix(".png"))
    return result


# ------------------------------------------------------------------------ stereo
def _file_sha256(path: Path) -> str:
    import hashlib

    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def _read_reference(cfg: ExperimentConfig):
    from .io.sequence import open_sequence

    seq_l = open_sequence(cfg.data_path(cfg.cameras.left.path), cfg.cameras.left.pattern)
    seq_r = open_sequence(cfg.data_path(cfg.cameras.right.path), cfg.cameras.right.pattern)
    i = cfg.reference_frame
    if i >= len(seq_l) or i >= len(seq_r):
        raise ValueError(f"reference_frame {i} out of range ({len(seq_l)} / {len(seq_r)} frames)")
    return seq_l.read(i), seq_r.read(i), seq_l.frame_name(i), seq_r.frame_name(i)


def _poi_mask(cfg: ExperimentConfig, shape: tuple[int, int]) -> np.ndarray | None:
    if cfg.poi_mask is None:
        return None
    import tifffile
    import cv2

    p = cfg.data_path(cfg.poi_mask)
    m = tifffile.imread(p) if p.suffix.lower() in (".tif", ".tiff") else cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise ValueError(f"cannot read poi_mask {p}")
    m = np.asarray(m)
    if m.ndim == 3:
        m = m.max(axis=2)
    if m.shape != shape:
        raise ValueError(f"poi_mask shape {m.shape} differs from the image shape {shape}")
    return m > 0


def run_stereo(cfg: ExperimentConfig, force: bool = False) -> Path:
    """Reference stage: POI grid on L0, stereo matching L0 -> R0, 3D reference shape.

    Writes `reference/*` and `calibration/*` into the experiment output file.
    """
    from .calibration.report import load_calibration
    from .core.types import ROI
    from .dic.mesh import make_poi_grid
    from .dic.stereo_match import options_from_config, stereo_match
    from .io.storage import ResultStore
    from .recon.shape import reference_shape

    if cfg.output_file is None or cfg.calibration_file is None:
        raise ValueError("experiment config needs output_file and calibration_file")
    out = cfg.results_path(cfg.output_file)
    stage_hash = config_hash(
        {
            "dic": cfg.dic.model_dump(mode="json"),
            "recon": cfg.recon.model_dump(mode="json"),
            "roi": cfg.roi,
            "poi_mask": str(cfg.poi_mask),
            "reference_frame": cfg.reference_frame,
            "cameras": cfg.cameras.model_dump(mode="json"),
            "calibration_sha256": _file_sha256(cfg.data_path(cfg.calibration_file)),
        }
    )
    if out.exists() and not force:
        with ResultStore.open(out, "r") as store:
            if store.stage_done("stereo", stage_hash):
                log.info("stereo stage up to date: %s", out)
                return out

    calib = load_calibration(cfg.data_path(cfg.calibration_file))
    left, right, name_l, name_r = _read_reference(cfg)
    origin_l = np.asarray(cfg.cameras.left.sensor_origin, dtype=np.float64)
    origin_r = np.asarray(cfg.cameras.right.sensor_origin, dtype=np.float64)
    roi = ROI.from_list(cfg.roi)
    mask = _poi_mask(cfg, left.shape)
    pois = make_poi_grid(left.shape, cfg.dic.poi.subset_size, cfg.dic.poi.step, roi=roi, mask=mask)
    log.info("%d POIs", len(pois))

    sm = stereo_match(
        left, right, pois, calib, options_from_config(cfg.dic), cfg.dic.backend, mask, (origin_l, origin_r)
    )
    shape = reference_shape(
        calib,
        pois.xy + origin_l,
        sm.xy_right + origin_r,
        sm.result.status,
        cfg.dic.stereo.max_reprojection_px,
        np.asarray(cfg.recon.span_direction),
        cfg.recon.normal_radius_steps,
    )

    manifest = {"reference_left": name_l, "reference_right": name_r}
    with ResultStore.create(out, config_snapshot=cfg.snapshot(), input_manifest=manifest, overwrite=True) as store:
        store.write_calibration(calib)
        g = "reference/"
        store.write_array(g + "frame_index", np.array(cfg.reference_frame))
        store.write_array(g + "point_ids", pois.point_ids)
        store.write_array(g + "points_left", pois.xy + origin_l, frame="full sensor")
        store.write_array(g + "points_right", sm.xy_right + origin_r, frame="full sensor")
        store.write_array(g + "points_left_image", pois.xy, frame="left image")
        store.write_array(g + "points_right_image", sm.xy_right, frame="right image")
        store.write_array(g + "points_3d", shape.points_3d, units="mm")
        store.write_array(g + "normals", shape.normals)
        store.write_array(g + "local_frames", shape.local_frames, rows="c, s, n")
        store.write_array(g + "connectivity", shape.connectivity)
        store.write_array(g + "stereo_znssd", sm.result.znssd.astype(np.float32))
        store.write_array(g + "stereo_status", shape.status)
        store.write_array(g + "stereo_params", sm.result.params, shape_order=cfg.dic.stereo.shape_order)
        store.write_array(g + "reprojection_error", shape.reprojection_error.astype(np.float32), units="px")
        store.mark_stage_done("stereo", stage_hash)
    return out
