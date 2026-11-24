"""Stage orchestration: calibrate -> stereo -> temporal -> reconstruct -> analyze (docs §12.2).

Scripts only parse arguments and call these functions. Each stage records its config
hash in the HDF5 `progress/` group; a finished stage with the same hash is skipped.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .core.config import ExperimentConfig, config_hash
from .core.progress import JobCancelled, ProgressReporter, ensure_progress
from .dic import shape_functions as sf

log = logging.getLogger(__name__)

STAGES = ("calibrate", "stereo", "temporal", "reconstruct", "rbm", "analyze", "uncertainty")


# ------------------------------------------------------------------- calibration
def run_calibration(cfg: ExperimentConfig, plot: bool = True, progress: ProgressReporter | None = None):
    """Detect ChArUco corners in the calibration images, calibrate, write `calibration_file`."""
    from .calibration.calibrate import CalibrationOptions, calibrate_stereo
    from .calibration.detect import CharucoBoardSpec, detect_charuco
    from .calibration.report import plot_residuals, save_calibration
    from .io.sequence import open_sequence

    progress = ensure_progress(progress)
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
    origin_l = np.asarray(
        cc.left.sensor_origin if cc.left.sensor_origin != [0, 0] else cc.roi_origin, dtype=np.float64
    )
    origin_r = np.asarray(
        cc.right.sensor_origin if cc.right.sensor_origin != [0, 0] else cc.roi_origin, dtype=np.float64
    )
    hl, wl = seq_l.frame_shape
    hr, wr = seq_r.frame_shape
    sensor_l = tuple(cc.sensor_size) if cc.sensor_size else (wl + int(origin_l[0]), hl + int(origin_l[1]))
    sensor_r = tuple(cc.sensor_size) if cc.sensor_size else (wr + int(origin_r[0]), hr + int(origin_r[1]))

    dets = {"left": [], "right": []}
    names = []
    progress.phase("detecting")
    for i in range(len(seq_l)):
        progress.check_cancel()
        progress.update(i, len(seq_l), f"view {seq_l.frame_name(i)}")
        names.append((seq_l.frame_name(i), seq_r.frame_name(i)))
        for side, seq, origin in (("left", seq_l, origin_l), ("right", seq_r, origin_r)):
            d = detect_charuco(seq.read(i), spec, min_corners=cc.min_corners)
            if d is not None:
                d.image_points = d.image_points + origin
            dets[side].append(d)
            log.info("%s %s: %s corners", side, seq.frame_name(i), 0 if d is None else len(d))
    progress.update(len(seq_l), len(seq_l))
    progress.check_cancel()
    progress.phase("calibrating")
    opts = CalibrationOptions(
        distortion=cc.distortion,
        refine_stereo_intrinsics=cc.refine_stereo_intrinsics,
        outlier_factor=cc.outlier_factor,
        min_views=cc.min_views,
        min_common_corners=cc.min_corners,
    )
    result = calibrate_stereo(dets["left"], dets["right"], sensor_l, sensor_r, opts)
    progress.phase("writing")
    out = cfg.data_path(cfg.calibration_file)
    save_calibration(
        out,
        result,
        roi_origin=(int(origin_l[0]), int(origin_l[1])),
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


def _read_mask_image(path: Path, shape: tuple[int, int] | None, label: str) -> np.ndarray:
    import cv2
    import tifffile

    m = tifffile.imread(path) if path.suffix.lower() in (".tif", ".tiff") else cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise ValueError(f"cannot read {label} {path}")
    m = np.asarray(m)
    if m.ndim == 3:
        m = m.max(axis=2)
    if shape is not None and m.shape != shape:
        raise ValueError(f"{label} shape {m.shape} differs from the image shape {shape}")
    return m > 0


def _poi_mask(cfg: ExperimentConfig, shape: tuple[int, int]) -> np.ndarray | None:
    if cfg.poi_mask is None:
        return None
    return _read_mask_image(cfg.data_path(cfg.poi_mask), shape, "poi_mask")


def _stereo_stage_hash(cfg: ExperimentConfig) -> str:
    if cfg.calibration_file is None:
        raise ValueError("experiment config needs calibration_file")
    return config_hash(
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


def run_stereo(cfg: ExperimentConfig, force: bool = False, progress: ProgressReporter | None = None) -> Path:
    """Reference stage: POI grid on L0, stereo matching L0 -> R0, 3D reference shape.

    Writes `reference/*` and `calibration/*` into the experiment output file.
    """
    from .calibration.report import load_calibration
    from .core.types import ROI
    from .dic.mesh import make_poi_grid
    from .dic.stereo_match import options_from_config, stereo_match
    from .io.storage import ResultStore
    from .recon.shape import reference_shape

    progress = ensure_progress(progress)
    if cfg.output_file is None or cfg.calibration_file is None:
        raise ValueError("experiment config needs output_file and calibration_file")
    out = cfg.results_path(cfg.output_file)
    stage_hash = _stereo_stage_hash(cfg)
    if out.exists() and not force:
        with ResultStore.open(out, "r") as store:
            if store.stage_done("stereo", stage_hash):
                log.info("stereo stage up to date: %s", out)
                return out

    progress.phase("loading")
    calib = load_calibration(cfg.data_path(cfg.calibration_file))
    left, right, name_l, name_r = _read_reference(cfg)
    origin_l = np.asarray(cfg.cameras.left.sensor_origin, dtype=np.float64)
    origin_r = np.asarray(cfg.cameras.right.sensor_origin, dtype=np.float64)
    roi = ROI.from_list(cfg.roi)
    mask = _poi_mask(cfg, left.shape)
    pois = make_poi_grid(left.shape, cfg.dic.poi.subset_size, cfg.dic.poi.step, roi=roi, mask=mask)
    log.info("%d POIs", len(pois))
    progress.check_cancel()

    progress.phase("matching")
    sm = stereo_match(
        left, right, pois, calib, options_from_config(cfg.dic), cfg.dic.backend, mask, (origin_l, origin_r)
    )
    progress.check_cancel()
    progress.phase("reconstructing")
    shape = reference_shape(
        calib,
        pois.xy + origin_l,
        sm.xy_right + origin_r,
        sm.result.status,
        cfg.dic.stereo.max_reprojection_px,
        np.asarray(cfg.recon.span_direction),
        cfg.recon.normal_radius_steps,
    )

    progress.check_cancel()
    progress.phase("writing")
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


# ---------------------------------------------------------------------- temporal
def _selected_frame_indices(cfg: ExperimentConfig, n_left: int, n_right: int) -> np.ndarray:
    from .io.sequence import frame_indices

    if n_left != n_right:
        raise ValueError(f"left/right sequence lengths differ: {n_left} / {n_right}")
    selected = list(frame_indices(n_left, cfg.frames.start, cfg.frames.stop, cfg.frames.step))
    if cfg.reference_frame >= n_left:
        raise ValueError(f"reference_frame {cfg.reference_frame} out of range ({n_left} frames)")
    if cfg.reference_frame not in selected:
        selected.append(cfg.reference_frame)
        selected.sort()
    return np.asarray(selected, dtype=np.int64)


def _temporal_stage_hash(
    cfg: ExperimentConfig,
    frame_indices: np.ndarray,
    calibration_path: Path,
    stereo_hash: str = "",
) -> str:
    return config_hash(
        {
            "stereo_hash": stereo_hash,
            "dic_poi": cfg.dic.poi.model_dump(mode="json"),
            "dic_temporal": cfg.dic.temporal.model_dump(mode="json"),
            "solver": cfg.dic.solver.model_dump(mode="json"),
            "interpolation": cfg.dic.interpolation.model_dump(mode="json"),
            "frames": frame_indices.tolist(),
            "reference_frame": cfg.reference_frame,
            "sensor_origins": [cfg.cameras.left.sensor_origin, cfg.cameras.right.sensor_origin],
            "calibration_sha256": _file_sha256(calibration_path),
        }
    )


def _ensure_timeseries(
    store,
    name: str,
    n_frames: int,
    n_points: int,
    n_components: int | None = None,
    *,
    dtype: np.dtype | type = np.float64,
    fill_value=None,
    reset: bool = False,
):
    expected_shape = (n_frames, n_points) if n_components is None else (n_frames, n_points, n_components)
    if name in store.h5:
        existing = store.h5[name]
        if reset or existing.shape != expected_shape or existing.dtype != np.dtype(dtype):
            del store.h5[name]
    if name not in store.h5:
        return store.create_timeseries(name, n_frames, n_points, n_components, dtype=dtype, fill_value=fill_value)
    return store.h5[name]


def _ensure_temporal_datasets(
    store,
    n_frames: int,
    n_points: int,
    n_params: int,
    shape_order: int,
    save_params: bool,
    *,
    reset: bool = False,
) -> None:
    for side in ("left", "right"):
        prefix = f"dic/{side}/"
        ds_xy = _ensure_timeseries(store, f"{prefix}xy", n_frames, n_points, 2, reset=reset)
        ds_xy.attrs["shape_order"] = shape_order
        _ensure_timeseries(store, f"{prefix}znssd", n_frames, n_points, dtype=np.float32, reset=reset)
        _ensure_timeseries(store, f"{prefix}iterations", n_frames, n_points, dtype=np.uint8, fill_value=0, reset=reset)
        _ensure_timeseries(store, f"{prefix}status", n_frames, n_points, dtype=np.uint8, fill_value=0, reset=reset)
        if save_params:
            ds_p = _ensure_timeseries(store, f"{prefix}params", n_frames, n_points, n_params, reset=reset)
            ds_p.attrs["shape_order"] = shape_order
        elif reset and f"{prefix}params" in store.h5:
            del store.h5[f"{prefix}params"]
    done_name = "progress/dic_frames_done"
    if done_name in store.h5 and (reset or store.h5[done_name].shape != (n_frames, 2)):
        del store.h5[done_name]
    if done_name not in store.h5:
        store.h5.create_dataset(
            done_name,
            shape=(n_frames, 2),
            dtype=np.bool_,
            chunks=(min(64, n_frames), 2),
            fillvalue=False,
        )


def _nanmedian_or_none(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else None


def _update_previous_from_store(
    store,
    side: str,
    row: int,
    centers: np.ndarray,
    previous_params: np.ndarray,
    shape_order: int,
) -> np.ndarray:
    """Update carried-over initial guess for points that succeeded on a resumed row."""
    from .core.status import Status

    st = store.h5[f"dic/{side}/status"][row]
    good = st == Status.OK
    if not np.any(good):
        return previous_params
    out = previous_params.copy()
    path = f"dic/{side}/params"
    if path in store.h5:
        values = store.h5[path][row]
        ok = good & np.all(np.isfinite(values), axis=1)
        out[ok] = values[ok]
        return out
    xy = store.h5[f"dic/{side}/xy"][row]
    ok = good & np.all(np.isfinite(xy), axis=1)
    if np.any(ok):
        out[ok] = sf.from_displacement(xy[ok] - centers[ok], shape_order)
    return out


def run_temporal(cfg: ExperimentConfig, force: bool = False, progress: ProgressReporter | None = None) -> Path:
    """Run fixed-reference DIC for both cameras with frame-level checkpoints.

    Cancellation (``progress.check_cancel``) is honoured between frames, after the
    pending rows are flushed and marked done, so the run can be resumed.
    """
    from .core.status import Status
    from .core.types import ROI
    from .dic import shape_functions as sf
    from .dic.temporal import TemporalMatchOptions, TemporalTracker
    from .io.sequence import open_sequence
    from .io.sync import pair_timestamps, sequence_timestamps
    from .io.storage import ResultStore

    progress = ensure_progress(progress)
    if cfg.output_file is None or cfg.calibration_file is None:
        raise ValueError("experiment config needs output_file and calibration_file")
    out = cfg.results_path(cfg.output_file)
    if not out.exists():
        raise FileNotFoundError(f"stereo output does not exist: {out}; run --stage stereo first")

    progress.phase("preparing")
    seq_l = open_sequence(cfg.data_path(cfg.cameras.left.path), cfg.cameras.left.pattern)
    seq_r = open_sequence(cfg.data_path(cfg.cameras.right.path), cfg.cameras.right.pattern)
    frame_ids = _selected_frame_indices(cfg, len(seq_l), len(seq_r))
    options = TemporalMatchOptions.from_config(cfg.dic)
    n_params = sf.n_params(options.icgn.shape_order)
    backend = cfg.dic.backend

    with ResultStore.open(out, "r+") as store:
        prog = store.h5.require_group("progress")
        stereo_hash = str(prog.attrs.get("stereo_config_hash", ""))
        stage_hash = _temporal_stage_hash(cfg, frame_ids, cfg.data_path(cfg.calibration_file), stereo_hash)
        if not force and store.stage_done("temporal", stage_hash):
            return out
        centers_l = store.read_array("reference/points_left_image")
        centers_r = store.read_array("reference/points_right_image")
        if centers_l.shape != centers_r.shape:
            raise ValueError("left/right reference point arrays have different shapes")
        n_points = centers_l.shape[0]
        same_run = (
            (not force)
            and (prog.attrs.get("temporal_run_hash") == stage_hash)
            and ("dic/left/xy" in store.h5)
            and (store.h5["dic/left/xy"].shape == (len(frame_ids), n_points, 2))
            and ("progress/dic_frames_done" in store.h5)
            and (store.h5["progress/dic_frames_done"].shape == (len(frame_ids), 2))
        )
        reset = not same_run
        _ensure_temporal_datasets(
            store,
            len(frame_ids),
            n_points,
            n_params,
            options.icgn.shape_order,
            cfg.dic.storage.save_params,
            reset=reset,
        )
        if reset:
            store.invalidate_stage("temporal")
            store.invalidate_stage("reconstruct")
            store.invalidate_stage("rbm")
            prog.attrs["temporal_run_hash"] = stage_hash
        store.write_array("time/frame_ids", np.column_stack([frame_ids, frame_ids]))

        ts_l = sequence_timestamps(
            len(seq_l), cfg.cameras.frame_rate_hz,
            cfg.data_path(cfg.cameras.left.timestamp_path) if cfg.cameras.left.timestamp_path else None,
        )
        ts_r = sequence_timestamps(
            len(seq_r), cfg.cameras.frame_rate_hz,
            cfg.data_path(cfg.cameras.right.timestamp_path) if cfg.cameras.right.timestamp_path else None,
        )
        max_delta_s = (
            None if cfg.cameras.max_sync_delta_us is None else float(cfg.cameras.max_sync_delta_us) * 1e-6
        )
        sync = pair_timestamps(ts_l, ts_r, max_delta_seconds=max_delta_s)
        if not (
            np.array_equal(sync.left_indices, np.arange(len(seq_l)))
            and np.array_equal(sync.right_indices, np.arange(len(seq_r)))
        ):
            raise ValueError("left/right timestamp pairing is incomplete; inspect with scripts/check_sync.py")
        store.write_array("time/timestamps_left", ts_l[frame_ids], units="s")
        store.write_array("time/timestamps_right", ts_r[frame_ids], units="s")
        store.write_array("time/timestamps", 0.5 * (ts_l[frame_ids] + ts_r[frame_ids]), units="s")
        store.write_array("time/sync_delta", sync.delta_seconds[frame_ids], units="s")

        done = store.h5["progress/dic_frames_done"]

        ref_index = int(cfg.reference_frame)
        ref_l = seq_l.read(ref_index, ROI.from_list(None))
        ref_r = seq_r.read(ref_index, ROI.from_list(None))
        tracker_l = TemporalTracker(ref_l, centers_l, options, backend)
        tracker_r = TemporalTracker(ref_r, centers_r, options, backend)
        previous_xy_l = None
        previous_xy_r = None
        previous_params_l = np.zeros((n_points, n_params), dtype=np.float64)
        previous_params_r = np.zeros((n_points, n_params), dtype=np.float64)

        batch_size = max(1, int(cfg.dic.temporal.frame_batch))
        pending_rows: list[int] = []

        def _flush_pending() -> None:
            if not pending_rows:
                return
            store.flush()
            for r in pending_rows:
                done[r] = [True, True]
            store.flush()
            pending_rows.clear()

        n_rows = len(frame_ids)
        progress.phase("tracking")
        for row, frame_id in enumerate(frame_ids.tolist()):
            try:
                progress.check_cancel()
            except JobCancelled:
                _flush_pending()
                raise
            if bool(done[row, 0]) and bool(done[row, 1]):
                previous_xy_l = store.h5["dic/left/xy"][row]
                previous_xy_r = store.h5["dic/right/xy"][row]
                previous_params_l = _update_previous_from_store(
                    store, "left", row, centers_l, previous_params_l, options.icgn.shape_order
                )
                previous_params_r = _update_previous_from_store(
                    store, "right", row, centers_r, previous_params_r, options.icgn.shape_order
                )
                progress.update(row + 1, n_rows)
                continue

            image_l = seq_l.read(frame_id)
            image_r = seq_r.read(frame_id)
            if frame_id == ref_index:
                result_l = tracker_l.reference_result(frame_id)
                result_r = tracker_r.reference_result(frame_id)
            else:
                p0_l = previous_params_l if options.initialization == "previous_solution" else None
                p0_r = previous_params_r if options.initialization == "previous_solution" else None
                result_l = tracker_l.solve_frame(image_l, p0_l)
                result_r = tracker_r.solve_frame(image_r, p0_r)
                tracker_l.check_jump(result_l, previous_xy_l)
                tracker_r.check_jump(result_r, previous_xy_r)

            for side, result in (("left", result_l), ("right", result_r)):
                store.h5[f"dic/{side}/xy"][row] = result.xy
                store.h5[f"dic/{side}/znssd"][row] = result.znssd
                store.h5[f"dic/{side}/iterations"][row] = result.iterations
                store.h5[f"dic/{side}/status"][row] = result.status
                if cfg.dic.storage.save_params:
                    store.h5[f"dic/{side}/params"][row] = result.params
            pending_rows.append(row)
            if len(pending_rows) >= batch_size:
                _flush_pending()
            previous_xy_l = result_l.xy.copy()
            previous_xy_r = result_r.xy.copy()
            good_l = result_l.status == Status.OK
            if np.any(good_l):
                previous_params_l[good_l] = result_l.params[good_l]
            good_r = result_r.status == Status.OK
            if np.any(good_r):
                previous_params_r[good_r] = result_r.params[good_r]
            progress.update(row + 1, n_rows, f"frame {frame_id}")
            progress.summary(
                {
                    "row": row,
                    "frame": int(frame_id),
                    "valid_left": float(good_l.mean()) if n_points else 0.0,
                    "valid_right": float(good_r.mean()) if n_points else 0.0,
                    "znssd_median": _nanmedian_or_none(
                        np.concatenate([result_l.znssd[good_l], result_r.znssd[good_r]])
                    ),
                }
            )

        _flush_pending()
        if bool(np.all(done[...])):
            store.mark_stage_done("temporal", stage_hash)
    return out


# ------------------------------------------------------------------- reconstruct
def _reconstruct_stage_hash(cfg: ExperimentConfig, progress_attrs, temporal_shape: tuple[int, ...]) -> str:
    return config_hash(
        {
            "stereo_hash": str(progress_attrs.get("stereo_config_hash", "")),
            "temporal_hash": str(progress_attrs.get("temporal_config_hash", "")),
            "stereo": cfg.dic.stereo.model_dump(mode="json"),
            "recon": cfg.recon.model_dump(mode="json"),
            "sensor_origins": [cfg.cameras.left.sensor_origin, cfg.cameras.right.sensor_origin],
            "calibration_sha256": _file_sha256(cfg.data_path(cfg.calibration_file)),
            "temporal_shape": list(temporal_shape),
        }
    )


def run_reconstruct(cfg: ExperimentConfig, force: bool = False, progress: ProgressReporter | None = None) -> Path:
    """Triangulate every valid temporal correspondence and write 3D displacement."""
    from .calibration.report import load_calibration
    from .core.status import Status
    from .io.storage import ResultStore
    from .recon.frames import to_local
    from .recon.triangulate import triangulate

    progress = ensure_progress(progress)
    if cfg.output_file is None or cfg.calibration_file is None:
        raise ValueError("experiment config needs output_file and calibration_file")
    out = cfg.results_path(cfg.output_file)
    calib = load_calibration(cfg.data_path(cfg.calibration_file))
    origin_l = np.asarray(cfg.cameras.left.sensor_origin, dtype=np.float64)
    origin_r = np.asarray(cfg.cameras.right.sensor_origin, dtype=np.float64)
    with ResultStore.open(out, "r+") as store:
        if "dic/left/xy" not in store.h5 or "dic/right/xy" not in store.h5:
            raise ValueError("temporal DIC datasets are missing; run --stage temporal first")
        xy_l = store.h5["dic/left/xy"]
        xy_r = store.h5["dic/right/xy"]
        T, N, _ = xy_l.shape
        prog = store.h5.require_group("progress")
        stage_hash = _reconstruct_stage_hash(cfg, prog.attrs, xy_l.shape)
        if not force and store.stage_done("reconstruct", stage_hash):
            return out
        same_run = (
            (not force)
            and (prog.attrs.get("reconstruct_run_hash") == stage_hash)
            and ("reconstruction/xyz" in store.h5)
            and (store.h5["reconstruction/xyz"].shape == (T, N, 3))
            and ("progress/reconstruction_done" in store.h5)
            and (store.h5["progress/reconstruction_done"].shape == (T,))
        )
        reset = not same_run
        for name, components, dtype in (
            ("reconstruction/xyz", 3, np.float64),
            ("reconstruction/displacement_raw", 3, np.float64),
            ("reconstruction/displacement_local", 3, np.float64),
            ("reconstruction/reprojection_error", None, np.float32),
            ("reconstruction/status", None, np.uint8),
            ("quality/valid_mask", None, np.bool_),
        ):
            _ensure_timeseries(
                store,
                name,
                T,
                N,
                components,
                dtype=dtype,
                fill_value=False if dtype is np.bool_ else None,
                reset=reset,
            )

        X0 = store.read_array("reference/points_3d")
        frames = store.read_array("reference/local_frames")
        done_name = "progress/reconstruction_done"
        if done_name in store.h5 and (reset or store.h5[done_name].shape != (T,)):
            del store.h5[done_name]
        if done_name not in store.h5:
            store.h5.create_dataset(done_name, shape=(T,), dtype=np.bool_, fillvalue=False)
        done = store.h5[done_name]
        if reset:
            store.invalidate_stage("reconstruct")
            store.invalidate_stage("rbm")
            prog.attrs["reconstruct_run_hash"] = stage_hash

        progress.phase("triangulating")
        for k in range(T):
            progress.check_cancel()  # every finished row is already flushed and marked done
            if bool(done[k]):
                progress.update(k + 1, T)
                continue
            left = xy_l[k]
            right = xy_r[k]
            dic_ok = np.all(np.isfinite(left), axis=1) & np.all(np.isfinite(right), axis=1)
            tri = triangulate(
                calib,
                np.where(dic_ok[:, None], left + origin_l, np.nan),
                np.where(dic_ok[:, None], right + origin_r, np.nan),
                max_reprojection_px=cfg.dic.stereo.max_reprojection_px,
            )
            status_l = store.h5["dic/left/status"][k]
            status_r = store.h5["dic/right/status"][k]
            status = np.full(N, Status.NOT_COMPUTED, dtype=np.uint8)
            both_ok = (status_l == Status.OK) & (status_r == Status.OK)
            status[both_ok] = tri.status[both_ok]
            bad_left = ~both_ok & (status_l != Status.OK)
            status[bad_left] = status_l[bad_left]
            status[~bad_left & ~both_ok] = status_r[~bad_left & ~both_ok]
            valid = status == Status.OK
            xyz = np.full((N, 3), np.nan, dtype=np.float64)
            xyz[valid] = tri.X[valid]
            disp = xyz - X0
            local = to_local(frames, disp)
            rep = np.full(N, np.nan, dtype=np.float32)
            rep[valid] = tri.reprojection_error[valid].astype(np.float32)
            store.h5["reconstruction/xyz"][k] = xyz
            store.h5["reconstruction/displacement_raw"][k] = disp
            store.h5["reconstruction/displacement_local"][k] = local
            store.h5["reconstruction/reprojection_error"][k] = rep
            store.h5["reconstruction/status"][k] = status
            store.h5["quality/valid_mask"][k] = valid
            done[k] = True
            store.flush()
            progress.update(k + 1, T, f"row {k}")
            progress.summary({"row": k, "valid": float(valid.mean()) if N else 0.0})
        if bool(np.all(done[...])):
            store.mark_stage_done("reconstruct", stage_hash)
    return out


def _rbm_poi_mask(cfg: ExperimentConfig, xy_left_image: np.ndarray) -> np.ndarray | None:
    if cfg.rbm_mask is None:
        return None
    mask_img = _read_mask_image(cfg.data_path(cfg.rbm_mask), None, "rbm_mask")
    h, w = mask_img.shape
    xy = np.asarray(xy_left_image, dtype=np.float64).reshape(-1, 2)
    ix = np.rint(xy[:, 0]).astype(np.int64)
    iy = np.rint(xy[:, 1]).astype(np.int64)
    inside = np.all(np.isfinite(xy), axis=1) & (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    out = np.zeros(len(xy), dtype=bool)
    out[inside] = mask_img[iy[inside], ix[inside]]
    return out


def _rbm_stage_hash(cfg: ExperimentConfig, progress_attrs, xyz_shape: tuple[int, ...]) -> str:
    rbm_mask_sha = _file_sha256(cfg.data_path(cfg.rbm_mask)) if cfg.rbm_mask is not None else ""
    return config_hash(
        {
            "reconstruct_hash": str(progress_attrs.get("reconstruct_config_hash", "")),
            "shape": list(xyz_shape),
            "reference": "reference/points_3d",
            "rbm_mask": str(cfg.rbm_mask),
            "rbm_mask_sha256": rbm_mask_sha,
        }
    )


def run_rbm(cfg: ExperimentConfig, force: bool = False, progress: ProgressReporter | None = None) -> Path:
    """Estimate and remove frame-wise rigid motion from reconstructed points."""
    from .core.status import Status
    from .io.storage import ResultStore
    from .recon.frames import to_local
    from .recon.rbm import compensate_trajectory

    progress = ensure_progress(progress)
    if cfg.output_file is None:
        raise ValueError("experiment config has no output_file")
    out = cfg.results_path(cfg.output_file)
    progress.phase("loading")
    with ResultStore.open(out, "r+") as store:
        if "reconstruction/xyz" not in store.h5 or "reference/points_3d" not in store.h5:
            raise ValueError("reconstruction data is missing; run --stage reconstruct first")
        xyz = store.read_array("reconstruction/xyz")
        reference = store.read_array("reference/points_3d")
        frames = store.read_array("reference/local_frames")
        status = store.read_array("reconstruction/status")
        T, N, _ = xyz.shape
        prog = store.h5.require_group("progress")
        stage_hash = _rbm_stage_hash(cfg, prog.attrs, xyz.shape)
        if not force and store.stage_done("rbm", stage_hash):
            return out
        ref_mask: np.ndarray | None = None
        if cfg.rbm_mask is not None:
            ref_mask = _rbm_poi_mask(cfg, store.read_array("reference/points_left_image"))
            if ref_mask is None or int(ref_mask.sum()) < 3:
                raise ValueError(f"rbm_mask selects fewer than 3 POIs ({0 if ref_mask is None else int(ref_mask.sum())})")
        else:
            log.warning("rbm_mask is not configured; fitting rigid-body motion on all valid POIs")
        progress.check_cancel()
        progress.phase("fitting")
        compensation = compensate_trajectory(reference, xyz, valid_mask=status == Status.OK, reference_mask=ref_mask)
        disp_local = to_local(frames, compensation.displacement)
        progress.check_cancel()
        progress.phase("writing")
        _ensure_timeseries(store, "reconstruction/displacement_rbm", T, N, 3, dtype=np.float64, reset=True)
        _ensure_timeseries(store, "reconstruction/displacement_rbm_local", T, N, 3, dtype=np.float64, reset=True)
        _ensure_timeseries(store, "reconstruction/rbm_R", T, 3, 3, dtype=np.float64, reset=True)
        _ensure_timeseries(store, "reconstruction/rbm_t", T, 3, dtype=np.float64, reset=True)
        _ensure_timeseries(store, "reconstruction/rbm_residual_rms", T, 1, dtype=np.float64, reset=True)
        _ensure_timeseries(store, "quality/rbm_valid", T, 1, dtype=np.bool_, fill_value=False, reset=True)
        store.h5["reconstruction/displacement_rbm"][:] = compensation.displacement
        store.h5["reconstruction/displacement_rbm_local"][:] = disp_local
        store.h5["reconstruction/rbm_R"][:] = compensation.rotations
        store.h5["reconstruction/rbm_t"][:] = compensation.translations
        store.h5["reconstruction/rbm_residual_rms"][:, 0] = compensation.residual_rms
        store.h5["quality/rbm_valid"][:, 0] = compensation.valid
        if ref_mask is not None:
            store.write_array("quality/rbm_reference_mask", ref_mask)
        store.mark_stage_done("rbm", stage_hash)
    return out


# ------------------------------------------------------------------ stage status
def _stage_record(state: str, detail: str = "", done_utc: str | None = None, fraction: float | None = None) -> dict:
    return {"state": state, "detail": detail, "done_utc": done_utc, "fraction": fraction}


def _calibration_status(cfg: ExperimentConfig) -> dict:
    import json

    if cfg.calibration_file is None:
        return _stage_record("missing", "calibration_file is not configured")
    path = cfg.data_path(cfg.calibration_file)
    if not path.exists():
        return _stage_record("missing")
    if cfg.calibration is None:
        return _stage_record("done", "external calibration file")
    sidecar = path.with_suffix(".json")
    try:
        report = json.loads(sidecar.read_text(encoding="utf-8")).get("report", {}) if sidecar.exists() else {}
    except (OSError, ValueError):
        report = {}
    stored = report.get("config_hash")
    mtime = _mtime_utc(path)
    if stored is None:
        return _stage_record("done", "no config hash in the calibration report", mtime)
    if stored == config_hash(cfg.calibration):
        return _stage_record("done", "", mtime)
    return _stage_record("stale", "calibration settings changed", mtime)


def _mtime_utc(path: Path) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(Path(path).stat().st_mtime, timezone.utc).isoformat()


def stage_status(cfg: ExperimentConfig) -> dict[str, dict]:
    """State of every pipeline stage for the current configuration (read-only).

    Each entry is ``{"state", "detail", "done_utc", "fraction"}`` with ``state`` one of
    ``missing`` (never run), ``done`` (finished with the current settings), ``stale``
    (finished, but settings or an upstream stage changed), ``partial`` (interrupted,
    resumable; ``fraction`` gives the finished share) or ``unknown`` (cannot be decided,
    see ``detail``). Must not be called while a job is writing the results file.
    """
    from .io.storage import ResultStore

    out: dict[str, dict] = {s: _stage_record("missing") for s in STAGES}
    try:
        out["calibrate"] = _calibration_status(cfg)
    except Exception as exc:  # noqa: BLE001 - status must never raise
        out["calibrate"] = _stage_record("unknown", str(exc))
    if cfg.output_file is None:
        for s in STAGES[1:]:
            out[s] = _stage_record("missing", "output_file is not configured")
        return out
    try:
        path = cfg.results_path(cfg.output_file)
    except ValueError as exc:
        for s in STAGES[1:]:
            out[s] = _stage_record("unknown", str(exc))
        return out
    if not path.exists():
        return out

    def done_record(attrs, stage: str, current: str | None) -> dict:
        utc = attrs.get(f"{stage}_done_utc")
        utc = str(utc) if utc is not None else None
        if not bool(attrs.get(f"{stage}_done", False)):
            return _stage_record("missing")
        if current is None or attrs.get(f"{stage}_config_hash") == current:
            return _stage_record("done", "", utc)
        return _stage_record("stale", "settings or inputs changed", utc)

    def resumable(rec: dict, attrs, stage: str, current: str | None, done_ds) -> dict:
        if rec["state"] != "missing" or current is None or done_ds is None:
            return rec
        if attrs.get(f"{stage}_run_hash") != current:
            return rec
        arr = np.asarray(done_ds[...])
        rows = arr.all(axis=1) if arr.ndim == 2 else arr
        frac = float(rows.mean()) if rows.size else 0.0
        return _stage_record("partial", f"{int(rows.sum())}/{rows.size} frames", None, frac) if frac > 0 else rec

    try:
        with ResultStore.open(path, "r") as store:
            h5 = store.h5
            attrs = h5.require_group("progress").attrs if "progress" in h5 else {}

            try:
                cur = _stereo_stage_hash(cfg)
            except Exception as exc:  # noqa: BLE001
                cur, out["stereo"] = None, _stage_record("unknown", str(exc))
            if cur is not None or out["stereo"]["state"] != "unknown":
                out["stereo"] = done_record(attrs, "stereo", cur)

            try:
                from .io.sequence import open_sequence

                seq_l = open_sequence(cfg.data_path(cfg.cameras.left.path), cfg.cameras.left.pattern)
                seq_r = open_sequence(cfg.data_path(cfg.cameras.right.path), cfg.cameras.right.pattern)
                frame_ids = _selected_frame_indices(cfg, len(seq_l), len(seq_r))
                cur = _temporal_stage_hash(
                    cfg, frame_ids, cfg.data_path(cfg.calibration_file), str(attrs.get("stereo_config_hash", ""))
                )
            except Exception as exc:  # noqa: BLE001
                cur = None
                out["temporal"] = _stage_record("unknown", str(exc))
            if cur is not None:
                rec = done_record(attrs, "temporal", cur)
                ds = h5["progress/dic_frames_done"] if "progress/dic_frames_done" in h5 else None
                out["temporal"] = resumable(rec, attrs, "temporal", cur, ds)

            if "dic/left/xy" in h5:
                cur = _reconstruct_stage_hash(cfg, attrs, h5["dic/left/xy"].shape)
                rec = done_record(attrs, "reconstruct", cur)
                ds = h5["progress/reconstruction_done"] if "progress/reconstruction_done" in h5 else None
                out["reconstruct"] = resumable(rec, attrs, "reconstruct", cur, ds)

            if "reconstruction/xyz" in h5:
                out["rbm"] = done_record(attrs, "rbm", _rbm_stage_hash(cfg, attrs, h5["reconstruction/xyz"].shape))

            for stage in ("analyze", "uncertainty"):
                out[stage] = done_record(attrs, stage, None)
    except Exception as exc:  # noqa: BLE001
        for s in STAGES[1:]:
            out[s] = _stage_record("unknown", f"cannot read results file: {exc}")
        return out

    # A finished stage whose upstream is not current is stale as well.
    # rbm is optional for analyze, so analyze depends on reconstruct.
    edges = (
        ("calibrate", "stereo"),
        ("stereo", "temporal"),
        ("temporal", "reconstruct"),
        ("reconstruct", "rbm"),
        ("reconstruct", "analyze"),
        ("stereo", "uncertainty"),
    )
    for up, down in edges:
        if out[up]["state"] in ("stale", "missing", "partial") and out[down]["state"] == "done":
            out[down] = _stage_record("stale", f"upstream stage {up} is not current", out[down]["done_utc"])
    return out
