"""Stereo matching of the reference pair L0 <-> R0 (docs §7.5–7.6).

1. SIFT features on both images (8-bit stretched copies, used for initial guesses only),
   ratio test, then outlier rejection with the epipolar geometry: the calibrated
   fundamental matrix when a calibration is given, otherwise RANSAC.
2. Seeds: for POIs with enough nearby SIFT matches, a RANSAC local affine fit gives the
   initial warp; IC-GN (second order by default) refines it.
3. RG-DIC propagates from the best seeds through the POI grid. Regions that the
   propagation cannot reach are seeded again from their own SIFT matches.
4. Points with ZNSSD above `max_znssd` are marked HIGH_RESIDUAL. Geometric (epipolar)
   consistency is checked by triangulation in `recon`.

Only the left image defines POIs; right positions are non-integer and are later used as
reference centres for the right-camera temporal DIC.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.spatial import cKDTree

from ..core.imaging import to_uint8
from ..core.status import Status
from ..core.types import DICBatchResult, POISet, StereoCalibration
from . import shape_functions as sf
from .icgn import ICGNOptions, ICGNSolver
from .interp import BSplineImage
from .mesh import grid_neighbors
from .rgdic import PropagationStats, propagate

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class StereoMatchOptions:
    icgn: ICGNOptions = ICGNOptions(shape_order=2)
    interp_order: int = 5
    max_znssd: float = 0.4
    seed_radius_px: float = 40.0
    min_seed_matches: int = 8
    n_seed_candidates: int = 30
    sift_ratio: float = 0.75
    epipolar_threshold_px: float = 1.5
    reliability_guided: bool = True


@dataclass
class FeatureMatches:
    left: np.ndarray  # (M, 2) pixels
    right: np.ndarray  # (M, 2)
    n_raw: int = 0


@dataclass
class StereoMatchResult:
    pois: POISet
    result: DICBatchResult  # xy = right-image positions, params = warp L0 -> R0
    seeds: np.ndarray
    matches: FeatureMatches
    stats: PropagationStats = field(default_factory=PropagationStats)

    @property
    def xy_left(self) -> np.ndarray:
        return self.pois.xy

    @property
    def xy_right(self) -> np.ndarray:
        return self.result.xy


# -------------------------------------------------------------------- features
def _sampson(F: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Sampson distance (px) of correspondences a <-> b for x_b^T F x_a = 0."""
    ha = np.column_stack([a, np.ones(len(a))])
    hb = np.column_stack([b, np.ones(len(b))])
    Fa = ha @ F.T
    Ftb = hb @ F
    num = np.einsum("ni,ni->n", hb, Fa) ** 2
    den = Fa[:, 0] ** 2 + Fa[:, 1] ** 2 + Ftb[:, 0] ** 2 + Ftb[:, 1] ** 2
    return np.sqrt(num / np.maximum(den, 1e-300))


def sift_matches(
    left: np.ndarray,
    right: np.ndarray,
    calib: StereoCalibration | None = None,
    ratio: float = 0.75,
    epipolar_threshold_px: float = 1.5,
    mask_left: np.ndarray | None = None,
    sensor_origins: tuple[np.ndarray, np.ndarray] | None = None,
) -> FeatureMatches:
    """SIFT correspondences left -> right, filtered by the ratio test and epipolar geometry.

    sensor_origins: (left, right) offsets of the image windows on the calibrated sensors,
    added before applying the calibration. Returned points are image coordinates.
    """
    sift = cv2.SIFT_create()
    m8 = None if mask_left is None else np.asarray(mask_left, dtype=np.uint8) * 255
    kl, dl = sift.detectAndCompute(to_uint8(left), m8)
    kr, dr = sift.detectAndCompute(to_uint8(right), None)
    if dl is None or dr is None or len(kl) < 8 or len(kr) < 8:
        return FeatureMatches(np.zeros((0, 2)), np.zeros((0, 2)), 0)
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(dl, dr, k=2)
    good = [m[0] for m in knn if len(m) == 2 and m[0].distance < ratio * m[1].distance]
    a = np.array([kl[m.queryIdx].pt for m in good], dtype=np.float64).reshape(-1, 2)
    b = np.array([kr[m.trainIdx].pt for m in good], dtype=np.float64).reshape(-1, 2)
    n_raw = len(a)
    if n_raw < 8:
        return FeatureMatches(a, b, n_raw)
    if calib is not None:
        from ..calibration.camera_model import fundamental_matrix, pixel_to_normalized

        ol, orr = (np.zeros(2), np.zeros(2)) if sensor_origins is None else (np.asarray(o, float) for o in sensor_origins)
        ua = pixel_to_normalized(calib.left, a + ol) @ calib.left.K[:2, :2].T + calib.left.K[:2, 2]
        ub = pixel_to_normalized(calib.right, b + orr) @ calib.right.K[:2, :2].T + calib.right.K[:2, 2]
        keep = _sampson(fundamental_matrix(calib), ua, ub) < epipolar_threshold_px
    else:
        _, inl = cv2.findFundamentalMat(a, b, cv2.FM_RANSAC, epipolar_threshold_px, 0.999)
        keep = np.zeros(n_raw, bool) if inl is None else inl.ravel().astype(bool)
    return FeatureMatches(a[keep], b[keep], n_raw)


def local_affine_guess(
    centers: np.ndarray, matches: FeatureMatches, radius: float, min_matches: int, order: int
) -> tuple[np.ndarray, np.ndarray]:
    """Initial warp per POI from a RANSAC affine fit to nearby matches.

    Returns params (N, P) (NaN where no guess) and the number of inliers (N,).
    """
    n = len(centers)
    p = np.full((n, sf.n_params(order)), np.nan)
    n_in = np.zeros(n, dtype=np.int64)
    if len(matches.left) < min_matches:
        return p, n_in
    tree = cKDTree(matches.left)
    for i, c in enumerate(centers):
        nb = tree.query_ball_point(c, radius)
        if len(nb) < min_matches:
            continue
        src = (matches.left[nb] - c).astype(np.float64)
        dst = matches.right[nb].astype(np.float64)
        A, inl = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=1.5, confidence=0.99)
        if A is None or inl is None or inl.sum() < min_matches:
            continue
        A = A.copy()
        A[:, 2] -= c  # local warp: x'(d) = A[:, :2] d + (b - c)
        p[i] = sf.from_affine(A, order)
        n_in[i] = int(inl.sum())
    return p, n_in


# ------------------------------------------------------------------- matching
def _pick_seed_candidates(guess_ok: np.ndarray, n_inliers: np.ndarray, centers: np.ndarray, k: int) -> np.ndarray:
    """Up to k POIs with a guess, spread over the field (farthest-point order, weighted by inliers)."""
    idx = np.nonzero(guess_ok)[0]
    if idx.size <= k:
        return idx[np.argsort(-n_inliers[idx])]
    chosen = [int(idx[np.argmax(n_inliers[idx])])]
    d = np.linalg.norm(centers[idx] - centers[chosen[0]], axis=1)
    for _ in range(k - 1):
        nxt = int(idx[np.argmax(d)])
        chosen.append(nxt)
        d = np.minimum(d, np.linalg.norm(centers[idx] - centers[nxt], axis=1))
    return np.asarray(chosen)


def stereo_match(
    left: np.ndarray,
    right: np.ndarray,
    pois: POISet,
    calib: StereoCalibration | None = None,
    options: StereoMatchOptions = StereoMatchOptions(),
    backend: str = "auto",
    mask_left: np.ndarray | None = None,
    sensor_origins: tuple[np.ndarray, np.ndarray] | None = None,
) -> StereoMatchResult:
    """Match left-image POIs into the right image (warp L0 -> R0).

    All positions are image coordinates; `sensor_origins` only serves the calibrated
    epipolar filter of the SIFT matches (see `sift_matches`).
    """
    order = options.icgn.shape_order
    solver = ICGNSolver(options.icgn, "numba" if backend in ("auto", "cuda") else backend)
    ref = BSplineImage(left, options.interp_order)
    tgt = BSplineImage(right, options.interp_order)
    refs = solver.prepare_reference(ref, pois.xy)
    n = len(pois)

    matches = sift_matches(
        left, right, calib, options.sift_ratio, options.epipolar_threshold_px, mask_left, sensor_origins
    )
    guess, n_in = local_affine_guess(pois.xy, matches, options.seed_radius_px, options.min_seed_matches, order)
    has_guess = np.all(np.isfinite(guess), axis=1) & (refs.status == Status.OK)
    log.info("SIFT: %d raw, %d epipolar inliers; %d POIs with a guess", matches.n_raw, len(matches.left), has_guess.sum())

    result = DICBatchResult.empty(n, sf.n_params(order))
    # POIs the reference stage rejected keep that reason; the rest start as NOT_COMPUTED
    result.status[:] = np.where(refs.status == Status.OK, Status.NOT_COMPUTED, refs.status)
    stats = PropagationStats()
    seeds: list[int] = []

    if not options.reliability_guided:
        idx = np.nonzero(has_guess)[0]
        if idx.size:
            r = solver.solve(refs, tgt, guess[idx], indices=idx)
            _scatter(result, idx, r)
        seeds = idx.tolist()
    else:
        neighbors = grid_neighbors(pois.xy)
        remaining = has_guess.copy()
        while True:
            cand = _pick_seed_candidates(remaining, n_in, pois.xy, options.n_seed_candidates)
            if cand.size == 0:
                break
            r = solver.solve(refs, tgt, guess[cand], indices=cand)
            good = (r.status == Status.OK) & (r.znssd <= options.max_znssd)
            record = good | (result.status[cand] != Status.OK)  # keep failure reasons, never overwrite a solution
            _scatter(result, cand[record], _take(r, record))
            remaining[cand] = False
            if not good.any():
                continue
            seeds.extend(cand[good].tolist())
            propagate(solver, refs, tgt, neighbors, result, cand[good], options.max_znssd, stats)
            # POIs reached (solved) are no longer seed candidates
            remaining &= ~((result.status == Status.OK) & (result.znssd <= options.max_znssd))
            if not remaining.any():
                break

    high = (result.status == Status.OK) & ~(result.znssd <= options.max_znssd)
    result.status[high] = Status.HIGH_RESIDUAL
    bad = result.status != Status.OK
    result.xy[bad] = np.nan
    stats.n_solved = int((result.status == Status.OK).sum())
    return StereoMatchResult(pois, result, np.asarray(seeds, dtype=np.int64), matches, stats)


def _take(r: DICBatchResult, sel: np.ndarray) -> DICBatchResult:
    return DICBatchResult(r.xy[sel], r.znssd[sel], r.iterations[sel], r.status[sel], r.params[sel])


def _scatter(result: DICBatchResult, idx: np.ndarray, r: DICBatchResult) -> None:
    result.xy[idx] = r.xy
    result.znssd[idx] = r.znssd
    result.iterations[idx] = r.iterations
    result.status[idx] = r.status
    result.params[idx] = r.params


def options_from_config(cfg) -> StereoMatchOptions:
    """StereoMatchOptions from a DICConfig."""
    return StereoMatchOptions(
        icgn=ICGNOptions.from_config(cfg, stage="stereo"),
        interp_order=cfg.interpolation.order,
        max_znssd=cfg.stereo.max_znssd,
        seed_radius_px=cfg.stereo.seed_radius_px,
        reliability_guided=cfg.stereo.reliability_guided,
    )
