"""Vibration analysis, quality / uncertainty, report and exports (U5)."""

from __future__ import annotations

import base64
import html
import io
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, Response

from stereodic.vibration.mac import mac_matrix

from .results import _clean, _open, _results

router = APIRouter(prefix="/api/experiments/{name}", tags=["analysis"])


def _noise(request: Request, source: str | None) -> dict[str, Any] | None:
    """Noise floor of another experiment (a static acquisition), if it has one."""
    if not source:
        return None
    try:
        path = _results(request, source)
    except HTTPException:
        return None
    with _open(path) as f:
        if "quality/noise_floor/std" not in f:
            return None
        g = f["quality/noise_floor"]
        return {
            "experiment": source,
            "std": g["std"][()],
            "n_frames": int(g.attrs.get("n_frames", 0)),
            "spectrum_frequencies": g["spectrum_frequencies"][()],
            "spectrum_amplitude": g["spectrum_amplitude"][()],
        }


@router.get("/vibration")
def vibration(request: Request, name: str, noise_from: str | None = None) -> dict[str, Any]:
    with _open(_results(request, name)) as f:
        if "vibration/spectrum_frequencies" not in f:
            raise HTTPException(404, "no vibration analysis yet")
        g = f["vibration"]
        opts = json.loads(g.attrs.get("options", "{}"))
        freqs = g["frequencies"][()]
        damp = g["damping"][()] if "damping" in g else np.full((freqs.size, 5), np.nan)
        peaks = [
            {"frequency_hz": float(fr), "amplitude_mm": float(a), "damping": None if not np.isfinite(d[3]) else float(d[3]),
             "damping_limited": bool(d[4]), "left_hz": float(d[1]), "right_hz": float(d[2])}
            for fr, a, d in zip(freqs, g["amplitudes"][()], damp)
        ]
        harmonic = []
        if "harmonic" in g:
            for k, h in g["harmonic"].items():
                amp = h["amplitude"][0]
                harmonic.append({
                    "key": k,
                    "frequency_hz": float(h["frequency_hz"][()]),
                    "requested_hz": float(h["requested_frequency_hz"][()]) if "requested_frequency_hz" in h else None,
                    "direction": int(h.attrs.get("direction", 2)),
                    "use_rbm": bool(h.attrs.get("use_rbm", False)),
                    "max_amplitude_mm": [float(np.nanmax(amp[:, c])) if np.isfinite(amp[:, c]).any() else None for c in range(3)],
                })
            harmonic.sort(key=lambda h: h["frequency_hz"])
        out: dict[str, Any] = {
            "options": opts,
            "sample_rate_hz": float(g.attrs.get("sample_rate_hz", np.nan)),
            "n_frames": int(f["time/timestamps"].shape[0]),
            "duration_s": float(np.ptp(f["time/timestamps"][()])),
            "spectrum": {
                "frequencies": _clean(g["spectrum_frequencies"][()], 6),
                "amplitude": _clean(g["spectrum_amplitude_all"][()].T, 9) if "spectrum_amplitude_all" in g
                else [_clean(g["spectrum_amplitude"][()], 9)] * 3,
            },
            "peaks": peaks,
            "damping_resolution_hz": float(g["damping"].attrs.get("resolution_hz", np.nan)) if "damping" in g else None,
            "harmonic": harmonic,
            "fdd": None,
            "frf": None,
        }
        if "fdd" in g:
            out["fdd"] = {
                "frequencies": _clean(g["fdd/frequencies"][()], 6),
                "singular_values": _clean(g["fdd/singular_values"][()].T, 12),
                "peak_frequencies": _clean(g["fdd/peak_frequencies"][()], 6),
            }
        if "frf" in g:
            out["frf"] = {
                "frequencies": _clean(g["frf/frequencies"][()], 6),
                "H1_mean_abs": _clean(g["frf/H1_mean_abs"][()], 9),
                "coherence_mean": _clean(g["frf/coherence_mean"][()], 4),
            }
    nf = _noise(request, noise_from)
    if nf is not None:
        out["noise"] = {"experiment": nf["experiment"], "frequencies": _clean(nf["spectrum_frequencies"], 6),
                        "amplitude": _clean(nf["spectrum_amplitude"].T, 9)}
    return out


@router.get("/vibration/harmonic/{key}")
def harmonic(request: Request, name: str, key: str, noise_from: str | None = None, k_sigma: float = 5.0) -> dict[str, Any]:
    """Amplitude, phase and complex response (first harmonic) of every point, 3 components."""
    with _open(_results(request, name)) as f:
        base = f"vibration/harmonic/{key}"
        if base not in f:
            raise HTTPException(404, f"no harmonic fit {key}")
        h = f[base]
        amp = h["amplitude"][0]
        resp = h["response"][0]
        T = int(f["time/timestamps"].shape[0])
        out = {
            "key": key,
            "frequency_hz": float(h["frequency_hz"][()]),
            "amplitude": _clean(amp, 9),
            "phase": _clean(h["phase"][0], 6),
            "re": _clean(resp.real, 9),
            "im": _clean(resp.imag, 9),
            "residual_rms": _clean(h["residual_rms"][()], 9),
            "significant": None,
        }
    nf = _noise(request, noise_from)
    if nf is not None and nf["std"].shape == amp.shape:
        # harmonic-fit noise of the amplitude: sigma_A ~ sqrt(2/N) sigma_n (white noise, docs §11.3)
        thr = k_sigma * np.sqrt(2.0 / T) * nf["std"]
        out["significant"] = (amp > thr).astype(int).tolist()
        out["noise_threshold_mm"] = _clean(thr, 9)
        out["noise_from"] = nf["experiment"]
    return out


@router.get("/vibration/fdd/{p}")
def fdd_mode(request: Request, name: str, p: int) -> dict[str, Any]:
    with _open(_results(request, name)) as f:
        if "vibration/fdd/peak_mode_shapes" not in f:
            raise HTTPException(404, "no FDD result")
        shapes = f["vibration/fdd/peak_mode_shapes"]
        if not 0 <= p < shapes.shape[0]:
            raise HTTPException(404, "peak index out of range")
        v = shapes[p]
        # normalize: largest component real positive, unit max modulus
        k = int(np.nanargmax(np.abs(v)))
        v = v * np.exp(-1j * np.angle(v[k])) / max(np.abs(v[k]), 1e-300)
        return {"p": p, "frequency_hz": float(f["vibration/fdd/peak_frequencies"][p]),
                "direction": json.loads(f["vibration"].attrs.get("options", "{}")).get("direction", 2),
                "re": _clean(v.real, 6), "im": _clean(v.imag, 6)}


@router.get("/vibration/mac")
def mac(request: Request, name: str) -> dict[str, Any]:
    """MAC between all stored harmonic ODS (analysis direction) and the FDD peak shapes."""
    with _open(_results(request, name)) as f:
        vecs, labels = [], []
        d = json.loads(f["vibration"].attrs.get("options", "{}")).get("direction", 2) if "vibration" in f else 2
        if "vibration/harmonic" in f:
            for k, h in sorted(f["vibration/harmonic"].items(), key=lambda kv: float(kv[1]["frequency_hz"][()])):
                vecs.append(h["response"][0][:, d])
                labels.append(f"ODS {float(h['frequency_hz'][()]):.4g} Hz")
        if "vibration/fdd/peak_mode_shapes" in f:
            for j, fr in enumerate(f["vibration/fdd/peak_frequencies"][()]):
                vecs.append(f["vibration/fdd/peak_mode_shapes"][j])
                labels.append(f"FDD {float(fr):.4g} Hz")
        if not vecs:
            raise HTTPException(404, "nothing to compare")
        V = np.array(vecs)
        ok = np.all(np.isfinite(V), axis=0)
        M = mac_matrix(V[:, ok], V[:, ok])
        return {"labels": labels, "mac": _clean(np.real(M), 4)}


# ------------------------------------------------------------------ quality / uncertainty
@router.get("/quality")
def quality(request: Request, name: str) -> dict[str, Any]:
    with _open(_results(request, name)) as f:
        out: dict[str, Any] = {"noise_floor": None, "uncertainty": [], "n_frames": int(f["time/timestamps"].shape[0]) if "time/timestamps" in f else None}
        if "quality/noise_floor/std" in f:
            g = f["quality/noise_floor"]
            std = g["std"][()]
            out["noise_floor"] = {
                "std": _clean(std, 9),
                "median_std_mm": [float(np.nanmedian(std[:, c])) for c in range(3)],
                "p95_std_mm": [float(np.nanpercentile(std[:, c], 95)) for c in range(3)],
                "n_frames": int(g.attrs.get("n_frames", 0)),
                "use_rbm": bool(g.attrs.get("use_rbm", False)),
                "spectrum": {"frequencies": _clean(g["spectrum_frequencies"][()], 6), "amplitude": _clean(g["spectrum_amplitude"][()].T, 9)},
            }
        if "uncertainty" in f:
            for k, g in f["uncertainty"].items():
                std = g["std_xyz"][()]
                item = {"key": k, "pixel_sigma": float(g.attrs.get("pixel_sigma", np.nan)), "std": _clean(std, 9),
                        "median_std_mm": [float(np.nanmedian(std[:, c])) for c in range(3)]}
                if "monte_carlo_std" in g:
                    mc = g["monte_carlo_std"][()]
                    item["monte_carlo_median_std_mm"] = [float(np.nanmedian(mc[:, c])) for c in range(3)]
                    item["monte_carlo"] = int(g.attrs.get("monte_carlo", 0))
                out["uncertainty"].append(item)
        if "quality/valid_mask" in f:
            vm = f["quality/valid_mask"]
            T = vm.shape[0]
            idx = np.unique(np.linspace(0, T - 1, min(T, 200)).astype(int))
            out["valid_fraction"] = float(vm[idx.tolist()].mean())
        return out


# ------------------------------------------------------------------ exports
def _csv(rows: list[list[Any]], header: list[str]) -> str:
    buf = io.StringIO()
    buf.write(",".join(header) + "\n")
    for r in rows:
        buf.write(",".join("" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.9g}" if isinstance(v, float) else str(v) for v in r) + "\n")
    return buf.getvalue()


def _attach(text: str, filename: str, media: str = "text/csv") -> Response:
    return Response(text, media_type=media, headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@router.get("/export/peaks.csv")
def export_peaks(request: Request, name: str) -> Response:
    v = vibration(request, name)
    rows = [[p["frequency_hz"], p["amplitude_mm"], p["damping"], int(p["damping_limited"])] for p in v["peaks"]]
    return _attach(_csv(rows, ["frequency_hz", "amplitude_mm", "damping_ratio", "damping_resolution_limited"]), f"{name}_peaks.csv")


@router.get("/export/harmonic/{key}.csv")
def export_harmonic(request: Request, name: str, key: str) -> Response:
    with _open(_results(request, name)) as f:
        h = f.get(f"vibration/harmonic/{key}")
        if h is None:
            raise HTTPException(404, "no such harmonic fit")
        X = f["reference/points_3d"][()]
        ids = f["reference/point_ids"][()]
        amp, ph = h["amplitude"][0], h["phase"][0]
        rows = [[int(ids[i]), *map(float, X[i]), *map(float, amp[i]), *map(float, ph[i])] for i in range(ids.size)]
        fr = float(h["frequency_hz"][()])
    return _attach(_csv(rows, ["point_id", "x_mm", "y_mm", "z_mm", "amp_c_mm", "amp_s_mm", "amp_n_mm",
                               "phase_c_rad", "phase_s_rad", "phase_n_rad"]), f"{name}_harmonic_{fr:.6g}Hz.csv")


@router.get("/export/shape.ply")
def export_ply(request: Request, name: str) -> Response:
    with _open(_results(request, name)) as f:
        X = f["reference/points_3d"][()]
        tri = f["reference/connectivity"][()]
    ok = np.all(np.isfinite(X), axis=1)
    remap = -np.ones(X.shape[0], dtype=np.int64)
    remap[ok] = np.arange(int(ok.sum()))
    faces = remap[tri]
    faces = faces[np.all(faces >= 0, axis=1)]
    buf = io.StringIO()
    buf.write(f"ply\nformat ascii 1.0\ncomment StereoDIC reference shape, left camera frame, mm\nelement vertex {int(ok.sum())}\n"
              "property float x\nproperty float y\nproperty float z\n"
              f"element face {faces.shape[0]}\nproperty list uchar int vertex_indices\nend_header\n")
    for p in X[ok]:
        buf.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
    for a, b, c in faces:
        buf.write(f"3 {a} {b} {c}\n")
    return _attach(buf.getvalue(), f"{name}_shape.ply", "application/octet-stream")


@router.get("/export/point/{i}.csv")
def export_point(request: Request, name: str, i: int) -> Response:
    with _open(_results(request, name)) as f:
        if "reconstruction/displacement_local" not in f:
            raise HTTPException(404, "no reconstruction")
        T, N, _ = f["reconstruction/displacement_local"].shape
        if not 0 <= i < N:
            raise HTTPException(404, "point out of range")
        t = f["time/timestamps"][()]
        d = f["reconstruction/displacement_local"][:, i, :]
        r = f["reconstruction/displacement_rbm_local"][:, i, :] if "reconstruction/displacement_rbm_local" in f else np.full_like(d, np.nan)
    rows = [[float(t[k]), *map(float, d[k]), *map(float, r[k])] for k in range(T)]
    return _attach(_csv(rows, ["t_s", "c_mm", "s_mm", "n_mm", "c_rbm_mm", "s_rbm_mm", "n_rbm_mm"]), f"{name}_point_{i}.csv")


# ------------------------------------------------------------------ report
def _fig_png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@router.get("/report.html", response_class=HTMLResponse)
def report(request: Request, name: str) -> HTMLResponse:
    """Self-contained HTML report (print to PDF from the browser)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from stereodic.pipeline import stage_status

    ws = request.app.state.workspace
    cfg = ws.load(name)
    path = _results(request, name)
    status = stage_status(cfg)
    e = html.escape
    parts: list[str] = []
    with _open(path) as f:
        meta = json.loads(f["metadata"].attrs.get("input_manifest", "{}"))
        git = f["metadata"].attrs.get("git_commit", "")
        T = int(f["time/timestamps"].shape[0]) if "time/timestamps" in f else 0
        N = int(f["reference/points_3d"].shape[0]) if "reference/points_3d" in f else 0
        parts.append(f"<h1>StereoDIC 测量报告 · {e(name)}</h1><p class='muted'>{e(cfg.description)}</p>")
        parts.append("<table class='kv'>" + "".join(f"<tr><th>{e(k)}</th><td>{e(str(v))}</td></tr>" for k, v in [
            ("生成时间", datetime.now().strftime("%Y-%m-%d %H:%M")), ("git commit", git),
            ("帧率", f"{cfg.cameras.frame_rate_hz:g} fps"), ("帧数 × 测点", f"{T} × {N}"),
            ("子区 / 步长", f"{cfg.dic.poi.subset_size} / {cfg.dic.poi.step} px"), ("参考帧", f"{cfg.reference_frame} ({meta.get('reference_left', '')})"),
        ]) + "</table>")
        parts.append("<h2>流程状态</h2><table><tr>" + "".join(f"<th>{e(k)}</th>" for k in status) + "</tr><tr>"
                     + "".join(f"<td>{e(v['state'])}</td>" for v in status.values()) + "</tr></table>")
        cal = Path(cfg.data_path(cfg.calibration_file)).with_suffix(".json") if cfg.calibration_file else None
        if cal and cal.exists():
            r = json.loads(cal.read_text(encoding="utf-8")).get("report", {})
            if r:
                parts.append("<h2>标定</h2><table class='kv'>" + "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in [
                    ("立体 RMS", f"{r.get('stereo_rms_px', float('nan')):.4f} px"), ("基线", f"{r.get('baseline_mm', float('nan')):.3f} mm"),
                    ("光轴夹角", f"{r.get('optical_axis_angle_deg', float('nan')):.2f}°"), ("参与视图", len(r.get("pairs", []))),
                ]) + "</table>")
        if "quality/valid_mask" in f:
            parts.append(f"<p>有效测点比例（时间平均）：{float(np.mean(f['quality/valid_mask'][::max(1, T // 200)])) * 100:.2f}%</p>")
        if "vibration/frequencies" in f:
            g = f["vibration"]
            fig, ax = plt.subplots(figsize=(8, 3))
            fr = g["spectrum_frequencies"][()]
            allamp = g["spectrum_amplitude_all"][()] if "spectrum_amplitude_all" in g else g["spectrum_amplitude"][()][:, None]
            for c, lab in enumerate(("c", "s", "n")[: allamp.shape[1]]):
                ax.semilogy(fr[1:], allamp[1:, c] * 1e3, lw=1, label=lab)
            for p in g["frequencies"][()]:
                ax.axvline(p, color="gray", lw=0.6, ls="--")
            ax.set(xlabel="frequency [Hz]", ylabel="amplitude [µm]")
            ax.legend()
            parts.append(f"<h2>振动分析</h2><img src='{_fig_png(fig)}'>")
            damp = g["damping"][()] if "damping" in g else None
            rows = "".join(
                f"<tr><td>{fq:.4f}</td><td>{a * 1e3:.3f}</td><td>{'' if damp is None or not np.isfinite(damp[i, 3]) else f'{damp[i, 3]:.4f}' + (' (分辨率受限)' if damp[i, 4] else '')}</td></tr>"
                for i, (fq, a) in enumerate(zip(g["frequencies"][()], g["amplitudes"][()]))
            )
            parts.append(f"<table><tr><th>峰值频率 Hz</th><th>幅值 µm</th><th>阻尼比</th></tr>{rows}</table>")
            if "harmonic" in g:
                X = f["reference/points_3d"][()]
                for k, h in sorted(g["harmonic"].items(), key=lambda kv: float(kv[1]["frequency_hz"][()])):
                    amp = h["amplitude"][0][:, int(h.attrs.get("direction", 2))] * 1e3
                    fig, ax = plt.subplots(figsize=(4.2, 3.4))
                    sc = ax.scatter(X[:, 0], X[:, 1], c=amp, s=8, cmap="viridis")
                    ax.invert_yaxis()
                    ax.set_aspect("equal")
                    ax.set(title=f"{float(h['frequency_hz'][()]):.4f} Hz amplitude [µm]", xlabel="x [mm]", ylabel="y [mm]")
                    fig.colorbar(sc, ax=ax)
                    parts.append(f"<img class='half' src='{_fig_png(fig)}'>")
        if "quality/noise_floor/std" in f:
            std = f["quality/noise_floor/std"][()] * 1e3
            parts.append("<h2>静态噪声底</h2><p>逐点标准差中位数 c / s / n："
                         + " / ".join(f"{np.nanmedian(std[:, c]):.3f}" for c in range(3)) + " µm</p>")
    css = ("html{color-scheme:light;background:#fff}body{background:#fff;font-family:-apple-system,'PingFang SC','Noto Sans SC',sans-serif;max-width:900px;margin:24px auto;color:#111;padding:0 16px}"
           "h1{font-size:22px}h2{font-size:16px;margin-top:28px;border-bottom:1px solid #ddd;padding-bottom:4px}"
           "table{border-collapse:collapse;margin:8px 0;font-size:13px}td,th{border:1px solid #ddd;padding:4px 8px;text-align:left}"
           ".kv th{background:#f6f6f6;font-weight:500}.muted{color:#666}img{max-width:100%}img.half{width:48%;margin-right:1%}"
           "@media print{body{margin:0}}")
    return HTMLResponse(f"<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><title>{e(name)} 报告</title><style>{css}</style></head><body>{''.join(parts)}</body></html>")

