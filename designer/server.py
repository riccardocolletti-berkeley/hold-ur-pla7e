"""Flask backend for the trajectory designer.

Holds two stateless launchers (``sim`` and ``real``) plus a mutable
"current backend" pointer; every ``/api/run*`` dispatches to whichever
launcher the pointer names, so switching backends at runtime takes
effect on the next click.

Run with::

    python -m designer.server

Open ``http://<host>:5050/`` from any device on the same LAN.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from threading import Lock

import numpy as np
import yaml
from flask import Flask, jsonify, request, send_from_directory

from ballplate.hardware import default_path as _hardware_default_path
from ballplate.hardware import load as load_hardware
from ballplate.trajectories import Circle, Figure8
from designer import processing
from designer.launcher import RealLauncher, SimLauncher

# ----------------------------------------------------------- repo layout --

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DRAWINGS_DIR = _PROJECT_ROOT / "designer" / "trajectories"
_POLICIES_DIR = _PROJECT_ROOT / "sim" / "policies"
_DESIGNER_CFG = Path(__file__).resolve().parent / "designer.yaml"
_STATIC_DIR = Path(__file__).resolve().parent / "static"


# ----------------------------------------------------------- config --------

PRESET_SHAPES = [
    {"shape": "stationary", "label": "Hold Position"},
    {"shape": "circle", "label": "Circle"},
    {"shape": "figure8", "label": "Figure 8"},
]

CONTROLLERS = [
    {"id": "pid", "label": "PID", "available_for": ["sim", "real"]},
    {"id": "mpc", "label": "MPC", "available_for": ["sim", "real"]},
    {"id": "rl", "label": "RL", "available_for": ["sim"]},
]

DESIGNER_DEFAULTS = {
    "server": {"port": 5050, "default_backend": "sim"},
    "canvas": {"safe_inset_frac": 0.10, "edge_margin_m": 0.01},
    "processing": {"smooth_sigma": 1.5, "n_resample": 200},
    "real": {
        "drawing_target_dir": "real/ros2_ws/src/ball_balance_controller/runtime",
        "ros_launch_commands": {
            "pid": "ros2 launch ball_balance_controller balance.launch.py",
            "mpc": "ros2 launch ball_balance_controller mpc_balance.launch.py",
        },
    },
}


def _designer_cfg() -> dict:
    """Merge the user's `designer.yaml` over the built-in defaults."""
    user = {}
    if _DESIGNER_CFG.exists():
        with open(_DESIGNER_CFG) as f:
            user = yaml.safe_load(f) or {}
    return {k: {**v, **(user.get(k) or {})} for k, v in DESIGNER_DEFAULTS.items()}


CFG = _designer_cfg()
HW = load_hardware(_hardware_default_path())

app = Flask(__name__, static_folder=str(_STATIC_DIR), static_url_path="/static")

LAUNCHERS = {
    "sim": SimLauncher(),
    "real": RealLauncher(
        drawing_target_dir=CFG["real"]["drawing_target_dir"],
        ros_launch_commands=CFG["real"]["ros_launch_commands"],
    ),
}
# Flask's dev server runs requests on a thread pool, so the read of
# ``current_backend`` and the matching launcher call must be serialised
# to prevent a backend switch from landing mid-request. The lock also
# guards the writer in ``/api/backend``.
_backend_lock = Lock()
current_backend = CFG["server"]["default_backend"]


# ============================================================ helpers --


def _slugify(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_-]+", "_", (name or "").strip())
    return s[:40] or f"drawing_{int(time.time())}"


def _backend():
    """Return the currently selected launcher under the shared lock."""
    with _backend_lock:
        return LAUNCHERS[current_backend]


def _safe_position_bounds() -> tuple[float, float]:
    """Largest absolute |x|, |y| target that stays inside the safe plate rectangle."""
    margin = CFG["canvas"]["edge_margin_m"]
    hx = HW.plate.size[0] / 2.0
    hy = HW.plate.size[1] / 2.0
    return (
        max(0.01, hx - HW.ball.radius - margin),
        max(0.01, hy - HW.ball.radius - margin),
    )


def _safe_max_radius(shape: str) -> float:
    """Largest preset radius that still fits inside the safe plate rectangle."""
    rx, ry = _safe_position_bounds()
    if shape == "figure8":
        return min(rx, 2.0 * ry)
    return min(rx, ry)


def _sample_preset(
    shape: str,
    radius: float,
    period: float = 10.0,
    n: int = 120,
    hold_x: float = 0.0,
    hold_y: float = 0.0,
):
    """Sample a built-in shape into a list of ``[x, y]`` points for a thumbnail."""
    if shape == "stationary":
        return [[hold_x, hold_y], [hold_x, hold_y]]
    if shape == "circle":
        ref = Circle(radius=radius, period=period)
    elif shape == "figure8":
        ref = Figure8(rx=radius, ry=radius / 2.0, period=period)
    else:
        return []
    out = []
    for t in np.linspace(0.0, period, n, endpoint=False):
        x, y, _, _ = ref.evaluate(t)
        out.append([float(x), float(y)])
    return out


# ============================================================ routes --


@app.route("/")
def index():
    return send_from_directory(_STATIC_DIR, "index.html")


@app.get("/api/config")
def api_config():
    """Plate / ball geometry, available controllers, and current backend."""
    return jsonify(
        plate_size=list(HW.plate.size),
        ball_radius=HW.ball.radius,
        controllers=CONTROLLERS,
        backends=[{"id": b.backend_id, "label": b.label} for b in LAUNCHERS.values()],
        current_backend=current_backend,
        safe_inset_frac=CFG["canvas"]["safe_inset_frac"],
    )


@app.post("/api/backend")
def api_set_backend():
    body = request.get_json(force=True)
    name = body.get("backend")
    if name not in LAUNCHERS:
        return jsonify(error=f"unknown backend {name!r}"), 400
    global current_backend
    with _backend_lock:
        current_backend = name
    return jsonify(ok=True, backend=current_backend)


# --------------------------------------------------------- drawings ----


@app.post("/api/save")
def api_save():
    body = request.get_json(force=True)
    raw = body.get("points") or []
    if len(raw) < 2:
        return jsonify(error="path too short"), 400

    pts = processing.process(
        raw,
        canvas_size_px=tuple(body.get("canvas_size", [600, 600])),
        plate_size_m=tuple(HW.plate.size),
        ball_radius_m=HW.ball.radius,
        mode=body.get("path_mode", "open"),
        safe_inset_frac=CFG["canvas"]["safe_inset_frac"],
        margin_m=CFG["canvas"]["edge_margin_m"],
        smooth_sigma=CFG["processing"]["smooth_sigma"],
        n_out=CFG["processing"]["n_resample"],
    )

    name = _slugify(body.get("name") or f"drawing_{int(time.time())}")
    duration = float(body.get("duration_s", 10.0))
    end_mode = body.get("end_mode")
    if end_mode not in ("loop", "stop", "pingpong"):
        end_mode = "loop" if body.get("path_mode") == "closed" else "stop"

    _DRAWINGS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _DRAWINGS_DIR / f"{name}.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "version": 1,
                "name": name,
                "created": time.time(),
                "points_xy_m": pts.tolist(),
                "duration_s": duration,
                "end_mode": end_mode,
                "path_mode": body.get("path_mode", "open"),
            },
            f,
        )
    return jsonify(ok=True, name=name)


@app.get("/api/list")
def api_list():
    if not _DRAWINGS_DIR.exists():
        return jsonify(items=[])
    items = []
    for p in sorted(_DRAWINGS_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            with open(p) as f:
                d = json.load(f)
        except Exception:
            continue
        pts = d.get("points_xy_m", [])
        items.append(
            {
                "name": d.get("name", p.stem),
                "created": d.get("created"),
                "duration_s": d.get("duration_s"),
                "end_mode": d.get("end_mode"),
                "path_mode": d.get("path_mode"),
                "n_points": len(pts),
                "preview_xy": pts[:: max(1, len(pts) // 60)],
            }
        )
    return jsonify(items=items)


@app.delete("/api/drawing/<name>")
def api_delete(name):
    p = _DRAWINGS_DIR / f"{_slugify(name)}.json"
    if p.exists():
        p.unlink()
        return jsonify(ok=True)
    return jsonify(error="not found"), 404


# --------------------------------------------------------- presets -----


@app.get("/api/presets")
def api_presets():
    bx, by = _safe_position_bounds()
    items = []
    for spec in PRESET_SHAPES:
        if spec["shape"] == "stationary":
            items.append(
                {
                    "shape": spec["shape"],
                    "label": spec["label"],
                    "x": 0.0,
                    "y": 0.0,
                    "x_bounds": [round(-bx, 3), round(bx, 3)],
                    "y_bounds": [round(-by, 3), round(by, 3)],
                    "preview_xy": _sample_preset(spec["shape"], radius=0.0),
                }
            )
            continue
        max_r = _safe_max_radius(spec["shape"])
        radius = min(0.075, max_r)
        items.append(
            {
                "shape": spec["shape"],
                "label": spec["label"],
                "radius": round(radius, 3),
                "min_radius": 0.02,
                "max_radius": round(max_r, 3),
                "preview_xy": _sample_preset(spec["shape"], radius),
            }
        )
    return jsonify(items=items)


# --------------------------------------------------------- run/stop ----


@app.post("/api/run")
def api_run():
    body = request.get_json(force=True)
    name = _slugify(body.get("name", ""))
    p = _DRAWINGS_DIR / f"{name}.json"
    if not p.exists():
        return jsonify(error="not found"), 404
    info = _backend().start_drawn(p, controller=body.get("controller"))
    return jsonify(ok=True, **info)


@app.post("/api/run_preset")
def api_run_preset():
    body = request.get_json(force=True)
    shape = body.get("shape")
    if shape not in {s["shape"] for s in PRESET_SHAPES}:
        return jsonify(error="unknown preset"), 400
    radius = body.get("radius")
    hold_x = body.get("x")
    hold_y = body.get("y")
    info = _backend().start_preset(
        shape,
        radius=float(radius) if radius is not None else None,
        hold_x=float(hold_x) if hold_x is not None else None,
        hold_y=float(hold_y) if hold_y is not None else None,
        controller=body.get("controller"),
    )
    return jsonify(ok=True, **info)


@app.post("/api/stop")
def api_stop():
    _backend().stop()
    return jsonify(ok=True)


@app.get("/api/status")
def api_status():
    backend = _backend()
    payload = {
        "backend": current_backend,
        "running": backend.is_running(),
    }
    last = getattr(backend, "last_started", None)
    if callable(last):
        info = last()
        if info is not None:
            payload["last_started"] = info
    return jsonify(payload)


# --------------------------------------------------------- training ----


@app.get("/api/training/runs")
def api_training_runs():
    if not _POLICIES_DIR.exists():
        return jsonify(items=[])
    items = []
    for d in sorted(_POLICIES_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not d.is_dir():
            continue
        metrics = d / "metrics.jsonl"
        if not metrics.exists():
            continue
        items.append(
            {
                "name": d.name,
                "updated": metrics.stat().st_mtime,
                "size_kb": round(metrics.stat().st_size / 1024, 1),
                "has_final": (d / "ppo_final.zip").exists(),
            }
        )
    return jsonify(items=items)


@app.get("/api/training/metrics/<name>")
def api_training_metrics(name):
    p = _POLICIES_DIR / _slugify(name) / "metrics.jsonl"
    if not p.exists():
        return jsonify(error="run not found"), 404
    series: dict = {}
    steps = []
    with open(p) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            steps.append(rec.get("step"))
            for k, v in rec.items():
                if k in ("step", "time"):
                    continue
                series.setdefault(k, []).append(v)
    return jsonify(steps=steps, series=series)


# ============================================================ entry --

if __name__ == "__main__":
    # 0.0.0.0 so other devices on the LAN can reach the designer.
    app.run(host="0.0.0.0", port=int(CFG["server"]["port"]), debug=False)
