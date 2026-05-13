# designer

Flask + vanilla-JS trajectory designer for the ball-on-plate stack. Draw a
path on an iPad (or any browser), pick a backend, and the server dispatches
the matching launch command.

## Layout

```
designer/
  server.py            Flask app: REST + static serve, holds the active backend
  run.py               wrapper that tears down a stale server on the port and restarts
  processing.py        pixels -> metres, Gaussian smooth, arc-length resample, safe-rect clamp
  launcher/
    base.py            backend protocol
    sim.py             SimLauncher: spawns ``mjpython -m sim.runner``
    real.py            RealLauncher: stages a drawing, returns the ``ros2 launch`` line to paste
  static/              SPA (HTML/CSS/vanilla JS)
  designer.yaml        server settings (port, default backend, processing knobs)
  trajectories/        gitignored, saved drawings land here
```

Plate / ball / physics constants come from ``config/hardware.yaml`` via
``ballplate.hardware``, so the canvas safe rectangle and the run targets
stay in lockstep with the simulator and the real robot.

## Backends

The server holds two stateless launchers and one "current backend" pointer.
The Sim / Real toggle in the UI flips the pointer; every Run button uses
the active launcher.

- **Sim**: ``SimLauncher`` spawns ``mjpython -m sim.runner`` with the chosen
  drawing or preset. Local process on the designer host.
- **Real**: ``RealLauncher`` copies the drawing into the ROS controller's
  ``runtime/`` directory and returns the exact ``ros2 launch
  ball_balance_controller ...`` command. The user pastes it into a terminal
  on the lab machine; the designer never SSHes anywhere.

## Setup

This package is part of the repo-root uv workspace; from the repo root:

```bash
uv sync
```

## Run

Easiest path on macOS: double-click ``designer/launch.command``. It kills
any previous server on the configured port, starts a fresh one, and opens
Chrome on the UI.

Equivalent CLI:

```bash
uv run python -m designer.run      # wrapper with auto-restart on port reuse
uv run python -m designer.server   # raw server, no auto-restart
```

The server binds ``0.0.0.0:5050`` so any device on the LAN (or the same
Tailscale tailnet) can reach it. From an iPad, open
``http://<your-mac-hostname>.local:5050`` (find the hostname with
``scutil --get LocalHostName``).

## REST surface

| Method | Path                          | Purpose                                                 |
|--------|-------------------------------|---------------------------------------------------------|
| GET    | ``/api/config``               | plate size, ball radius, available controllers, backends |
| POST   | ``/api/backend``              | switch the active backend (``{"backend": "sim"}``)      |
| POST   | ``/api/save``                 | save raw points as a processed JSON drawing             |
| GET    | ``/api/list``                 | list saved drawings                                     |
| DELETE | ``/api/drawing/<name>``       | delete a saved drawing                                  |
| POST   | ``/api/run``                  | dispatch a drawn trajectory through the active backend  |
| POST   | ``/api/run_preset``           | dispatch a built-in preset (stationary, circle, fig8)   |
| POST   | ``/api/stop``                 | stop the active backend (sim only)                      |
| GET    | ``/api/status``               | run flag plus the last staged real-launch payload       |
| GET    | ``/api/presets``              | built-in shapes with sampled previews                   |
| GET    | ``/api/training/runs``         | RL training runs found under ``sim/policies/``          |
| GET    | ``/api/training/metrics/<name>`` | parsed ``metrics.jsonl`` ready for plotting           |
