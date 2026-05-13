"""Stop any stale designer on the configured port, start a fresh one, open Chrome.

Used by ``launch.command``. Anything still bound to the configured port
gets SIGTERM before the new instance binds, so the user never has to stop
the previous run by hand.

Run with::

    python -m designer.run
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time

from designer import server


def _kill_existing(port: int) -> None:
    """SIGTERM anything still listening on ``port`` so the new server can bind."""
    try:
        out = subprocess.check_output(["lsof", "-ti", f"tcp:{port}"], text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return
    for pid in out.strip().splitlines():
        try:
            os.kill(int(pid), signal.SIGTERM)
        except (ProcessLookupError, ValueError):
            continue
    # Give the OS a moment to release the socket before we re-bind it.
    time.sleep(0.5)


def _open_browser(url: str) -> None:
    """Open the URL in Chrome (preferred) or the default browser as fallback."""
    # Wait for the Flask app to actually bind so the first request does not race.
    time.sleep(1.0)
    try:
        subprocess.Popen(
            ["open", "-a", "Google Chrome", url],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        subprocess.Popen(["open", url])


def main() -> None:
    port = int(server.CFG["server"]["port"])
    _kill_existing(port)

    url = f"http://localhost:{port}/"
    print(f"Designer: serving on {url}", flush=True)
    threading.Thread(target=_open_browser, args=(url,), daemon=True).start()

    server.app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    main()
