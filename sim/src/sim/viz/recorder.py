"""Offscreen video capture using MuJoCo's renderer.

Frames are rendered directly from the simulator (not screen-grabbed) and
written to an mp4 via imageio. ``imageio`` and a working ffmpeg backend are
declared in the optional ``[video]`` extra; importing this module without
them installed raises ``ImportError`` immediately so the failure is loud
and obvious.
"""

import os
from pathlib import Path

import imageio
import mujoco
import numpy as np


class Recorder:
    """Offscreen renderer that buffers frames in memory and writes an mp4
    on :meth:`save`. One renderer per instance; not thread-safe.
    """

    def __init__(
        self,
        model,
        data,
        output_path: str | Path,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
    ):
        self.model = model
        self.data = data
        self.output_path = str(output_path)
        self.fps = int(fps)
        self._frames: list[np.ndarray] = []

        out_dir = os.path.dirname(self.output_path) or "."
        os.makedirs(out_dir, exist_ok=True)

        # The renderer keeps its own GL state; one per Recorder is fine.
        self.renderer = mujoco.Renderer(model, height=int(height), width=int(width))

    def capture(self) -> None:
        """Render the current `data` into an in-memory frame."""
        self.renderer.update_scene(self.data)
        frame = self.renderer.render()
        self._frames.append(frame.copy())

    def save(self) -> None:
        """Write the buffered frames to the configured mp4 path and clear them."""
        if not self._frames:
            return
        writer = imageio.get_writer(self.output_path, fps=self.fps)
        for frame in self._frames:
            writer.append_data(frame)
        writer.close()
        print(f"Saved {len(self._frames)} frames to {self.output_path}")
        self._frames.clear()
