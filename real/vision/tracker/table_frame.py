"""Table-frame estimator: ArUco markers → pixel-to-table homography.

Each ArUco detection returns four image corners per marker. Once the system
has observed all four corner markers cleanly at startup it caches each
marker's four corners *in the table frame* (by projecting their image
corners through a centroid-fit homography). From that point on the
homography is fit from the union of corner correspondences across whichever
markers happen to be visible. A single visible marker yields four point
pairs, enough to fully constrain a planar homography. Brief occlusion of
one or even three markers no longer drops the frame.

Bootstrap requirements:
  * At startup the plate is stationary at home and all four markers are
    visible. Each marker is promoted independently once it has been seen
    in :data:`_BOOTSTRAP_OBSERVATIONS` frames; its table-frame corner
    cache is the per-frame average over those observations.
  * A marker that never appears during the initial window is
    characterised opportunistically the first time it shows up alongside
    already-characterised markers.
"""

from __future__ import annotations

import cv2
import numpy as np

from vision.config import ARUCO_DICT_NAME, MARKER_POSITIONS_TABLE

#: Number of four-marker observations averaged before a marker's corner
#: positions are locked in. At 30 fps this is ~0.7 s, well inside the
#: controller's startup home window.
_BOOTSTRAP_OBSERVATIONS = 20


class TableFrame:
    """Owns the latest ArUco detection and the corresponding homography."""

    def __init__(self) -> None:
        dict_id = getattr(cv2.aruco, ARUCO_DICT_NAME)
        self._aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        self._aruco_params = cv2.aruco.DetectorParameters()
        self._detector = cv2.aruco.ArucoDetector(self._aruco_dict, self._aruco_params)

        self.homography: np.ndarray | None = None
        self._homography_inv: np.ndarray | None = None
        self.markers_found: int = 0

        # Per-marker bootstrap state. ``_marker_corners_obs`` accumulates
        # plate-frame corner stacks (shape (4, 2)) projected through the
        # centroid-fit homography; once a marker reaches
        # ``_BOOTSTRAP_OBSERVATIONS`` samples it is averaged and promoted
        # into ``_marker_corners_table``. After promotion, the marker
        # contributes four (pixel, table) correspondences per frame.
        self._marker_corners_obs: dict[int, list[np.ndarray]] = {}
        self._marker_corners_table: dict[int, np.ndarray] = {}

        # Latest detection cache so :meth:`draw_debug` can render the exact
        # corners used by the homography fit instead of re-detecting on a
        # different frame and producing a stale or distorted overlay.
        self._last_corners: tuple[np.ndarray, ...] | None = None
        self._last_ids: np.ndarray | None = None

    def update(self, frame: np.ndarray) -> bool:
        """Refit the homography from the current frame. Returns ``True`` on success."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners_list, ids, _ = self._detector.detectMarkers(gray)

        self._last_corners = corners_list
        self._last_ids = ids

        if ids is None:
            self.markers_found = 0
            return False
        ids = ids.flatten()

        # Collect every recognised marker visible this frame as
        # ``(id, image_corners)`` with ``image_corners`` of shape (4, 2).
        known: list[tuple[int, np.ndarray]] = []
        for i, marker_id in enumerate(ids):
            mid = int(marker_id)
            if mid in MARKER_POSITIONS_TABLE:
                known.append((mid, corners_list[i][0]))
        self.markers_found = len(known)

        if not known:
            return False

        # ----- Path A: corner-based fit using already-characterised markers --
        pts_pixel: list[np.ndarray] = []
        pts_table: list[np.ndarray] = []
        for mid, image_corners in known:
            cached = self._marker_corners_table.get(mid)
            if cached is None:
                continue
            for k in range(4):
                pts_pixel.append(image_corners[k])
                pts_table.append(cached[k])

        if len(pts_pixel) >= 4:
            H = self._fit_homography(pts_pixel, pts_table)
            if H is not None:
                # Use the (more accurate, corner-based) homography to keep
                # characterising any remaining uncharacterised markers seen
                # alongside the already-known ones.
                self._absorb_bootstrap_observations(H, known)
                return True
            # Fit failed numerically; fall through to centroid bootstrap.

        # ----- Path B: bootstrap centroid fit (today's logic) ----------------
        # Requires all four markers; same constraint as before, but only
        # for the first ~20 frames of the session.
        if len(known) < 4:
            return False
        centroid_pixel = np.array([img.mean(axis=0) for _, img in known], dtype=np.float64)
        centroid_table = np.array(
            [MARKER_POSITIONS_TABLE[mid] for mid, _ in known], dtype=np.float64
        )
        H = self._fit_homography(centroid_pixel, centroid_table)
        if H is None:
            return False
        self._absorb_bootstrap_observations(H, known)
        return True

    def pixel_to_table(self, u: float, v: float) -> tuple[float, float] | None:
        """Project a single pixel into the table frame, returning ``None`` if no homography is available."""
        if self.homography is None:
            return None
        pt = np.array([[[u, v]]], dtype=np.float64)
        out = cv2.perspectiveTransform(pt, self.homography)
        x, y = out[0, 0]
        return float(x), float(y)

    def table_to_pixel(self, x: float, y: float) -> tuple[float, float] | None:
        """Project a table-frame point (metres) back to image pixels."""
        if self._homography_inv is None:
            return None
        pt = np.array([[[x, y]]], dtype=np.float64)
        out = cv2.perspectiveTransform(pt, self._homography_inv)
        u, v = out[0, 0]
        return float(u), float(v)

    def draw_debug(self, frame: np.ndarray) -> np.ndarray:
        """Overlay the markers cached by the most recent :meth:`update` call.

        Reuses the detection from ``update`` rather than re-running the ArUco
        detector; saves work and guarantees the overlay matches the
        corners that were actually fed into the homography fit.
        """
        if self._last_ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, self._last_corners, self._last_ids)
        return frame

    # ------------------------------------------------------------- internals --

    def _fit_homography(
        self,
        pts_pixel: list[np.ndarray] | np.ndarray,
        pts_table: list[np.ndarray] | np.ndarray,
    ) -> np.ndarray | None:
        """Solve the planar homography and cache it together with its inverse."""
        pts_pixel_arr = np.asarray(pts_pixel, dtype=np.float64)
        pts_table_arr = np.asarray(pts_table, dtype=np.float64)
        H, _ = cv2.findHomography(pts_pixel_arr, pts_table_arr)
        if H is None:
            self.homography = None
            self._homography_inv = None
            return None
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            self.homography = None
            self._homography_inv = None
            return None
        self.homography = H
        self._homography_inv = H_inv
        return H

    def _absorb_bootstrap_observations(
        self,
        H: np.ndarray,
        known: list[tuple[int, np.ndarray]],
    ) -> None:
        """Refine each visible-but-uncharacterised marker's corner cache."""
        for mid, image_corners in known:
            if mid in self._marker_corners_table:
                continue
            corners_table = cv2.perspectiveTransform(
                image_corners.reshape(-1, 1, 2).astype(np.float64), H
            ).reshape(4, 2)
            obs = self._marker_corners_obs.setdefault(mid, [])
            obs.append(corners_table)
            if len(obs) >= _BOOTSTRAP_OBSERVATIONS:
                self._marker_corners_table[mid] = np.stack(obs, axis=0).mean(axis=0)
                del self._marker_corners_obs[mid]
