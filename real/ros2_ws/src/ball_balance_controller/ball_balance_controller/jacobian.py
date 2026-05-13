"""Geometric Jacobian provider for the UR arm + plate adapter.

Wraps an ikpy chain so callers can request ``J(q)`` without touching
ikpy's chain-padding convention. Only the rotational block is exposed
(the ball-on-plate task only needs angular velocity). ikpy ships FK but
no Jacobian, so each column is recovered by centred finite differences
on FK plus a vee-operator read of the antisymmetric residual.
"""

from __future__ import annotations

from pathlib import Path

import ikpy.chain
import numpy as np

#: Joint perturbation used by the centred finite-difference Jacobian.
_FD_EPSILON = 1.0e-5

#: Active-joint count expected for a UR arm; sanity-checked after auto-mask.
_EXPECTED_JOINTS = 6


def _mask_from_joint_types(chain: ikpy.chain.Chain) -> list[bool]:
    """Return one boolean per chain link, True for revolute/prismatic joints."""
    return [getattr(link, "joint_type", "fixed") != "fixed" for link in chain.links]


class JacobianProvider:
    """Compute the rotational Jacobian of the plate at the current joint configuration."""

    def __init__(
        self,
        urdf_path: Path | str,
        base_elements: list[str] | None = None,
        active_links_mask: list[bool] | None = None,
    ):
        self.chain = ikpy.chain.Chain.from_urdf_file(
            str(urdf_path),
            base_elements=base_elements or ["base_link"],
        )
        mask = active_links_mask or _mask_from_joint_types(self.chain)
        self.chain.active_links_mask = np.asarray(mask, dtype=bool)
        self.n_links = len(self.chain.links)
        self.n_joints = int(self.chain.active_links_mask.sum())

        if active_links_mask is None and self.n_joints != _EXPECTED_JOINTS:
            raise RuntimeError(
                f"URDF at {urdf_path} produced {self.n_joints} active joints "
                f"after auto-mask; expected {_EXPECTED_JOINTS} for a UR arm. "
                f"Pass an explicit ``active_links_mask`` if this is intentional."
            )

        # Cached scratch state for the hot-loop FD Jacobian. Recomputing these
        # inside ``rotational`` allocates ~6× per call, ~3000 allocs/sec at
        # 500 Hz; precomputing the index list and identity matrix saves real
        # microseconds on the per-tick budget.
        self._active_indices: tuple[int, ...] = tuple(
            i for i, active in enumerate(self.chain.active_links_mask) if active
        )
        self._I3 = np.eye(3)
        self._q_perturb = np.zeros(self.n_links, dtype=float)

    # ------------------------------------------------------------- helpers --

    def _ros_to_ikpy(self, q_ros: np.ndarray) -> np.ndarray:
        """Pad a six-element ROS joint vector to the chain's full length."""
        out = np.zeros(self.n_links, dtype=float)
        out[list(self._active_indices)] = q_ros
        return out

    def _ikpy_to_ros(self, q_ikpy: np.ndarray) -> np.ndarray:
        """Strip the chain-padding indices and return the six joint values."""
        return q_ikpy[list(self._active_indices)].astype(float, copy=True)

    # ----------------------------------------------------- forward kinematics --

    def forward_kinematics(self, q_ros: np.ndarray) -> np.ndarray:
        """Return the 4×4 transform of the chain's tip at ``q_ros``."""
        return self.chain.forward_kinematics(self._ros_to_ikpy(q_ros))

    # ----------------------------------------------------------- jacobian --

    def rotational(self, q_ros: np.ndarray, epsilon: float = _FD_EPSILON) -> np.ndarray:
        """Return the 3x6 angular Jacobian at ``q_ros`` in the world frame.

        Each column is the contribution of one joint's unit angular rate to
        the tip angular velocity, computed by centred finite differences on
        ``forward_kinematics``.
        """
        # Reuse one padded buffer and restore the entry after each FK pair
        # instead of allocating two copies per column.
        q = self._q_perturb
        q[:] = 0.0
        q[list(self._active_indices)] = q_ros

        n_active = len(self._active_indices)
        inv_two_eps = 0.5 / epsilon
        jac = np.zeros((3, n_active), dtype=float)
        for col, link_idx in enumerate(self._active_indices):
            original = q[link_idx]
            q[link_idx] = original + epsilon
            r_plus = self.chain.forward_kinematics(q)[:3, :3]
            q[link_idx] = original - epsilon
            r_minus = self.chain.forward_kinematics(q)[:3, :3]
            q[link_idx] = original

            # The relative rotation R+ R-^T is approximately I + (2ε)·[ω]×.
            # Reading the antisymmetric part out via the vee operator gives
            # the angular velocity contribution of joint `link_idx`.
            delta = r_plus @ r_minus.T - self._I3
            jac[0, col] = (delta[2, 1] - delta[1, 2]) * 0.5 * inv_two_eps
            jac[1, col] = (delta[0, 2] - delta[2, 0]) * 0.5 * inv_two_eps
            jac[2, col] = (delta[1, 0] - delta[0, 1]) * 0.5 * inv_two_eps

        return jac
