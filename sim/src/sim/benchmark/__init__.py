"""Benchmark utilities for the residual RL policy.

PID and PID+RL share the same `BallPlateEnv` step path: passing ``action=0``
yields PID-only, while passing the trained policy's prediction yields the
residual sum. The benchmark exploits this so both controllers run on byte-
identical environments and seeds; the difference between the two columns
in the output table is the controller, never the conditions.
"""
