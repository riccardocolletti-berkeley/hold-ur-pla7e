"""Side-by-side dual-arm demo: PID vs PID+RL in one MuJoCo viewer.

Two UR5e arms, two plates, two balls in a single MjModel; both controllers
share the simulator clock and the reference trajectory, so any difference
between the two ball traces is the controller, not the conditions. Built
on top of the existing single-arm scene by duplicating it; ``sim.scene``
itself is left untouched.
"""
