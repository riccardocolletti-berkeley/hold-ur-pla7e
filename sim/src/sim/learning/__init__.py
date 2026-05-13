"""RL training pipeline for the ball-on-plate task.

Modules:
    * `env`: Gymnasium environment wrapping the MuJoCo scene.
    * `randomize`: domain-randomization sampling and model mutation.
    * `reward`: Gaussian-tracking reward with proportional drop penalty.
    * `curriculum`: shape progression as a function of total training steps.
    * `callbacks`: SB3 callbacks for metrics, curriculum, scheduling, saving.
    * `train`: PPO training entry point (`python -m sim.learning.train`).
"""
