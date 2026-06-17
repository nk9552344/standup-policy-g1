"""RL configuration for Unitree G1 standup task.

Forked from the velocity locomotion RL cfg. This file is almost entirely
training-infrastructure hyperparameters with no direct coupling to task
semantics (network architecture, PPO mechanics), so most values are carried
over unchanged. Two changes:

  - experiment_name: "g1_velocity" -> "g1_standup", since it was simply
    stale, not a deliberate choice.
  - entropy_coef: 0.01 -> 0.03. Locomotion mostly explores variations
    around a single stable walking gait; standup needs to discover a
    qualitatively different recovery strategy from a wide range of fallen
    starting states (face-down, face-up, on a side, mid-tumble after a
    push), so more exploration pressure early in training is a reasonable
    starting point. This is a starting guess, not a tuned value -- treat it
    as the first thing to revisit if training stalls into a single rigid
    "flailing" policy or, conversely, fails to converge at all.

Everything else (hidden_dims, activation, obs_normalization,
distribution_cfg, clip_param, gamma, lam, learning_rate, schedule,
desired_kl, max_grad_norm, num_steps_per_env, max_iterations, save_interval)
left unchanged -- no strong signal these need to differ for standup, and
guessing new values without tuning data would be worse than keeping known-
reasonable defaults. Revisit num_steps_per_env/max_iterations once you have
a sense of how standup's convergence behavior compares to locomotion's.
"""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def unitree_g1_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Unitree G1 standup task."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.03,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_standup",
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=30_000,
  )