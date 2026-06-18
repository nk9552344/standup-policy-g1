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
        # init_std=0.2 was still too large: training graphs showed Loss/
        # entropy *rising* monotonically (policy std growing) and
        # Metrics/mean_action_acc climbing, i.e. the policy becoming more
        # chaotic over training, not converging. Drop to 0.1 so the
        # initial policy is essentially "hold default pose with small
        # exploration" -- the std will grow on its own from surrogate
        # gradients if larger exploration genuinely helps.
        "init_std": 0.1,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=0.5,
      use_clipped_value_loss=True,
      clip_param=0.2,
      # entropy_coef=0.001 still pushed std upward against a weak reward
      # signal (entropy climbed from -4 to +2 across the last run). With
      # init_std=0.1 the policy already has enough exploration; setting
      # this to 0 removes any bias toward making the policy noisier.
      entropy_coef=0.0,
      # Adaptive KL schedule collapsed the LR to its hardcoded floor
      # (~1e-5) within 1 iteration on every previous run and never
      # recovered -- visible as a flat Loss/learning_rate line at 1.1e-5.
      # Switching to a fixed schedule decouples training from the noisy
      # early-training KL signal. 3e-4 is a conservative LR that keeps
      # updates small enough to be stable without the adaptive-KL
      # throttling pathology.
      learning_rate=3.0e-4,
      schedule="fixed",
      num_learning_epochs=5,
      num_mini_batches=4,
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,  # Unused under schedule="fixed" but kept for clarity.
      max_grad_norm=1.0,
    ),
    experiment_name="g1_standup",
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=30_000,
    # Hard-clip sampled actions to +/-3.0 (i.e. ~3 sigma at init_std=0.1,
    # but absolute -- even if policy std grows, sampled actions stay
    # bounded). This is a safety net against the catastrophic feedback
    # loop seen in the previous run: chaotic action samples -> chaotic
    # joint targets -> chaotic angular velocities -> huge reward magnitudes
    # -> value function divergence -> policy collapse. With actions
    # clipped, the physics state can't blow up no matter what the policy
    # does.
    clip_actions=3.0,
  )