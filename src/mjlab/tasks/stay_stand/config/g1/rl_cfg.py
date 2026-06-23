"""RL configuration for Unitree G1 stay-stand task."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def unitree_g1_stay_stand_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Unitree G1 stay-stand task."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(256, 128, 64),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        # Small init_std: with per-joint action_scale ~0.05-0.15 rad, init_std=1.0
        # produces joint perturbations that immediately topple G1. Start tight.
        "init_std": 0.3,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(256, 128, 64),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      # entropy_coef=0.001 still inflated Policy/mean_std (0.30 -> 0.325+)
      # against a weak balance gradient and triggered the post-step-1000
      # reward collapse. Zero removes the bias toward growing std; the
      # init_std=0.3 Gaussian still provides plenty of exploration.
      entropy_coef=0.0,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_stay_stand",
    save_interval=50,
    # 24 steps * 0.02 s = 0.48 s of rollout, but episodes run 60-84 steps
    # (1.2-1.7 s) before termination. The critic was learning value targets
    # on sub-second fragments while balance outcomes played out over a
    # full second-plus. 128 steps = 2.56 s covers a full episode, so the
    # advantage estimator can attribute survival to actions.
    num_steps_per_env=128,
    max_iterations=3_000,
  )
