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
        # FREEZE THE POLICY STD.
        # Without this, Policy/mean_std drifts upward across training even at
        # entropy_coef=0 because PPO's surrogate gradient on log_std slowly
        # inflates it whenever any outlier action lands on positive advantage.
        # The agent_context (section 5, symptom 3 + section 6, item 3) documents
        # this exact failure: training plateaus for ~2200 iters at the bad local
        # optimum, the growing std finally provides enough random exploration to
        # stumble on a balance strategy (reward climbs 14 -> 55, ep length 100
        # -> 550), then the std keeps growing past the noise threshold of the
        # converged policy and the balance collapses (reward 55 -> 38). With
        # learn_std=False the std is fixed at init_std and the converged
        # policy stays converged. Trade-off: the random-exploration-driven
        # breakthrough takes more iterations to discover purely via mean
        # updates -- if it stalls, lower init_std to 0.15 (agent_context
        # section 6, item 2) or extend max_iterations.
        "learn_std": False,
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
    # Bumped 3000 -> 6000. With learn_std=False the breakthrough relies on
    # gradient-driven mean updates instead of random-exploration luck and
    # therefore takes longer to discover. The previous 3000-iter run
    # broke through at iter ~2200 (then collapsed); refinement past the
    # break-through point needs the additional budget.
    max_iterations=6_000,
  )
