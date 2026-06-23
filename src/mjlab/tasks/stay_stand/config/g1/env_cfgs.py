"""Unitree G1 stay-stand environment configurations."""

from mjlab.asset_zoo.robots import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.tasks.stay_stand.stay_stand_env_cfg import make_stay_stand_env_cfg

# Per-joint posture std. mdp.posture uses exp(-mean(error**2/std**2)). The
# mean across all 29 joints is unforgiving: a single tight joint with even a
# moderate error pulls the entire reward toward zero. waist_roll=0.2 and
# hip_roll=0.25 were the culprits behind the post-step-1000 reward collapse
# -- a 0.15 rad drift on waist_roll alone contributes 0.56 to the mean,
# enough that a few simultaneous joint perturbations cut posture by 60%+.
# A flat 0.5 across all joints keeps the positive reward firing reliably
# during exploration. Tighten selectively via curriculum once reward stabilizes.
_G1_POSTURE_STD = {r".*": 0.5}


def unitree_g1_stay_stand_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat stay-stand configuration."""
  cfg = make_stay_stand_env_cfg()

  # Flat-ground sim params.
  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64

  cfg.scene.entities = {"robot": get_g1_robot_cfg()}

  # Per-joint action scaling derived from G1 actuator specs.
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_ACTION_SCALE

  cfg.viewer.body_name = "torso_link"

  cfg.rewards["posture"].params["std"] = _G1_POSTURE_STD

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False

  return cfg
