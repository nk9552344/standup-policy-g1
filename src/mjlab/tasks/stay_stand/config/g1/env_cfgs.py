"""Unitree G1 stay-stand environment configurations."""

from mjlab.asset_zoo.robots import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.tasks.stay_stand.stay_stand_env_cfg import make_stay_stand_env_cfg

# Per-joint posture std. mdp.posture uses exp(-mean(error**2/std**2)). These
# values are intentionally loose so that the positive reward fires meaningfully
# during exploration. Tighten via curriculum once the policy converges.
_G1_POSTURE_STD = {
  r".*hip_pitch.*": 0.4,
  r".*hip_roll.*": 0.25,
  r".*hip_yaw.*": 0.3,
  r".*knee.*": 0.5,
  r".*ankle_pitch.*": 0.3,
  r".*ankle_roll.*": 0.25,
  r".*waist_yaw.*": 0.3,
  r".*waist_roll.*": 0.2,
  r".*waist_pitch.*": 0.25,
  r".*shoulder_pitch.*": 0.5,
  r".*shoulder_roll.*": 0.5,
  r".*shoulder_yaw.*": 0.5,
  r".*elbow.*": 0.5,
  r".*wrist.*": 0.6,
}


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
