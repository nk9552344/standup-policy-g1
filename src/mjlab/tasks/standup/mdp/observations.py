"""Useful methods for MDP observations.

Forked from the velocity locomotion observations.py. Most of this file is
unchanged: these are generic proprioceptive/sensor readers with no coupling
to velocity commands or gait, so they apply just as well to a standup/
recovery policy. The one addition is `base_height`, since knowing how far
through the standup motion the robot is (height above ground) is central
to this task in a way it isn't for steady-state walking.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, RayCastSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


##
# Root state.
##


def base_lin_vel(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Unchanged. Still essential: the policy needs to know its current
  velocity whether that means walking speed or how fast it's still
  tumbling/settling after a fall or push."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_lin_vel_b


def base_ang_vel(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Unchanged. Same reasoning as base_lin_vel."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_ang_vel_b


def projected_gravity(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Unchanged. Arguably the single most important observation for
  standup -- tells the policy its orientation relative to gravity (e.g.
  lying face-down vs. face-up vs. on its side vs. upright), which a
  locomotion policy mostly takes for granted as "always roughly upright"."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.projected_gravity_b


def base_height(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Base height above the world origin/ground plane.

  New for standup: nothing in the original file exposes height directly
  (height_scan exists, but it's a terrain raycast sensor reading, not a
  simple root-height signal, and requires a sensor to be present in the
  scene). Height is one of the clearest signals of "how far through
  standing up am I", so it's useful as a direct, always-available
  observation alongside projected_gravity.

  If your terrain isn't flat at z=0, prefer height_scan or subtract a
  terrain-relative offset instead -- this returns raw world-frame z.
  """
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_pos_w[:, 2:3]


##
# Joint state.
##


def joint_pos_rel(
  env: ManagerBasedRlEnv,
  biased: bool = False,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Unchanged. Generic proprioception."""
  asset: Entity = env.scene[asset_cfg.name]
  default_joint_pos = asset.data.default_joint_pos
  assert default_joint_pos is not None
  jnt_ids = asset_cfg.joint_ids
  joint_pos = asset.data.joint_pos_biased if biased else asset.data.joint_pos
  return joint_pos[:, jnt_ids] - default_joint_pos[:, jnt_ids]


def joint_vel_rel(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Unchanged. Generic proprioception."""
  asset: Entity = env.scene[asset_cfg.name]
  default_joint_vel = asset.data.default_joint_vel
  assert default_joint_vel is not None
  jnt_ids = asset_cfg.joint_ids
  return asset.data.joint_vel[:, jnt_ids] - default_joint_vel[:, jnt_ids]


##
# Actions.
##


def last_action(env: ManagerBasedRlEnv, action_name: str | None = None) -> torch.Tensor:
  """Unchanged. Generic."""
  if action_name is None:
    return env.action_manager.action
  return env.action_manager.get_term(action_name).raw_action


##
# Commands.
##


def generated_commands(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Unchanged. Implementation is already command-agnostic -- it just
  forwards whatever command_manager.get_command(command_name) returns.
  Point command_name at your StandStillCommand term (or whatever you named
  it) in the observation manager config, and this works without
  modification: it'll return the all-zero hold-still target instead of a
  sampled velocity."""
  command = env.command_manager.get_command(command_name)
  assert command is not None
  return command


##
# Sensors.
##


def builtin_sensor(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Unchanged. Generic."""
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, BuiltinSensor)
  return sensor.data


def projected_gravity_from_sensor(
  env: ManagerBasedRlEnv, sensor_name: str
) -> torch.Tensor:
  """Unchanged. Same value as projected_gravity but from an IMU site
  sensor rather than root orientation -- useful if you're modeling IMU
  mounting/site pose randomization, equally relevant to standup."""
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, BuiltinSensor)
  return -sensor.data


def height_scan(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  offset: float = 0.0,
  miss_value: float | None = None,
) -> torch.Tensor:
  """Unchanged. Still useful if standing up on uneven/non-flat terrain --
  arguably more useful here than in locomotion, since you may want the
  policy to sense ground contact/shape around it while still lying down,
  before any single root-height reading is informative on sloped terrain.

  Returns the height of the sensor frame above each hit point.
  Supports multi-frame sensors: each ray uses its own frame's Z.
  """
  sensor: RayCastSensor = env.scene[sensor_name]
  if miss_value is None:
    miss_value = sensor.cfg.max_distance

  data = sensor.data
  F, N = sensor.num_frames, sensor.num_rays_per_frame
  B = data.distances.shape[0]

  frame_z = data.frame_pos_w[:, :, 2:3]  # [B, F, 1]
  hit_z = data.hit_pos_w[..., 2].view(B, F, N)  # [B, F, N]
  heights = (frame_z - hit_z - offset).view(B, F * N)

  miss_mask = data.distances < 0
  return torch.where(miss_mask, torch.full_like(heights, miss_value), heights)