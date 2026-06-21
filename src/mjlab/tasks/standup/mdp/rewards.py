"""Reward functions for the stay-stand (balance) policy.

Forked from the standup recovery rewards.py. Key differences from the
recovery version:

  - standup_progress REMOVED. That function only existed to give dense
    gradient during the get-up motion (height + uprightness from any fallen
    state). Stay-stand always starts upright, so there is no get-up arc to
    shape and the function is meaningless.

  - _is_standing gate REMOVED from hold_still and variable_posture. In
    recovery, these rewards returned zero while the robot was on the ground
    so the get-up motion wasn't penalized as "error". In stay-stand the
    robot is always at standing height (or the episode terminates via
    fell_over), so the gate is either vacuously true or actively harmful
    (it would suppress reward during the brief tilt after a push, exactly
    when the policy most needs a correction signal). Both terms now fire
    unconditionally every step.

  - variable_posture simplified to a single std tensor. The two-band
    (std_recovering / std_standing) design existed because recovery needed
    loose tolerance during ground-contact phases and tight tolerance once
    upright. With the gate gone there is only one operating regime, so
    __init__ and __call__ are simplified to a single std_values param.
    Config entries that used to pass std_recovering/std_standing now pass
    std_values (a single joint-name-pattern -> float dict).

  - body_angular_velocity_penalty and angular_momentum_penalty are now
    active (non-zero weights in env_cfgs.py). In recovery they were
    disabled because the tumbling/flailing phase produced angular velocities
    that made these penalties enormous and destabilised the value function.
    In stay-stand there is no tumbling phase, so the penalties are safe to
    enable. HOWEVER: both now use a bounded exp kernel
    (body_angular_velocity_penalty_bounded, angular_momentum_penalty_bounded)
    rather than raw squared magnitude, to prevent the same value-function
    divergence if the policy is ever pushed hard.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor
from mjlab.tasks.standup.mdp.terrain_utils import terrain_normal_from_sensors
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse
from mjlab.utils.lab_api.string import (
  resolve_matching_names_values,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def hold_still(
  env: ManagerBasedRlEnv,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward zero base linear + angular velocity. Fires unconditionally.

  In the recovery policy this was gated by _is_standing so that the
  necessary motion of getting up wasn't penalized as velocity error. In
  stay-stand the robot is always upright (a fall terminates the episode),
  so no gate is needed -- any non-zero velocity is genuine error and should
  be penalized immediately. Removing the gate also means the policy gets a
  correction signal during the brief tilt after a push, which is exactly
  when it most needs one.

  Both linear and angular velocity are penalized together (sum of squared
  components), bounded via an exp kernel so the penalty never blows up
  the value function.
  """
  asset: Entity = env.scene[asset_cfg.name]

  lin_error = torch.sum(torch.square(asset.data.root_link_lin_vel_b), dim=1)
  ang_error = torch.sum(torch.square(asset.data.root_link_ang_vel_b), dim=1)
  vel_error = lin_error + ang_error

  return torch.exp(-vel_error / std**2)


class upright:
  """Reward for keeping the base upright.

  Unchanged from locomotion's reward.py -- uprightness matters at least as
  much for standup (it's effectively the terminal goal) as for walking, and
  this implementation is already task-agnostic.

  Without ``terrain_sensor_names``, penalizes tilt relative to world up (correct for
  flat ground).

  With ``terrain_sensor_names``, penalizes tilt relative to the terrain surface normal.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self._terrain_sensor_names: tuple[str, ...] | None = cfg.params.get(
      "terrain_sensor_names"
    )
    self._debug_vis_enabled = True
    self._env = env
    self._asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    terrain_sensor_names: tuple[str, ...] | None = None,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]

    if asset_cfg.body_ids:
      body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # [B, N, 4]
      body_quat_w = body_quat_w.squeeze(1)  # [B, 4]
    else:
      body_quat_w = asset.data.root_link_quat_w  # [B, 4]

    if terrain_sensor_names is not None:
      terrain_normal = terrain_normal_from_sensors(env, terrain_sensor_names)  # [B, 3]
      target_b = quat_apply_inverse(body_quat_w, terrain_normal)  # [B, 3]
      xy_squared = torch.sum(torch.square(target_b[:, :2]), dim=1)
    else:
      gravity_w = asset.data.gravity_vec_w  # [3]
      projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)
      xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)

    return torch.exp(-xy_squared / std**2)

  def reset(self, env_ids: torch.Tensor) -> None:
    del env_ids  # Unused.

  def debug_vis(self, visualizer: DebugVisualizer) -> None:
    if not self._debug_vis_enabled or self._terrain_sensor_names is None:
      return

    env = self._env
    asset: Entity = env.scene[self._asset_cfg.name]

    env_indices = list(visualizer.get_env_indices(env.num_envs))
    if not env_indices:
      return

    terrain_normal = terrain_normal_from_sensors(env, self._terrain_sensor_names)
    if self._asset_cfg.body_ids:
      body_quat_w = asset.data.body_link_quat_w[:, self._asset_cfg.body_ids, :].squeeze(
        1
      )
    else:
      body_quat_w = asset.data.root_link_quat_w
    up_local = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand_as(
      body_quat_w[:, :3]
    )
    body_up_w = quat_apply(body_quat_w, up_local)

    positions = asset.data.root_link_pos_w.cpu().numpy()
    offset = np.array([0.0, 0.3, 0.0])
    terrain_normal_np = terrain_normal.cpu().numpy()
    body_up_np = body_up_w.cpu().numpy()
    scale = 0.25

    for i in env_indices:
      origin = positions[i] + offset
      # Terrain normal (magenta).
      visualizer.add_arrow(
        start=origin,
        end=origin + terrain_normal_np[i] * scale,
        color=(0.8, 0.2, 0.8, 0.8),
        width=0.01,
      )
      # Body up (orange).
      visualizer.add_arrow(
        start=origin,
        end=origin + body_up_np[i] * scale,
        color=(1.0, 0.5, 0.0, 0.8),
        width=0.01,
      )


def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Penalize self-collisions. Unchanged -- task-agnostic.

  When the sensor provides force history (from ``history_length > 0``),
  counts substeps where any contact force exceeds *force_threshold*.
  Falls back to the instantaneous ``found`` count otherwise.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    hit = (force_mag > force_threshold).any(dim=1)  # [B, H]
    return hit.sum(dim=-1).float()  # [B]
  assert data.found is not None
  return data.found.sum(dim=-1).float()


def body_angular_velocity_penalty(
  env: ManagerBasedRlEnv,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize excessive body angular velocities with a bounded exp kernel.

  In the recovery policy this used raw squared magnitude and was disabled
  (weight=0) because tumbling produced angular velocities of ~50-100 rad/s,
  making the penalty -250 to -1000 per step and destabilising the value
  function. In stay-stand there is no tumbling phase, so we can re-enable
  it. However, we still use exp(-x²/std²) rather than raw squared magnitude:
    - Bounded in [0, 1] -- the penalty can never blow up the value function
      no matter how chaotic the state after a hard push.
    - Still provides a strong gradient near zero (small angular velocities
      near std matter just as much as large ones).

  Only roll and pitch (xy) are penalized; yaw rotation is a separate concern
  (the robot should be able to yaw to settle, and yaw angular velocity alone
  doesn't indicate instability the same way pitch/roll does).

  std: characteristic angular velocity (rad/s) at which the reward falls to
  exp(-1) ≈ 0.37. A value of ~1.0 rad/s is a reasonable starting point --
  tighter than walking gait but loose enough not to penalize normal balance
  corrections.
  """
  asset: Entity = env.scene[asset_cfg.name]
  ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :]
  ang_vel = ang_vel.squeeze(1)
  ang_vel_xy = ang_vel[:, :2]  # Roll + pitch only.
  xy_sq = torch.sum(torch.square(ang_vel_xy), dim=1)
  return torch.exp(-xy_sq / std**2)


def angular_momentum_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  std: float,
) -> torch.Tensor:
  """Penalize whole-body angular momentum with a bounded exp kernel.

  Same reasoning as body_angular_velocity_penalty: was disabled in recovery
  because tumbling/flailing produced magnitudes that blew up the value
  function. Now re-enabled with exp(-|L|²/std²) so the penalty is bounded
  in [0, 1] regardless of what the policy does after a hard push.

  std: characteristic angular momentum magnitude at which reward falls to
  exp(-1) ≈ 0.37. Units depend on your robot's inertia tensor; start with
  std ~1.0 and adjust based on typical standing angular momentum logged
  during early training.
  """
  angmom_sensor: BuiltinSensor = env.scene[sensor_name]
  angmom = angmom_sensor.data
  angmom_magnitude_sq = torch.sum(torch.square(angmom), dim=-1)
  angmom_magnitude = torch.sqrt(angmom_magnitude_sq)
  env.extras["log"]["Metrics/angular_momentum_mean"] = torch.mean(angmom_magnitude)
  return torch.exp(-angmom_magnitude_sq / std**2)


class variable_posture:
  """Penalize deviation from default pose.

  Simplified from the recovery version, which had two std bands
  (std_recovering / std_standing) gated by _is_standing. That design
  existed because recovery needed loose tolerance during ground-contact
  phases (large joint excursions while pushing off the floor) and tight
  tolerance once upright. In stay-stand there is only one operating
  regime -- the robot is always upright -- so the gate and the second band
  are both removed. Config passes a single std_values dict instead of two.

  Uses per-joint standard deviations to control how much each joint can
  deviate from default pose. Smaller std = stricter gradient, larger std =
  more forgiving. Reward is: exp(-mean(error² / std²)).

  Map joint name patterns to std values, e.g. {".*knee.*": 0.25}.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    self.default_joint_pos = default_joint_pos

    _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)

    _, _, std_values = resolve_matching_names_values(
      data=cfg.params["std_values"],
      list_of_strings=joint_names,
    )
    self.std = torch.tensor(std_values, device=env.device, dtype=torch.float32)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std_values: dict,
    asset_cfg: SceneEntityCfg,
  ) -> torch.Tensor:
    del std_values  # Consumed in __init__; unused at call time.

    asset: Entity = env.scene[asset_cfg.name]
    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
    error_squared = torch.square(current_joint_pos - desired_joint_pos)

    return torch.exp(-torch.mean(error_squared / (self.std**2), dim=1))