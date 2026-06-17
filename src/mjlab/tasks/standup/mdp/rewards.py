# DONE
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor
from mjlab.tasks.velocity.mdp.terrain_utils import terrain_normal_from_sensors
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse
from mjlab.utils.lab_api.string import (
  resolve_matching_names_values,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _is_standing(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  min_height: float,
  min_uprightness: float,
) -> torch.Tensor:
  """Shared 'has the robot finished standing up' gate.

  Combines base height and uprightness (projected-gravity z-component) so
  that downstream terms which only make sense once standing -- velocity
  hold-still, tight posture tolerance -- don't fire during the standup
  motion itself and fight the recovery behavior.

  uprightness is cos(tilt from vertical): 1.0 means perfectly upright,
  -1.0 means upside down. Using gravity projection (not just quat) keeps
  this consistent with the `upright` reward below, rather than introducing
  a second way of measuring tilt.
  """
  asset: Entity = env.scene[asset_cfg.name]
  height = asset.data.root_link_pos_w[:, 2]

  gravity_w = asset.data.gravity_vec_w  # [3], points down, unit norm
  root_quat_w = asset.data.root_link_quat_w
  projected_gravity_b = quat_apply_inverse(root_quat_w, gravity_w)  # [B, 3]
  # gravity_vec_w points down (e.g. [0, 0, -1]); when upright, the body-frame
  # z-axis is anti-parallel to gravity, so -projected_gravity_b[:, 2] -> 1.0.
  uprightness = -projected_gravity_b[:, 2]

  return (height > min_height) & (uprightness > min_uprightness)


def standup_progress(
  env: ManagerBasedRlEnv,
  target_height: float,
  height_weight: float = 1.0,
  uprightness_weight: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Dense shaping reward toward standing, from any starting state.

  Locomotion's reward.py has nothing playing this role: its terms only
  reward good behavior once already walking. Standup additionally needs a
  signal that's informative *during* the recovery motion -- whether starting
  from lying down, mid-fall after a push, or mid-tumble after a run stopped
  abruptly -- so there is gradient to climb well before the robot is
  upright.

  Returns a value in roughly [-uprightness_weight, height_weight +
  uprightness_weight]: height progress is clamped to [0, 1] (no bonus for
  overshooting target_height), uprightness ranges [-1, 1] (penalizes being
  upside down, rewards being upright) so the two combine into one dense
  per-step reward.
  """
  asset: Entity = env.scene[asset_cfg.name]
  height = asset.data.root_link_pos_w[:, 2]
  height_progress = torch.clamp(height / target_height, min=0.0, max=1.0)

  gravity_w = asset.data.gravity_vec_w
  root_quat_w = asset.data.root_link_quat_w
  projected_gravity_b = quat_apply_inverse(root_quat_w, gravity_w)
  uprightness = -projected_gravity_b[:, 2]

  env.extras["log"]["Metrics/standup_height_progress_mean"] = torch.mean(
    height_progress
  )
  env.extras["log"]["Metrics/standup_uprightness_mean"] = torch.mean(uprightness)

  return height_weight * height_progress + uprightness_weight * uprightness


def hold_still(
  env: ManagerBasedRlEnv,
  std: float,
  min_standing_height: float,
  min_standing_uprightness: float = 0.8,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward zero base linear+angular velocity, only once standing.

  Replaces track_linear_velocity/track_angular_velocity's role of "track the
  commanded velocity". Standup has no velocity command -- the implicit
  target is always zero -- so this combines both into one term and gates
  the whole reward to 0 while still recovering, rather than gating an error
  term, so the policy gets no signal at all (positive or negative) about
  velocity until it's actually standing. Before that gate, the necessary
  motion of getting up would otherwise be penalized as "error".
  """
  asset: Entity = env.scene[asset_cfg.name]
  standing = _is_standing(
    env, asset_cfg, min_standing_height, min_standing_uprightness
  )

  lin_error = torch.sum(torch.square(asset.data.root_link_lin_vel_b), dim=1)
  ang_error = torch.sum(torch.square(asset.data.root_link_ang_vel_b), dim=1)
  vel_error = lin_error + ang_error

  reward = torch.exp(-vel_error / std**2)
  return torch.where(standing, reward, torch.zeros_like(reward))


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
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize excessive body angular velocities. Unchanged -- task-agnostic.

  Note: for standup specifically, consider whether you want this active
  during the early tumble/recovery phase, where some roll/pitch angular
  velocity is an unavoidable part of e.g. rolling from prone to a crouch.
  If it fights the recovery motion in practice, gate it with `_is_standing`
  the same way `hold_still` is gated.
  """
  asset: Entity = env.scene[asset_cfg.name]
  ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :]
  ang_vel = ang_vel.squeeze(1)
  ang_vel_xy = ang_vel[:, :2]  # Don't penalize z-angular velocity.
  return torch.sum(torch.square(ang_vel_xy), dim=1)


def angular_momentum_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Penalize whole-body angular momentum. Unchanged -- task-agnostic.

  For locomotion this encourages natural arm swing; for standup it
  discourages flailing/wild limb motion during recovery, which is equally
  desirable.
  """
  angmom_sensor: BuiltinSensor = env.scene[sensor_name]
  angmom = angmom_sensor.data
  angmom_magnitude_sq = torch.sum(torch.square(angmom), dim=-1)
  angmom_magnitude = torch.sqrt(angmom_magnitude_sq)
  env.extras["log"]["Metrics/angular_momentum_mean"] = torch.mean(angmom_magnitude)
  return angmom_magnitude_sq


class variable_posture:
  """Penalize deviation from default pose with recovery-phase-dependent tolerance.

  Reframed from locomotion's command-speed-banded version (standing/walking/
  running, driven by commanded velocity) since standup has no velocity
  command. Instead bands are driven by `_is_standing`: tight tolerance once
  standing, loose tolerance while still recovering (lying/tumbling/getting
  up), so the policy isn't punished for the large joint excursions needed to
  push up from the ground, push off after a fall, or recover from a push
  mid-standup.

  Uses per-joint standard deviations to control how much each joint can
  deviate from default pose. Smaller std = stricter, larger std = more
  forgiving. Reward is: exp(-mean(error² / std²)).

  Map joint name patterns to std values, e.g. {".*knee.*": 0.35}.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    self.default_joint_pos = default_joint_pos

    _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)

    _, _, std_recovering = resolve_matching_names_values(
      data=cfg.params["std_recovering"],
      list_of_strings=joint_names,
    )
    self.std_recovering = torch.tensor(
      std_recovering, device=env.device, dtype=torch.float32
    )

    _, _, std_standing = resolve_matching_names_values(
      data=cfg.params["std_standing"],
      list_of_strings=joint_names,
    )
    self.std_standing = torch.tensor(
      std_standing, device=env.device, dtype=torch.float32
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std_recovering,
    std_standing,
    asset_cfg: SceneEntityCfg,
    min_standing_height: float,
    min_standing_uprightness: float = 0.8,
  ) -> torch.Tensor:
    del std_recovering, std_standing  # Unused; consumed in __init__.

    asset: Entity = env.scene[asset_cfg.name]
    standing = _is_standing(
      env, asset_cfg, min_standing_height, min_standing_uprightness
    )

    std = torch.where(
      standing.unsqueeze(1), self.std_standing, self.std_recovering
    )

    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
    error_squared = torch.square(current_joint_pos - desired_joint_pos)

    return torch.exp(-torch.mean(error_squared / (std**2), dim=1))