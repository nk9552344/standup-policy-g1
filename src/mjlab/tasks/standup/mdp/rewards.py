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


def base_height_reward(
  env: ManagerBasedRlEnv,
  target_height: float,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward the pelvis being at standing height.

  Returns exp(-((height - target_height) / std)²): 1.0 at exactly target_height,
  decaying symmetrically below and above. For stay-stand this fills the gap
  between the fell_over termination threshold (0.45 m) and the actual standing
  height (~0.72 m): without this reward a policy can earn full upright_gated
  score while slowly squatting downward, since upright orientation alone does
  not penalise downward displacement.

  target_height: desired pelvis z in metres. HOME_KEYFRAME with bent knees
    gives an actual pelvis height of ~0.72 m (not 0.783 m which is the raw
    MJCF spawn z). Override per-robot.
  std: kernel width in metres. 0.10 m gives exp(-1)≈0.37 when 10 cm below
    target and approaches 0 near the 0.45 m fell_over floor.
  """
  asset: Entity = env.scene[asset_cfg.name]
  height = asset.data.root_link_pos_w[:, 2]  # [B] world-frame pelvis z
  error = height - target_height             # [B]
  return torch.exp(-(error**2) / std**2)


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

  Returns penalty intensity in [0, 1]: 0 at rest and saturating to 1 at high
  angular velocity, computed as ``1 - exp(-xy²/std²)``. Combined with the
  negative weight in env cfg, this gives a real penalty that grows with
  motion and is bounded so it can't blow up the value function.

  HISTORY: the previous implementation returned ``exp(-xy²/std²)`` (highest
  at zero, lowest at chaos). Combined with a negative weight, that
  *rewarded* high angular velocity (cost was largest when still, zero when
  spinning), which actively trained the torso-swinging balance strategy
  this term was meant to suppress. Every other ``exp(-x²/std²)`` reward in
  this file (upright, pose, hold_still, base_height, upright_gated) uses a
  positive weight because that kernel shape is a stability *reward*, not a
  penalty. Fixed by flipping the kernel here and in angular_momentum_penalty.

  Only roll and pitch (xy) are penalized; yaw rotation is a separate concern
  (the robot should be able to yaw to settle, and yaw angular velocity alone
  doesn't indicate instability the same way pitch/roll does).

  std: characteristic angular velocity (rad/s) at which the penalty reaches
  ``1 - exp(-1) ≈ 0.63``. A value of ~1.0 rad/s is a reasonable starting
  point -- tighter than walking gait but loose enough not to fully saturate
  the penalty during normal balance corrections (which run at ~0.2-0.4 rad/s).
  """
  asset: Entity = env.scene[asset_cfg.name]
  ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :]
  ang_vel = ang_vel.squeeze(1)
  ang_vel_xy = ang_vel[:, :2]  # Roll + pitch only.
  xy_sq = torch.sum(torch.square(ang_vel_xy), dim=1)
  return 1.0 - torch.exp(-xy_sq / std**2)


def angular_momentum_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  std: float,
) -> torch.Tensor:
  """Penalize whole-body angular momentum with a bounded exp kernel.

  Returns penalty intensity in [0, 1] computed as ``1 - exp(-|L|²/std²)``: 0
  at rest and saturating to 1 at high angular momentum. Combined with the
  negative weight in env cfg this is a bounded, correctly-oriented penalty.

  HISTORY: see body_angular_velocity_penalty -- the previous form
  ``exp(-|L|²/std²)`` was used with a negative weight, which inverted the
  intended cost (rewarding chaos, penalising stillness). Flipped to
  ``1 - exp(...)`` so the function name and semantics match.

  std: characteristic angular momentum magnitude at which the penalty
  reaches ``1 - exp(-1) ≈ 0.63``. Units depend on the robot's inertia
  tensor; start with std ~1.0 and adjust based on the
  ``Metrics/angular_momentum_mean`` log during early training.
  """
  angmom_sensor: BuiltinSensor = env.scene[sensor_name]
  angmom = angmom_sensor.data
  angmom_magnitude_sq = torch.sum(torch.square(angmom), dim=-1)
  angmom_magnitude = torch.sqrt(angmom_magnitude_sq)
  env.extras["log"]["Metrics/angular_momentum_mean"] = torch.mean(angmom_magnitude)
  return 1.0 - torch.exp(-angmom_magnitude_sq / std**2)


def feet_bearing_weight(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  bodyweight_n: float = 225.0,
) -> torch.Tensor:
  """Reward feet bearing the robot's weight through ground contact.

  Reads the net ground-reaction force from a ContactSensor configured with
  reduce="netforce" and two primaries (left and right foot subtrees). Returns
  tanh((|F_left| + |F_right|) / bodyweight_n), which is:
    - 0.0  when no foot touches the ground (legs fully abandoned)
    - ~0.46 when only one foot bears bodyweight
    - ~0.76 when both feet together bear full bodyweight

  This directly incentivizes leg use: the robot must push through its legs
  to generate ground reaction force. When the policy learns to balance using
  only the upper body (the "arm-flapping" failure mode), foot forces drop to
  zero and this reward collapses, opposing that local optimum.

  The sensor must expose a ``force`` field with shape [B, 2, 3] (one net
  force vector per foot, in world frame). For G1 with two primaries
  (left_ankle_roll_link, right_ankle_roll_link) and reduce="netforce" this
  is satisfied out-of-the-box.

  bodyweight_n: normalisation constant in Newtons. Set to robot_mass * 9.8.
    For G1 (~23 kg): 225 N. tanh saturates near 1.0 at 2× bodyweight.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  assert data.force is not None, (
    "feet_bearing_weight requires the contact sensor to have fields=('force', ...)"
  )
  # data.force: [B, 2, 3] — left foot at index 0, right foot at index 1.
  force_left = torch.norm(data.force[:, 0, :], dim=-1)   # [B]
  force_right = torch.norm(data.force[:, 1, :], dim=-1)  # [B]
  total_force = force_left + force_right                  # [B]
  return torch.tanh(total_force / bodyweight_n)


def upright_with_feet_gate(
  env: ManagerBasedRlEnv,
  std: float,
  sensor_name: str,
  bodyweight_n: float = 225.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Upright reward gated by ground-reaction force — the primary balance signal.

  Returns upright_score × feet_gate where:
    upright_score = exp(-xy²/std²) ∈ [0, 1]   (1.0 when perfectly vertical)
    feet_gate     = clamp(force / bodyweight_n, 0, 1) ∈ [0, 1]
                    (1.0 when both feet bear full bodyweight, 0.0 when airborne)

  The gate makes it IMPOSSIBLE to earn the primary balance reward without
  bearing weight through the feet. This prevents the "arm-only balance"
  failure mode (~1500 iters) where the policy earns full upright reward by
  flapping arms while freezing legs, because:
    - arm-only balance:  feet leave ground → gate → 0 → reward → 0
    - proper balance:    feet bear weight  → gate → 1 → reward = upright_score

  The feet_gate uses a linear clamp (not tanh) so it reaches exactly 1.0 at
  bodyweight_n Newtons total force, giving a clear full-reward target for the
  policy to aim at.

  asset_cfg: body_names=() (root link = pelvis) for G1. Do NOT set this to
    "torso_link" — torso tracking allows the waist-bending compensation local
    optimum (torso stays vertical by bending the waist while the pelvis falls,
    earning full upright reward without any leg corrections). body_names=()
    → asset_cfg.body_ids is falsy → uses root_link_quat_w (pelvis).
    Set per-robot.
  sensor_name: ContactSensor with reduce="netforce" and two primaries (left
    and right foot subtrees). data.force shape must be [B, 2, 3].
  bodyweight_n: Normalisation constant in Newtons (robot_mass × 9.8).
    Gate = 1.0 at this force. For G1 (~23 kg): 225 N. Use half-bodyweight
    (112.5 N) if stepping should not be penalised (gate=1 when one foot
    bears full load).
  """
  # --- Uprightness component ---
  asset: Entity = env.scene[asset_cfg.name]
  if asset_cfg.body_ids:
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # [B, 1, 4]
    body_quat_w = body_quat_w.squeeze(1)  # [B, 4]
  else:
    body_quat_w = asset.data.root_link_quat_w  # [B, 4]
  gravity_w = asset.data.gravity_vec_w  # [3]
  projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)  # [B, 3]
  xy_sq = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)  # [B]
  upright_score = torch.exp(-xy_sq / std**2)  # [B], in [0, 1]

  # --- Foot-contact gate ---
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  assert data.force is not None, (
    "upright_with_feet_gate requires the contact sensor to have fields=('force', ...)"
  )
  # data.force: [B, 2, 3] — left foot at index 0, right foot at index 1.
  force_left = torch.norm(data.force[:, 0, :], dim=-1)   # [B]
  force_right = torch.norm(data.force[:, 1, :], dim=-1)  # [B]
  feet_gate = torch.clamp((force_left + force_right) / bodyweight_n, 0.0, 1.0)  # [B]

  return upright_score * feet_gate


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


class ankle_corrective:
  """Directly reward ankle_pitch joints being in the corrective direction for pelvis tilt.

  ROOT CAUSE addressed: hip joint corrections change pelvis orientation in ONE
  step (feet fixed → hip flex → pelvis tilts → upright_gated improves), while
  ankle corrections require 3–5 physics steps via CoP shift → GRF change →
  pelvis acceleration. PPO correctly prefers the faster signal (hip strategy).
  No amount of upright_gated tuning fixes this timing asymmetry.

  This reward fires IN THE SAME STEP that the ankle moves, giving the policy
  immediate dense gradient for ankle corrections BEFORE the physics delay:

    Forward tilt  (projected_gravity_b[0] > 0):
      correct ankle response = dorsiflexion (ankle_pitch < HOME = −0.2 rad)
      reward is positive when −tilt × (ankle − HOME) > 0  ✓

    Backward tilt (projected_gravity_b[0] < 0):
      correct ankle response = plantarflexion (ankle_pitch > HOME)
      reward is positive when −tilt × (ankle − HOME) > 0  ✓

    Upright (tilt ≈ 0): reward ≈ 0; no incentive to move ankles needlessly ✓
    Wrong direction (ankle dorsiflexed while tilting backward): reward = 0
      (clamped, not penalised — the falling upright_gated provides enough
      negative signal already)

  reward = clamp(−tilt_x × mean(ankle_pitch − HOME_ankle) / std, 0, 1)

  Both lateral tilt (projected_gravity_b[1]) and ankle_roll are included
  with the same formula to also incentivise lateral ankle corrections.

  std: typical product magnitude at which reward saturates to 1.0.
    ≈ tilt_at_operating_range × typical_ankle_deviation.
    At 10° tilt (tilt_x ≈ 0.17) and 0.15 rad ankle correction:
      product = 0.17 × 0.15 = 0.026 → reward = clamp(0.026/0.025, 0, 1) ≈ 1
    Use std ≈ 0.025.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    # Find ankle pitch and roll joint indices.
    self._pitch_ids, _ = asset.find_joints([".*_ankle_pitch_joint"])
    self._roll_ids, _ = asset.find_joints([".*_ankle_roll_joint"])
    default_pos = asset.data.default_joint_pos
    assert default_pos is not None
    self._home_pitch = default_pos[:, self._pitch_ids]  # [B, 2]
    self._home_roll = default_pos[:, self._roll_ids]    # [B, 2]

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std: float,
    asset_cfg: SceneEntityCfg,
  ) -> torch.Tensor:
    del asset_cfg  # joint ids cached in __init__
    asset: Entity = env.scene["robot"]

    # Forward (x) and lateral (y) tilt in pelvis body frame.
    gravity_b = asset.data.projected_gravity_b  # [B, 3]
    tilt_x = gravity_b[:, 0]  # [B], positive = forward tilt
    tilt_y = gravity_b[:, 1]  # [B], positive = rightward tilt

    # Ankle deviations from HOME.
    ankle_pitch = asset.data.joint_pos[:, self._pitch_ids]  # [B, 2]
    ankle_roll = asset.data.joint_pos[:, self._roll_ids]    # [B, 2]
    dev_pitch = (ankle_pitch - self._home_pitch).mean(dim=1)  # [B]
    dev_roll = (ankle_roll - self._home_roll).mean(dim=1)     # [B]

    # Corrective alignment score: positive when ankle is in the right place.
    # Forward tilt → dorsiflexion (dev_pitch < 0) → −tilt_x × dev_pitch > 0 ✓
    # Rightward tilt → ankle_roll correction (dev_roll < 0 for right ankle) ✓
    corrective = (-tilt_x * dev_pitch + -tilt_y * dev_roll) * 0.5  # [B]

    return torch.clamp(corrective / std, 0.0, 1.0)

  def reset(self, env_ids: torch.Tensor) -> None:
    del env_ids