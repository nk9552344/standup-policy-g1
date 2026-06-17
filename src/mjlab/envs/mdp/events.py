"""Useful methods for MDP events.

Forked from the velocity locomotion events.py. Almost nothing here actually
needed to change: event terms describe *what physically happens to the
robot* (resets, pushes, disturbances), not what it's being asked to do, so
this machinery was never locomotion-specific to begin with. The only
addition is `reset_fallen_state`, a thin convenience wrapper around
`reset_root_state_uniform` that makes "spawn fallen / mid-motion" configs
self-documenting instead of relying on someone reading wide, unlabeled
pose/velocity ranges in a task cfg and guessing the intent.

Everything else below -- randomize_terrain, reset_scene_to_default,
reset_root_state_uniform, reset_root_state_from_flat_patches,
reset_joints_by_offset, push_by_setting_velocity,
apply_external_force_torque, apply_body_impulse -- is unchanged and is
exactly the mechanism you want for "robot was running and stopped" (reset-
time velocity) and "someone pushed it while standing up" (push_by_setting_
velocity / apply_body_impulse, mode="interval" or "step", firing mid-
episode). Use these directly rather than hand-rolling separate logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_from_euler_xyz,
  quat_mul,
  sample_uniform,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_SE3_KEYS = ("x", "y", "z", "roll", "pitch", "yaw")


def _sample_se3_range(
  range_dict: dict[str, tuple[float, float]] | None,
  shape: tuple[int, ...],
  device: str,
) -> torch.Tensor:
  """Sample uniform ``[x, y, z, roll, pitch, yaw]`` offsets.

  ``range_dict`` maps any subset of those keys to ``(min, max)`` ranges; missing
  keys default to ``(0.0, 0.0)`` (no offset). ``None`` is treated as empty. The
  returned tensor has the requested ``shape`` whose last dimension must be 6.
  """
  range_dict = range_dict or {}
  range_list = [range_dict.get(key, (0.0, 0.0)) for key in _SE3_KEYS]
  ranges = torch.tensor(range_list, device=device)
  return sample_uniform(ranges[:, 0], ranges[:, 1], shape, device=device)


def resolve_env_ids(
  env: ManagerBasedRlEnv, env_ids: torch.Tensor | None
) -> torch.Tensor:
  """Return ``env_ids`` unchanged, or all environment indices if ``None``.

  Event functions receive ``env_ids=None`` to mean "all environments" (a full
  reset, or a global-time interval term). This normalizes that sentinel to a
  concrete index tensor so the function body can assume a real ``torch.Tensor``.
  """
  if env_ids is None:
    return torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  return env_ids


def randomize_terrain(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None) -> None:
  """Randomize the sub-terrain for each environment on reset. Unchanged."""
  env_ids = resolve_env_ids(env, env_ids)

  terrain = env.scene.terrain
  if terrain is not None:
    terrain.randomize_env_origins(env_ids)


def reset_scene_to_default(
  env: ManagerBasedRlEnv, env_ids: torch.Tensor | None
) -> None:
  """Reset all entities in the scene to their default states. Unchanged.

  For floating-base entities: Resets root state (position, orientation, velocities).
  For fixed-base mocap entities: Resets mocap pose.
  For all articulated entities: Resets joint positions and velocities.

  Automatically applies env_origins offset to position all entities correctly.
  """
  env_ids = resolve_env_ids(env, env_ids)

  for entity in env.scene.entities.values():
    if not isinstance(entity, Entity):
      continue

    if entity.is_fixed_base and entity.is_mocap:
      default_root_state = entity.data.default_root_state[env_ids].clone()
      mocap_pose = torch.zeros((len(env_ids), 7), device=env.device)
      mocap_pose[:, 0:3] = default_root_state[:, 0:3] + env.scene.env_origins[env_ids]
      mocap_pose[:, 3:7] = default_root_state[:, 3:7]
      entity.write_mocap_pose_to_sim(mocap_pose, env_ids=env_ids)
    elif not entity.is_fixed_base:
      default_root_state = entity.data.default_root_state[env_ids].clone()
      default_root_state[:, 0:3] += env.scene.env_origins[env_ids]
      entity.write_root_state_to_sim(default_root_state, env_ids=env_ids)

    if entity.is_articulated:
      default_joint_pos = entity.data.default_joint_pos[env_ids].clone()
      default_joint_vel = entity.data.default_joint_vel[env_ids].clone()
      entity.write_joint_state_to_sim(
        default_joint_pos, default_joint_vel, env_ids=env_ids
      )


def reset_root_state_uniform(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  pose_range: dict[str, tuple[float, float]],
  velocity_range: dict[str, tuple[float, float]] | None = None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Reset root state for floating-base or mocap fixed-base entities. Unchanged.

  This is the core mechanism for standup's "start in some arbitrary state"
  requirement: pass wide roll/pitch ranges (e.g. (-3.14, 3.14)) to cover
  fallen orientations, and a nonzero velocity_range to cover "still moving
  when the episode starts". See `reset_fallen_state` below for a named
  wrapper instead of inlining these ranges in your task cfg directly.

  For floating-base entities: Resets pose and velocity via write_root_state_to_sim().
  For fixed-base mocap entities: Resets pose only via write_mocap_pose_to_sim().

  .. note::
    This function applies the env_origins offset to position entities in a grid.
    For fixed-base robots, this is the ONLY way to position them per-environment.
    Without calling this function in a reset event, fixed-base robots will stack
    at (0,0,0).

  Args:
    env: The environment.
    env_ids: Environment IDs to reset. If None, resets all environments.
    pose_range: Dictionary with keys {"x", "y", "z", "roll", "pitch", "yaw"}.
    velocity_range: Velocity range (only used for floating-base entities).
    asset_cfg: Asset configuration.
  """
  env_ids = resolve_env_ids(env, env_ids)

  asset: Entity = env.scene[asset_cfg.name]

  pose_samples = _sample_se3_range(pose_range, (len(env_ids), 6), env.device)

  if asset.is_fixed_base:
    if not asset.is_mocap:
      raise ValueError(
        f"Cannot reset root state for fixed-base non-mocap entity '{asset_cfg.name}'."
      )

    default_root_state = asset.data.default_root_state
    assert default_root_state is not None
    root_states = default_root_state[env_ids].clone()

    positions = (
      root_states[:, 0:3] + pose_samples[:, 0:3] + env.scene.env_origins[env_ids]
    )
    orientations_delta = quat_from_euler_xyz(
      pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5]
    )
    orientations = quat_mul(root_states[:, 3:7], orientations_delta)

    asset.write_mocap_pose_to_sim(
      torch.cat([positions, orientations], dim=-1), env_ids=env_ids
    )
    return

  default_root_state = asset.data.default_root_state
  assert default_root_state is not None
  root_states = default_root_state[env_ids].clone()

  positions = (
    root_states[:, 0:3] + pose_samples[:, 0:3] + env.scene.env_origins[env_ids]
  )
  orientations_delta = quat_from_euler_xyz(
    pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5]
  )
  orientations = quat_mul(root_states[:, 3:7], orientations_delta)

  vel_samples = _sample_se3_range(velocity_range, (len(env_ids), 6), env.device)
  velocities = root_states[:, 7:13] + vel_samples

  asset.write_root_link_pose_to_sim(
    torch.cat([positions, orientations], dim=-1), env_ids=env_ids
  )

  asset.write_root_link_velocity_to_sim(velocities, env_ids=env_ids)


def reset_fallen_state(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  orientation_mode: str = "any",
  height_range: tuple[float, float] = (0.0, 0.0),
  position_xy_range: dict[str, tuple[float, float]] | None = None,
  velocity_range: dict[str, tuple[float, float]] | None = None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Spawn the robot fallen, mid-tumble, or mid-motion for standup training.

  Thin, self-documenting wrapper around `reset_root_state_uniform`. The
  point of this function is naming the intent explicitly in your task cfg
  rather than relying on someone reading raw roll/pitch numbers and
  inferring "oh, this means fallen".

  Note that "fallen" and "moving when the episode starts" are physically
  different distributions, not one blended one -- a robot lying flat on the
  ground does not also have high root velocity, and a robot that just
  stopped running is unlikely to already be lying flat. Use
  `orientation_mode` to pick which regime this call samples, and mix
  multiple `EventTerm`s (each with their own probability/weight in your
  event manager) if you want a curriculum across several fallen/moving
  states rather than trying to cover all of them from a single call.

  Args:
    env: The environment.
    env_ids: Environment IDs to reset. If None, resets all environments.
    orientation_mode: One of:
      - "any": uniform roll and pitch over the full range (any orientation,
        including upside-down). Use for "robot is lying on the ground in
        an arbitrary way".
      - "side": roll randomized widely, pitch kept near zero -- robot on
        its side rather than face-up/face-down. Useful if your robot's
        get-up motion differs meaningfully by fall direction and you want
        to isolate side-falls as their own training distribution.
      - "near_upright": small roll/pitch perturbation only (e.g. +-30deg).
        Use this for "was running/standing and got knocked off-balance or
        pushed mid-standup", as distinct from a full fall -- pair with a
        nonzero velocity_range.
    height_range: Offset added to the default root height (m). For "any"/
      "side" you typically want this near the robot's lying-down height
      (often a small positive offset so it doesn't spawn clipped into the
      ground); for "near_upright" leave near 0 since the default standing
      height is already appropriate.
    position_xy_range: Optional dict with "x"/"y" keys for randomizing spawn
      position on top of env_origins; omitted keys default to no offset.
    velocity_range: Root velocity range, same format as
      `reset_root_state_uniform`. Use this for the "stopped from running"
      case -- e.g. {"x": (-2.0, 2.0)} for residual forward velocity --
      while leaving it at defaults (zero) for a settled fall.
    asset_cfg: Asset configuration.
  """
  env_ids = resolve_env_ids(env, env_ids)

  if orientation_mode == "any":
    roll_range = (-3.14159, 3.14159)
    pitch_range = (-3.14159, 3.14159)
  elif orientation_mode == "side":
    roll_range = (-3.14159, 3.14159)
    pitch_range = (-0.2, 0.2)
  elif orientation_mode == "near_upright":
    roll_range = (-0.5, 0.5)
    pitch_range = (-0.5, 0.5)
  else:
    raise ValueError(
      f"Unknown orientation_mode '{orientation_mode}'; "
      "expected 'any', 'side', or 'near_upright'."
    )

  pose_range: dict[str, tuple[float, float]] = {
    "z": height_range,
    "roll": roll_range,
    "pitch": pitch_range,
    "yaw": (-3.14159, 3.14159),
  }
  if position_xy_range is not None:
    pose_range.update(position_xy_range)

  reset_root_state_uniform(
    env,
    env_ids,
    pose_range=pose_range,
    velocity_range=velocity_range,
    asset_cfg=asset_cfg,
  )


def reset_root_state_from_flat_patches(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  patch_name: str = "spawn",
  pose_range: dict[str, tuple[float, float]] | None = None,
  velocity_range: dict[str, tuple[float, float]] | None = None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Reset root state by placing the asset on a randomly chosen flat patch.
  Unchanged -- still useful if you want standup training to happen on
  varied/uneven terrain rather than only flat ground.

  Selects a random flat patch from the terrain for each environment and positions
  the asset there. Falls back to ``reset_root_state_uniform`` if the terrain has
  no flat patches.

  Args:
    env: The environment.
    env_ids: Environment IDs to reset. If None, resets all environments.
    patch_name: Key into ``terrain.flat_patches`` to use.
    pose_range: Optional random offset applied on top of the patch position.
      Keys: ``{"x", "y", "z", "roll", "pitch", "yaw"}``.
    velocity_range: Optional velocity range (floating-base only).
    asset_cfg: Asset configuration.
  """
  env_ids = resolve_env_ids(env, env_ids)

  terrain = env.scene.terrain
  if terrain is None or patch_name not in terrain.flat_patches:
    reset_root_state_uniform(
      env,
      env_ids,
      pose_range=pose_range or {},
      velocity_range=velocity_range,
      asset_cfg=asset_cfg,
    )
    return

  patches = terrain.flat_patches[patch_name]  # (num_rows, num_cols, num_patches, 3)
  num_patches = patches.shape[2]

  levels = terrain.terrain_levels[env_ids]
  types = terrain.terrain_types[env_ids]

  patch_ids = torch.randint(0, num_patches, (len(env_ids),), device=env.device)
  positions = patches[levels, types, patch_ids]

  asset: Entity = env.scene[asset_cfg.name]
  default_root_state = asset.data.default_root_state
  assert default_root_state is not None
  root_states = default_root_state[env_ids].clone()

  pose_samples = _sample_se3_range(pose_range, (len(env_ids), 6), env.device)

  final_positions = positions.clone()
  final_positions[:, 0] += pose_samples[:, 0]
  final_positions[:, 1] += pose_samples[:, 1]
  final_positions[:, 2] += root_states[:, 2] + pose_samples[:, 2]

  orientations_delta = quat_from_euler_xyz(
    pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5]
  )
  orientations = quat_mul(root_states[:, 3:7], orientations_delta)

  if asset.is_fixed_base:
    if not asset.is_mocap:
      raise ValueError(
        f"Cannot reset root state for fixed-base non-mocap entity '{asset_cfg.name}'."
      )
    asset.write_mocap_pose_to_sim(
      torch.cat([final_positions, orientations], dim=-1), env_ids=env_ids
    )
    return

  vel_samples = _sample_se3_range(velocity_range, (len(env_ids), 6), env.device)
  velocities = root_states[:, 7:13] + vel_samples

  asset.write_root_link_pose_to_sim(
    torch.cat([final_positions, orientations], dim=-1), env_ids=env_ids
  )
  asset.write_root_link_velocity_to_sim(velocities, env_ids=env_ids)


def reset_joints_by_offset(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  position_range: tuple[float, float],
  velocity_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Unchanged. Still useful: randomizing joint pose/vel at reset is equally
  relevant whether the robot starts standing or fallen -- e.g. randomizing
  limb positions while lying down so the policy doesn't overfit to one
  exact fallen pose."""
  env_ids = resolve_env_ids(env, env_ids)

  asset: Entity = env.scene[asset_cfg.name]
  default_joint_pos = asset.data.default_joint_pos
  assert default_joint_pos is not None
  default_joint_vel = asset.data.default_joint_vel
  assert default_joint_vel is not None
  soft_joint_pos_limits = asset.data.soft_joint_pos_limits
  assert soft_joint_pos_limits is not None

  joint_pos = default_joint_pos[env_ids][:, asset_cfg.joint_ids].clone()
  joint_pos += sample_uniform(*position_range, joint_pos.shape, env.device)
  joint_pos_limits = soft_joint_pos_limits[env_ids][:, asset_cfg.joint_ids]
  joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])

  joint_vel = default_joint_vel[env_ids][:, asset_cfg.joint_ids].clone()
  joint_vel += sample_uniform(*velocity_range, joint_vel.shape, env.device)

  joint_ids = asset_cfg.joint_ids
  if isinstance(joint_ids, list):
    joint_ids = torch.tensor(joint_ids, device=env.device)

  asset.write_joint_state_to_sim(
    joint_pos.view(len(env_ids), -1),
    joint_vel.view(len(env_ids), -1),
    env_ids=env_ids,
    joint_ids=joint_ids,
  )


def push_by_setting_velocity(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  velocity_range: dict[str, tuple[float, float]],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Push an entity by overwriting its root velocity with a sampled offset.
  Unchanged -- this is exactly your "someone pushed the robot while
  standing up" mechanism. Use with ``mode="interval"`` so it fires
  periodically during an episode, not just at reset.

  This is an *instantaneous, mass-independent* kick: it adds a uniformly sampled
  delta directly to the root velocity, ignoring inertia and contact dynamics. It
  is the cheapest disturbance and the standard locomotion "push the robot" term.

  For force-based disturbances that respect the entity's dynamics, see
  :func:`apply_external_force_torque` (a constant wrench you manage yourself) or
  :class:`apply_body_impulse` (transient, self-managing impulses).
  """
  env_ids = resolve_env_ids(env, env_ids)
  asset: Entity = env.scene[asset_cfg.name]
  vel_w = asset.data.root_link_vel_w[env_ids]
  vel_w += _sample_se3_range(velocity_range, vel_w.shape, env.device)
  asset.write_root_link_velocity_to_sim(vel_w, env_ids=env_ids)


def apply_external_force_torque(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  force_range: tuple[float, float],
  torque_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Apply a single *constant* external wrench to bodies. Unchanged.

  Samples a force and torque once and writes them to ``xfrc_applied``. The wrench
  is **stateless and never expires**: MuJoCo holds it constant on every physics
  step until something overwrites or zeroes it. There is no duration, cooldown,
  or auto-clear.

  **When to use this vs.** :class:`apply_body_impulse`:

  - Use ``apply_external_force_torque`` for a *steady, episode-long* disturbance
    such as a fixed payload, a constant wind, or a sustained load. The intended
    pattern is ``mode="reset"``: re-randomize the wrench each episode so it holds
    for that episode's duration. Because it never turns itself off, **you are
    responsible for clearing or overwriting it** (e.g. via the next reset). It is
    *not* suited to transient bumps on its own.

  - Use :class:`apply_body_impulse` for *transient, repeated, randomized*
    disturbances during an episode (bumps, gusts, collisions). It runs a full
    cooldown -> trigger -> sustain -> expire lifecycle per environment, zeroing
    the wrench automatically when each impulse ends, and ticks on ``mode="step"``.

  For an instantaneous, mass-independent kick instead of a force, see
  :func:`push_by_setting_velocity`.
  """
  env_ids = resolve_env_ids(env, env_ids)
  asset: Entity = env.scene[asset_cfg.name]
  num_bodies = (
    len(asset_cfg.body_ids)
    if isinstance(asset_cfg.body_ids, list)
    else asset.num_bodies
  )
  size = (len(env_ids), num_bodies, 3)
  forces = sample_uniform(*force_range, size, env.device)
  torques = sample_uniform(*torque_range, size, env.device)
  asset.write_external_wrench_to_sim(
    forces, torques, env_ids=env_ids, body_ids=asset_cfg.body_ids
  )


class apply_body_impulse:
  """Apply random impulses to bodies for a sampled duration. Unchanged --
  this is the other half of your "pushed while standing up" requirement,
  for repeated/randomized transient bumps during an episode rather than a
  single push. Use with ``mode="step"``.

  Simulates transient external disturbances such as bumps, wind gusts, or
  collisions with unseen objects. A constant force/torque wrench is applied
  to one or more bodies for a randomly sampled duration, followed by a
  cooldown period of silence before the next impulse.

  **Lifecycle of a single impulse:**

  1. **Cooldown.** The event is idle for a random duration sampled from ``cooldown_s``.
    No force is applied.
  2. **Trigger.** A force vector is sampled uniformly per component from ``force_range``
    and written to ``xfrc_applied`` on the selected bodies.
  3. **Sustain.** The force is held constant for a random duration sampled from
    ``duration_s``.
  4. **Expire.** The force is zeroed and the cooldown restarts at step 1.

  Each environment runs its own independent timer so impulses are decorrelated across
  the batch.

  **Application point.** By default, forces act at each body's center of mass.
  ``body_point_offset`` shifts the application point in the body's local frame, for
  example ``(0, 0, 0.1)`` for 10 cm above the CoM. The offset produces additional
  torque via the cross product ``offset x force``, causing the body to tip rather than
  just translate. This is analogous to choosing where on the body an external push is
  applied.

  For a *constant* episode-long wrench instead of transient impulses, see
  :func:`apply_external_force_torque`. For an instantaneous, mass-independent
  velocity kick, see :func:`push_by_setting_velocity`.
  """

  @dataclass
  class VizCfg:
    """Arrow visualization settings for active impulse forces."""

    rgba: tuple[float, float, float, float] = (0.9, 0.2, 0.8, 0.9)
    """Arrow color (RGBA)."""
    scale: float = 0.005
    """Arrow length in meters per Newton of force."""
    width: float = 0.015
    """Arrow shaft width in meters."""
    min_force: float = 1.0
    """Minimum force magnitude (N) below which arrows are hidden."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self._asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    self._body_ids = cfg.params["asset_cfg"].body_ids
    self._num_envs = env.num_envs
    self._device = env.device
    self._step_dt = env.step_dt
    self._viz_cfg: apply_body_impulse.VizCfg = cfg.params.get(
      "viz_cfg", apply_body_impulse.VizCfg()
    )
    offset = cfg.params.get("body_point_offset", None)
    self._body_point_offset: torch.Tensor | None = (
      torch.tensor(offset, device=self._device, dtype=torch.float32)
      if offset is not None
      else None
    )

    self._num_bodies = (
      len(self._body_ids)
      if isinstance(self._body_ids, list)
      else self._asset.num_bodies
    )

    self._cooldown_s: tuple[float, float] = cfg.params["cooldown_s"]
    self._time_remaining = torch.zeros(self._num_envs, device=self._device)
    self._active = torch.zeros(self._num_envs, device=self._device, dtype=torch.bool)
    self._interval_time_left = self._sample_cooldown(self._num_envs)

  def _sample_cooldown(self, n: int) -> torch.Tensor:
    low, high = self._cooldown_s
    return sample_uniform(low, high, n, self._device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    force_range: tuple[float, float],
    torque_range: tuple[float, float],
    duration_s: tuple[float, float],
    cooldown_s: tuple[float, float],
    asset_cfg: SceneEntityCfg,
    body_point_offset: tuple[float, float, float] | None = None,
  ) -> None:
    """Tick impulse state: expire old impulses, trigger new ones.

    Args:
      env: The environment instance.
      env_ids: Unused (step events always operate on all envs).
      force_range: ``(min, max)`` uniform range for each force component (N).
      torque_range: ``(min, max)`` uniform range for each torque component (Nm).
      duration_s: ``(min, max)`` uniform range for impulse duration in seconds.
      cooldown_s: ``(min, max)`` uniform range for the cooldown between consecutive
        impulses in seconds. Captured at init so the first impulse can be
        preceded by a sampled cooldown; the kwarg passed here is unused.
      asset_cfg: Entity and body selection. ``body_ids`` on the config selects which
        bodies receive forces.
      body_point_offset: Optional ``(x, y, z)`` offset in the body frame where the
        force is applied. Generates additional torque via ``cross(offset, force)``.
    """
    del env, env_ids, asset_cfg, cooldown_s  # Unused at call time.
    dt = self._step_dt

    self._time_remaining[self._active] -= dt

    expired = self._active & (self._time_remaining <= 0)
    if expired.any():
      expired_ids = expired.nonzero(as_tuple=False).squeeze(-1)
      zeros = torch.zeros((len(expired_ids), self._num_bodies, 3), device=self._device)
      self._asset.write_external_wrench_to_sim(
        zeros, zeros, env_ids=expired_ids, body_ids=self._body_ids
      )
      self._active[expired_ids] = False
      self._time_remaining[expired_ids] = 0.0
      self._interval_time_left[expired_ids] = self._sample_cooldown(len(expired_ids))

    self._interval_time_left -= dt

    eligible = (~self._active) & (self._interval_time_left <= 0)
    if not eligible.any():
      return

    trigger_ids = eligible.nonzero(as_tuple=False).squeeze(-1)
    n = len(trigger_ids)

    size = (n, self._num_bodies, 3)
    forces = sample_uniform(*force_range, size, self._device)
    torques = sample_uniform(*torque_range, size, self._device)

    if body_point_offset is not None:
      offset_local = torch.tensor(
        body_point_offset, device=self._device, dtype=torch.float32
      )
      body_quat = self._asset.data.body_com_quat_w[trigger_ids][:, self._body_ids]
      offset_w = quat_apply(
        body_quat.reshape(-1, 4), offset_local.expand(n * self._num_bodies, 3)
      ).reshape(n, self._num_bodies, 3)
      torques = torques + torch.cross(offset_w, forces, dim=-1)

    self._asset.write_external_wrench_to_sim(
      forces, torques, env_ids=trigger_ids, body_ids=self._body_ids
    )

    dur_low, dur_high = duration_s
    self._time_remaining[trigger_ids] = (
      torch.rand(n, device=self._device) * (dur_high - dur_low) + dur_low
    )
    self._active[trigger_ids] = True

    self._interval_time_left[trigger_ids] = self._sample_cooldown(n)

  def debug_vis(self, visualizer: DebugVisualizer) -> None:
    """Draw arrows for active impulse forces."""
    if not self._active.any():
      return
    viz = self._viz_cfg
    min_sq = viz.min_force * viz.min_force
    wrench = self._asset.data.body_external_wrench  # (nworld, nbody, 6)
    com_pos = self._asset.data.body_com_pos_w  # (nworld, nbody, 3)
    offset = self._body_point_offset
    com_quat = self._asset.data.body_com_quat_w if offset is not None else None
    for env_idx in visualizer.get_env_indices(self._num_envs):
      if not self._active[env_idx]:
        continue
      for i in range(wrench.shape[1]):
        force = wrench[env_idx, i, :3]
        if (force * force).sum().item() < min_sq:
          continue
        force_np = force.cpu().numpy()
        start_np = com_pos[env_idx, i].cpu().numpy()
        if offset is not None and com_quat is not None:
          offset_w = quat_apply(com_quat[env_idx, i], offset)
          start_np = start_np + offset_w.cpu().numpy()
        end_np = start_np + force_np * viz.scale
        visualizer.add_arrow(
          start=start_np,
          end=end_np,
          color=viz.rgba,
          width=viz.width,
        )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)

    if self._active[env_ids].any():
      if isinstance(env_ids, slice):
        active_ids = self._active.nonzero(as_tuple=False).squeeze(-1)
      else:
        active_ids = env_ids[self._active[env_ids]]
      if len(active_ids) > 0:
        zeros = torch.zeros(
          (len(active_ids), self._num_bodies, 3),
          device=self._device,
        )
        self._asset.write_external_wrench_to_sim(
          zeros, zeros, env_ids=active_ids, body_ids=self._body_ids
        )

    n = self._num_envs if isinstance(env_ids, slice) else len(env_ids)
    self._time_remaining[env_ids] = 0.0
    self._interval_time_left[env_ids] = self._sample_cooldown(n)
    self._active[env_ids] = False