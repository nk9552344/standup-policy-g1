"""Useful methods for MDP curricula.

Forked from the velocity locomotion curriculum.py. Both original terms were
specific to "tracking a commanded velocity while walking" and needed a real
rework, not just a rename, since standup has neither distance-to-walk nor a
velocity range to widen:

  - `terrain_levels_vel` progressed terrain difficulty based on distance
    walked relative to terrain size and commanded speed. Standup doesn't
    walk anywhere -- there's no analogous distance signal. Replaced by
    `terrain_levels_standup`, which progresses based on whether the robot
    successfully reached standing height in that episode instead.
  - `commands_vel` widened `UniformVelocityCommandCfg.ranges` (lin_vel_x/y,
    ang_vel_z) over training steps. Standup has no command ranges to widen
    -- the command is always zero (see StandStillCommand). The actual
    curriculum axis you want is fall/disturbance severity, which lives on
    *event term* params (reset_fallen_state's orientation_mode, push
    force/velocity ranges), not on a command term. Replaced by
    `fall_difficulty`, which mutates event term params the same way
    commands_vel mutated command term params, just via env.event_manager
    instead of env.command_manager.

NOTE: `fall_difficulty` assumes `env.event_manager.get_term_cfg(name)`
returns a mutable cfg with a `.params` dict, mirroring how the original used
`env.command_manager.get_term(name)`. If your EventManager's actual API
differs (different method name, immutable cfg, etc.), the lookup line is the
only thing that needs to change -- the staging logic is unaffected.

NOTE: `terrain_levels_standup` computes "did this env stand successfully"
from current state at curriculum-call time, before any reset has been
applied to env_ids for this step (matching the original's assumption that
curriculum terms read pre-reset distance). If your manager actually invokes
curriculum terms *after* state has already been reset, this signal will be
wrong and you'd need to track success in a per-step buffer instead (e.g. set
a flag in a termination or reward term the moment standing is achieved, and
read that buffer here instead of recomputing from instantaneous state).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")


def _is_standing(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  asset_cfg: SceneEntityCfg,
  min_height: float,
  min_uprightness: float,
) -> torch.Tensor:
  """Same height+uprightness gate used in the reward file, recomputed here
  rather than imported to keep this module's dependencies minimal -- adjust
  to import from your reward module instead if you prefer a single source
  of truth."""
  asset: Entity = env.scene[asset_cfg.name]
  height = asset.data.root_link_pos_w[env_ids, 2]

  gravity_w = asset.data.gravity_vec_w
  root_quat_w = asset.data.root_link_quat_w[env_ids]
  projected_gravity_b = quat_apply_inverse(root_quat_w, gravity_w)
  uprightness = -projected_gravity_b[:, 2]

  return (height > min_height) & (uprightness > min_uprightness)


def terrain_levels_standup(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  min_standing_height: float,
  min_standing_uprightness: float = 0.8,
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
) -> dict[str, torch.Tensor]:
  """Progress/regress terrain difficulty based on standup success.

  Replaces distance-walked progression with: did the robot reach a
  standing pose (height + uprightness gate, same definition used in
  reward/termination) by the time this episode ended. Envs that succeeded
  move to harder terrain; envs that failed move to easier terrain.
  """
  terrain = env.scene.terrain
  assert terrain is not None
  terrain_generator = terrain.cfg.terrain_generator
  assert terrain_generator is not None

  succeeded = _is_standing(
    env, env_ids, asset_cfg, min_standing_height, min_standing_uprightness
  )
  move_up = succeeded
  move_down = ~succeeded

  terrain.update_env_origins(env_ids, move_up, move_down)

  levels = terrain.terrain_levels.float()
  result: dict[str, torch.Tensor] = {
    "mean": torch.mean(levels),
    "max": torch.max(levels),
  }

  sub_terrain_names = list(terrain_generator.sub_terrains.keys())
  terrain_origins = terrain.terrain_origins
  assert terrain_origins is not None
  num_cols = terrain_origins.shape[1]
  if num_cols == len(sub_terrain_names):
    types = terrain.terrain_types
    for i, name in enumerate(sub_terrain_names):
      mask = types == i
      if mask.any():
        result[name] = torch.mean(levels[mask])
  return result


class FallStage(TypedDict):
  step: int
  orientation_mode: str | None
  velocity_range: dict[str, tuple[float, float]] | None
  push_force_range: tuple[float, float] | None
  push_torque_range: tuple[float, float] | None


def fall_difficulty(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  reset_event_name: str,
  fall_stages: list[FallStage],
  push_event_name: str | None = None,
) -> dict[str, torch.Tensor]:
  """Curriculum over fall/disturbance severity instead of velocity ranges.

  Mirrors the structure of the original commands_vel (ordered stages keyed
  by env.common_step_counter, each optionally overriding a subset of
  params), but targets event term params instead of command term ranges:
  start with mild near-upright disturbances, progress to full random falls
  and stronger pushes as training advances.

  Args:
    env: The environment.
    env_ids: Unused, kept for curriculum-manager call signature consistency.
    reset_event_name: Name of the reset_fallen_state EventTerm in your event
      manager config, whose orientation_mode/velocity_range get staged.
    fall_stages: Ordered list of stages; each stage applies once
      env.common_step_counter reaches its "step" value. Only keys present
      and non-None in a stage override the current config -- omitted keys
      leave the prior stage's value in place.
    push_event_name: Optional name of a push_by_setting_velocity or
      apply_body_impulse EventTerm whose force/torque ranges also get
      staged via push_force_range/push_torque_range.
  """
  del env_ids  # Unused; curriculum applies globally via term cfg mutation.

  reset_cfg = env.event_manager.get_term_cfg(reset_event_name)
  assert reset_cfg is not None

  push_cfg = None
  if push_event_name is not None:
    push_cfg = env.event_manager.get_term_cfg(push_event_name)
    assert push_cfg is not None

  for stage in fall_stages:
    if env.common_step_counter >= stage["step"]:
      if stage.get("orientation_mode") is not None:
        reset_cfg.params["orientation_mode"] = stage["orientation_mode"]
      if stage.get("velocity_range") is not None:
        reset_cfg.params["velocity_range"] = stage["velocity_range"]
      if push_cfg is not None:
        if stage.get("push_force_range") is not None:
          push_cfg.params["force_range"] = stage["push_force_range"]
        if stage.get("push_torque_range") is not None:
          push_cfg.params["torque_range"] = stage["push_torque_range"]

  log: dict[str, torch.Tensor] = {
    "orientation_mode_is_any": torch.tensor(
      float(reset_cfg.params.get("orientation_mode") == "any")
    ),
  }
  vel_range = reset_cfg.params.get("velocity_range") or {}
  if "x" in vel_range:
    log["init_lin_vel_x_min"] = torch.tensor(vel_range["x"][0])
    log["init_lin_vel_x_max"] = torch.tensor(vel_range["x"][1])
  if push_cfg is not None:
    force_range = push_cfg.params.get("force_range")
    if force_range is not None:
      log["push_force_min"] = torch.tensor(force_range[0])
      log["push_force_max"] = torch.tensor(force_range[1])

  return log