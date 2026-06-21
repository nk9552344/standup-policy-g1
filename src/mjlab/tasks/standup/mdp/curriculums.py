"""Useful methods for MDP curricula for the stay-stand (balance) policy.

Forked from the standup recovery curriculums.py. One function kept, one
removed:

  - `terrain_levels_standup` KEPT, with an updated success definition.
    In recovery, "success" meant "the robot reached standing height from
    a fallen start state". In stay-stand, "success" means "the robot was
    still upright when the episode ended" -- i.e. it survived the full
    episode (or was still standing at time_out) without triggering
    fell_over. The height + uprightness gate is identical; only the
    semantic meaning changes. The `min_step_counter` gate is also removed:
    in recovery it held terrain at level 0 until Stage 1 of fall_difficulty
    so the robot could first master easy terrain before facing rough terrain.
    In stay-stand there is no fall_difficulty stage gating and the robot
    starts standing from iteration 0, so the gate is unnecessary -- any
    robot that survives an episode on terrain level 0 should immediately
    advance.

  - `fall_difficulty` REMOVED entirely. That function staged the severity
    of fallen start states and push disturbances over training (standing ->
    near_upright -> side -> any). Stay-stand has no fallen start states:
    the reset always spawns from HOME_KEYFRAME (upright). Push disturbance
    strength is set as a fixed param in env_cfgs.py rather than staged.
    `FallStage` TypedDict also removed as it only served `fall_difficulty`.

NOTE: `terrain_levels_standup` reads pre-reset state at curriculum-call
time, same as the recovery version. See the note in the recovery docstring
if your manager calls curriculum terms after reset.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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
  """Progress/regress terrain difficulty based on stay-stand success.

  Success definition: was the robot still upright (height + uprightness gate)
  when this episode ended? In recovery, success meant "reached standing height
  from a fallen start". In stay-stand, any episode that ends with the robot
  still upright is a success (it survived without triggering fell_over);
  any episode that ended via fell_over is a failure. The gate computation is
  identical -- only the semantic interpretation changes.

  Envs that were upright at episode end advance to harder terrain; envs that
  had fallen move to easier terrain.

  The `min_step_counter` gate present in the recovery version is removed.
  That gate held terrain at level 0 until fall_difficulty Stage 1 unlocked
  harder falls, so the robot could master easy terrain before facing rough
  terrain while also dealing with arbitrary fallen starts. In stay-stand
  there is no fall_difficulty and the robot starts upright from iteration 0,
  so a robot that survives episode 1 on level 0 terrain should advance
  immediately.
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