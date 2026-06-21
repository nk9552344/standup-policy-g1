"""Useful methods for MDP terminations for the stay-stand (balance) policy.

Forked from the standup recovery terminations.py. Key differences:

  - fell_over: NEW (restored from the original velocity locomotion policy's
    bad_orientation term). In recovery this term was dropped because the
    robot starts fallen and "being badly oriented" is the whole premise.
    In stay-stand the robot starts upright; any tilt past the threshold
    IS a failure and must terminate the episode immediately. This is the
    primary failure signal for the stay-stand task.

    Combines two checks: base height below min_height (the robot has
    collapsed), and uprightness below min_uprightness (the robot has tilted
    past the allowed range). Either condition alone terminates. Using both
    avoids a pathological middle ground where the robot is low but
    technically still upright (e.g. deep squat) or upright but very low
    (crouching all the way down). Tune thresholds to your robot; for the
    G1 start with min_height=0.5 and min_uprightness=0.5 (~60 degrees of
    allowed tilt -- generous enough to survive hard pushes without
    triggering spurious terminations during normal balance corrections).

  - stuck_no_progress: REMOVED. That term existed specifically for recovery
    episodes where the robot could stall on the ground for 17-19 seconds
    doing nothing useful. In stay-stand the robot starts upright and any
    prolonged ground contact terminates via fell_over instead.

  - catastrophic_state: KEPT. Still a useful sanity net for sim
    explosions/clipping, independent of the fell_over logic.

  - time_out, out_of_terrain_bounds, nan_detection: unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.managers.termination_manager import TerminationTermCfg

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def time_out(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Terminate when the episode length exceeds its maximum. Unchanged."""
  return env.episode_length_buf >= env.max_episode_length


def fell_over(
  env: ManagerBasedRlEnv,
  min_height: float,
  min_uprightness: float = 0.5,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terminate when the robot has fallen over. Primary failure signal for stay-stand.

  Checks two conditions (either alone terminates):
    - height < min_height: the robot's base has dropped below the collapse
      threshold. For G1, min_height=0.5 means the pelvis is below ~63% of
      standing height -- the robot has clearly fallen.
    - uprightness < min_uprightness: the robot has tilted past the allowed
      angle. uprightness = -projected_gravity_b[:, 2], so uprightness=1.0
      is perfectly vertical and 0.0 is horizontal (lying on its side).
      min_uprightness=0.5 corresponds to ~60 degrees of tilt -- generous
      enough to survive hard pushes without spurious termination during
      normal balance corrections, but tight enough to catch genuine falls
      quickly and not waste episode time on a robot that is already on the
      floor.

  Tune min_height and min_uprightness together: if the robot ever reaches a
  state where it is "low but technically upright" (e.g. deep squat where
  height < min_height but uprightness > min_uprightness), tighten
  min_height; if the robot is terminating during normal push recovery,
  loosen min_uprightness.
  """
  asset: Entity = env.scene[asset_cfg.name]
  height = asset.data.root_link_pos_w[:, 2]

  gravity_w = asset.data.gravity_vec_w
  root_quat_w = asset.data.root_link_quat_w
  projected_gravity_b = quat_apply_inverse(root_quat_w, gravity_w)
  uprightness = -projected_gravity_b[:, 2]

  return (height < min_height) | (uprightness < min_uprightness)


def catastrophic_state(
  env: ManagerBasedRlEnv,
  min_height: float = -0.5,
  max_height: float = 3.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terminate only on physically impossible states.

  Replaces illegal_contact for standup. Deliberately checks neither contact
  nor orientation: hands/knees/torso touching the ground and the robot
  being in any orientation are normal, expected parts of recovering from a
  fall and must never be treated as termination conditions. The only thing
  checked is base height falling far below the ground plane (clipping/
  tunneling through terrain) or rising far above any plausible standing
  height (sim explosion from a bad contact resolution).

  Defaults are intentionally loose: min_height=-0.5 should never
  legitimately occur for a robot resting on the ground, and max_height=3.0
  should be well above any humanoid/quadruped's standing height. Tighten to
  your specific robot's dimensions, but keep both well outside the range
  covered by your reset_fallen_state / push_by_setting_velocity event
  configs so you don't accidentally terminate intentional training states.
  """
  asset: Entity = env.scene[asset_cfg.name]
  height = asset.data.root_link_pos_w[:, 2]
  return (height < min_height) | (height > max_height)


def out_of_terrain_bounds(
  env: ManagerBasedRlEnv,
  margin: float = 0.3,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Truncate if robot leaves the generated terrain footprint. Unchanged --
  pure xy-position-vs-terrain-footprint geometry, no gait/velocity coupling.

  Returns all-false for non-generator terrains (e.g. plane).
  """
  terrain = env.scene.terrain
  if terrain is None or terrain.cfg.terrain_type != "generator":
    return torch.zeros(
      (env.num_envs,),
      device=env.device,
      dtype=torch.bool,
    )
  terrain_generator = terrain.cfg.terrain_generator
  if terrain_generator is None or terrain.terrain_origins is None:
    return torch.zeros(
      (env.num_envs,),
      device=env.device,
      dtype=torch.bool,
    )
  asset: Entity = env.scene[asset_cfg.name]
  root_xy_w = asset.data.root_link_pos_w[:, :2]
  num_rows, num_cols = terrain.terrain_origins.shape[:2]
  half_x = 0.5 * (num_rows * terrain_generator.size[0]) + terrain_generator.border_width
  half_y = 0.5 * (num_cols * terrain_generator.size[1]) + terrain_generator.border_width
  limit_x = max(0.0, half_x - margin)
  limit_y = max(0.0, half_y - margin)
  return (root_xy_w[:, 0].abs() > limit_x) | (root_xy_w[:, 1].abs() > limit_y)


def nan_detection(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Terminate environments that have NaN/Inf values in their physics
  state. NOTE: this was not present in either termination.py shown in this
  conversation -- carried over from an earlier draft of this file as a
  numerical safety net. Verify it exists in your actual codebase (e.g.
  mjlab.utils.nan_guard.NanGuard) before relying on it; remove this
  function and its registration in standup_env_cfg.py if it doesn't.
  """
  from mjlab.utils.nan_guard import NanGuard

  return NanGuard.detect_nans(env.sim.data)