"""Useful methods for MDP terminations.

Forked from the REAL mjlab.tasks.velocity.mdp.termination.py (time_out,
illegal_contact, out_of_terrain_bounds, terrain_edge_reached) -- a previous
version of this file in this conversation was mistakenly based on a
different, shorter termination.py (time_out, bad_orientation,
root_height_below_minimum, nan_detection) and should be considered
superseded by this one.

Per-term conversion:

  - time_out: unchanged. Episode length cutoff is task-agnostic.
  - illegal_contact: DROPPED ENTIRELY for standup. This term terminates
    when a forbidden body part (e.g. knee, hand, torso -- whatever your
    locomotion config wires it to) touches the ground above a force
    threshold. In locomotion that's always a failure (the robot scuffed or
    fell). In standup it's the opposite: hands pushing off the ground,
    knees touching down, rolling onto the torso or a side are the expected,
    necessary mechanics of getting up. Keeping this verbatim would
    terminate nearly every standup episode immediately, the same failure
    mode bad_orientation/root_height_below_minimum had in the other
    termination.py. No narrowed/partial version is kept per your
    confirmation that no body part is illegal to touch ground during
    recovery.
  - out_of_terrain_bounds: unchanged. Pure xy-position-vs-terrain-footprint
    geometry check, no coupling to gait or velocity commands.
  - terrain_edge_reached: DROPPED for standup. This is a "successful
    traversal" signal (displacement from spawn exceeding sub-terrain size,
    time_out=True, not penalized) -- the termination-side counterpart to
    terrain_levels_vel's distance-traveled curriculum metric. Standup has
    no "walk to the edge of the terrain" goal, so there's no analogous
    success condition to detect this way; "success" for standup is height +
    uprightness, not displacement, and is better expressed via reward
    (standup_progress) and the terrain curriculum's own success check
    (terrain_levels_standup), not a termination.

What replaces the removed failure/success signals:

  - catastrophic_state: NEW. Sanity-only check (height far below ground or
    far above plausible standing height -- sim explosion/clipping), since
    standup necessarily starts and spends time in states illegal_contact/
    bad_orientation would have flagged as failures. Does not check contact
    or orientation at all.
  - stuck_no_progress: NEW. Ends an episode early if height hasn't improved
    for a configured duration, so a stalled recovery attempt doesn't burn
    the full episode length doing nothing -- locomotion's termination set
    never needed an analog since there's no single-episode goal to stall
    on while walking.

nan_detection: kept from the earlier (different-source) version of this
file as a numerical safety net. It was not present in either termination.py
you've shown me -- if it doesn't actually exist in your codebase, drop the
import/registration for it; it's independent of everything else here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.managers.termination_manager import TerminationTermCfg

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def time_out(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Terminate when the episode length exceeds its maximum. Unchanged."""
  return env.episode_length_buf >= env.max_episode_length


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


class stuck_no_progress:
  """Terminate early if the robot hasn't gained height for too long.

  No analog in the velocity termination.py: locomotion termination is only
  about ongoing failure (illegal contact, out of bounds, timed out) or
  successful traversal (terrain_edge_reached), never "isn't making progress
  toward a goal" within a single attempt, since steady walking has no
  single-episode goal to stall on. Standup does have one (reach standing
  height), so a stalled attempt -- e.g. wedged in a pose it can't escape, or
  has stopped trying -- is worth ending early.

  Tracks each env's best (highest) height seen so far and a per-env stall
  timer that resets whenever a new best height is reached by at least
  `min_improvement`. Terminates an env once its timer exceeds `patience_s`
  without sufficient improvement.
  """

  def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRlEnv):
    self._asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)
    self._step_dt = env.step_dt
    self._best_height = torch.full(
      (env.num_envs,), float("-inf"), device=env.device
    )
    self._stall_time = torch.zeros(env.num_envs, device=env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    patience_s: float,
    min_improvement: float = 0.01,
    min_standing_height: float = 0.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    height = asset.data.root_link_pos_w[:, 2]

    improved = height > (self._best_height + min_improvement)
    self._best_height = torch.where(improved, height, self._best_height)
    self._stall_time = torch.where(
      improved,
      torch.zeros_like(self._stall_time),
      self._stall_time + self._step_dt,
    )

    stuck = self._stall_time > patience_s
    # Don't terminate an env that is *currently* at or above standing height.
    # Robots that are standing (even if they've been stationary for a while,
    # e.g. holding the standing pose) must never be cut off. Only terminate
    # robots that are *currently below* standing height and haven't improved
    # for `patience_s` seconds -- those are genuinely stuck on the ground.
    #
    # The previous `ever_standing` gate (`_best_height >= min_standing_height`)
    # was wrong: it becomes True at step 1 for any robot that spawns from
    # HOME_KEYFRAME (height 0.783m > 0.65m), so stuck & ~ever_standing was
    # always False. Episodes ran to full time_out with the robot lying on the
    # ground for 17-19 s after being pushed, generating long stretches of
    # near-zero reward that destabilised the value function.
    currently_standing = height >= min_standing_height
    return stuck & ~currently_standing

  def reset(self, env_ids: torch.Tensor) -> None:
    self._best_height[env_ids] = float("-inf")
    self._stall_time[env_ids] = 0.0


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