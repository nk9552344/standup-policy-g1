"""Useful methods for MDP terminations.

Forked from the velocity locomotion termination.py. Two of the original
four terms could not be carried over as-is:

  - `bad_orientation` terminates when tilt from upright exceeds a limit
    angle. For locomotion that's a real failure (the robot has fallen).
    For standup, the robot is *expected* to start near-180-degrees from
    upright (lying down) -- using this verbatim would terminate every
    episode at t=0, before the policy can do anything.
  - `root_height_below_minimum` terminates when height drops below a
    standing threshold. Same problem inverted: standup necessarily starts
    below that threshold and needs the episode to continue while it climbs
    back up.

Both are replaced by `catastrophic_state`, which only fires on physically
impossible states (sim explosion / clipping through the floor), not on
"still recovering". `time_out` and `nan_detection` are unchanged -- neither
encodes any locomotion-specific assumption.

New: `stuck_no_progress`, a stateful termination that ends an episode early
if the robot hasn't gained height for a configured duration, so failed
recovery attempts don't waste the full episode length. This has no analog
in the original file since locomotion termination is purely about ongoing
failure, not stalled progress toward a goal.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.nan_guard import NanGuard

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.managers.termination_manager import TerminationTermCfg

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def time_out(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Terminate when the episode length exceeds its maximum. Unchanged."""
  return env.episode_length_buf >= env.max_episode_length


def nan_detection(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Terminate environments that have NaN/Inf values in their physics
  state. Unchanged -- a numerical safety net, not locomotion-specific."""
  return NanGuard.detect_nans(env.sim.data)


def catastrophic_state(
  env: ManagerBasedRlEnv,
  min_height: float = -0.5,
  max_height: float = 3.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terminate only on physically impossible states, not "still recovering".

  Replaces bad_orientation and root_height_below_minimum. Deliberately does
  NOT check orientation at all -- lying on the back/front/side is a normal,
  expected starting state for standup and must never be treated as a
  termination condition. The only thing checked is base height falling far
  below the ground plane (indicates clipping/tunneling through terrain) or
  rising far above any plausible standing height (indicates the sim has
  exploded, e.g. from an exceptionally bad contact resolution).

  Defaults are intentionally loose: min_height=-0.5 should never legitimately
  occur for a robot resting on the ground (it would be at most ~0, allowing
  for some interpenetration tolerance), and max_height=3.0 should be well
  above any humanoid/quadruped's standing height. Tighten these to your
  specific robot's dimensions if you want a stricter sanity check, but keep
  them well outside the range covered by your `reset_fallen_state` /
  `push_by_setting_velocity` event configs so you don't accidentally
  terminate intentional training states.
  """
  asset: Entity = env.scene[asset_cfg.name]
  height = asset.data.root_link_pos_w[:, 2]
  return (height < min_height) | (height > max_height)


class stuck_no_progress:
  """Terminate early if the robot hasn't gained height for too long.

  No analog in the original locomotion file: locomotion termination is only
  about ongoing failure (fallen, NaN, timed out), never about "isn't making
  progress toward a goal", because steady walking has no single-episode goal
  to stall on. Standup does have one (reach standing height), so a stalled
  attempt -- e.g. the robot is wedged in a pose it can't escape, or has
  given up pushing -- is worth ending early rather than burning the rest of
  the episode for no learning signal.

  Tracks each env's best (highest) height seen so far and a per-env stall
  timer that resets whenever a new best height is reached by at least
  `min_improvement`. Terminates an env once its timer exceeds
  `patience_s` without sufficient improvement.
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

    return self._stall_time > patience_s

  def reset(self, env_ids: torch.Tensor) -> None:
    self._best_height[env_ids] = float("-inf")
    self._stall_time[env_ids] = 0.0