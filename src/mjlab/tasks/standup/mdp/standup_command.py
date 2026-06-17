# DONE

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import matrix_from_quat, quat_apply

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class StandStillCommand(CommandTerm):
  """Command term for a standup/recovery policy.

  Unlike UniformVelocityCommand, there is nothing to sample: the target is
  always zero base linear and angular velocity -- "get up and hold still".

  What's dropped from UniformVelocityCommand and why:
    - heading_command / heading_target / heading_error / heading stiffness:
      no heading goal, the robot just needs to stop wherever it ends up.
    - vel_command_w / is_world_env / world-frame rotation: nothing to
      rotate, command is the zero vector in every frame.
    - is_forward_env / rel_forward_envs: no forward-walking bias needed.
    - joystick GUI: there's no command to manually drive in this task.
      (If you want a manual "shove the robot" debug control instead, that's
      a different GUI -- ask if you want it added here.)

  What's kept and why:
    - is_standing_env / rel_standing_envs renamed conceptually but same
      mechanism isn't needed either, since command is always zero -- see
      note below, it's actually removed too (command is unconditionally 0).
    - init_velocity_prob and the reset-time velocity injection: this is
      exactly your "robot was running and just stopped" / "pushed mid-
      standup" scenario. Kept and slightly generalized (separate xyz ranges,
      optional angular randomization on all axes, not just yaw).
    - error_vel_xy / error_vel_yaw metrics: kept, still meaningful as
      "how far is the robot's actual velocity from zero" tracking error,
      gated so it only accumulates once the robot is actually standing
      (see min_standing_height) -- otherwise it penalizes the standup
      motion itself, which you don't want.
  """

  cfg: StandStillCommandCfg

  def __init__(self, cfg: StandStillCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.entity_name]

    # Command is always zero; kept as a real tensor (not a constant) so it
    # still slots into observation/reward code that calls self.command.
    self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)

    self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.vel_command_b

  def _update_metrics(self) -> None:
    max_command_time = self.cfg.resampling_time_range[1]
    max_command_step = max_command_time / self._env.step_dt

    # Only count tracking error once the robot is actually standing, so the
    # necessary motion of getting up isn't penalized as "error".
    height = self.robot.data.root_link_pos_w[:, 2]
    is_standing = height > self.cfg.min_standing_height

    lin_err = torch.norm(self.robot.data.root_link_lin_vel_b[:, :2], dim=-1)
    yaw_err = torch.abs(self.robot.data.root_link_ang_vel_b[:, 2])

    self.metrics["error_vel_xy"] += torch.where(
      is_standing, lin_err / max_command_step, torch.zeros_like(lin_err)
    )
    self.metrics["error_vel_yaw"] += torch.where(
      is_standing, yaw_err / max_command_step, torch.zeros_like(yaw_err)
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    # Nothing to sample -- command stays the zero vector. This still runs
    # on the configured resampling_time_range cadence, which is what
    # triggers init-velocity injection below if you want fresh disturbances
    # mid-episode rather than only at reset. Set init_velocity_prob to 0.0
    # and rely on a reset-mode EventTerm instead if you only want the
    # disturbance at episode start.
    self.vel_command_b[env_ids] = 0.0

    r = torch.empty(len(env_ids), device=self.device)
    apply_mask = r.uniform_(0.0, 1.0) < self.cfg.init_velocity_prob
    init_ids = env_ids[apply_mask]
    if len(init_ids) == 0:
      return

    root_pos = self.robot.data.root_link_pos_w[init_ids]
    root_quat = self.robot.data.root_link_quat_w[init_ids]

    lin_vel_w = torch.zeros(len(init_ids), 3, device=self.device)
    lin_vel_w[:, 0].uniform_(*self.cfg.init_lin_vel_x)
    lin_vel_w[:, 1].uniform_(*self.cfg.init_lin_vel_y)
    lin_vel_w[:, 2].uniform_(*self.cfg.init_lin_vel_z)

    ang_vel_b = torch.zeros(len(init_ids), 3, device=self.device)
    ang_vel_b[:, 0].uniform_(*self.cfg.init_ang_vel_roll)
    ang_vel_b[:, 1].uniform_(*self.cfg.init_ang_vel_pitch)
    ang_vel_b[:, 2].uniform_(*self.cfg.init_ang_vel_yaw)

    root_state = torch.cat([root_pos, root_quat, lin_vel_w, ang_vel_b], dim=-1)
    self.robot.write_root_state_to_sim(root_state, init_ids)

  def _update_command(self) -> None:
    # No-op: command is always zero, nothing to recompute each step.
    # (UniformVelocityCommand uses this hook for heading control and
    # world-frame rotation; neither applies here.)
    pass

  # Visualization.

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    """Draw actual base velocity arrows (no command arrow -- it's always zero)."""
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    base_pos_ws = self.robot.data.root_link_pos_w.cpu().numpy()
    base_quat_w = self.robot.data.root_link_quat_w
    base_mat_ws = matrix_from_quat(base_quat_w).cpu().numpy()
    lin_vel_bs = self.robot.data.root_link_lin_vel_b.cpu().numpy()
    ang_vel_bs = self.robot.data.root_link_ang_vel_b.cpu().numpy()

    scale = self.cfg.viz.scale
    z_offset = self.cfg.viz.z_offset

    for batch in env_indices:
      base_pos_w = base_pos_ws[batch]
      base_mat_w = base_mat_ws[batch]
      lin_vel_b = lin_vel_bs[batch]
      ang_vel_b = ang_vel_bs[batch]

      if np.linalg.norm(base_pos_w) < 1e-6:
        continue

      def local_to_world(
        vec: np.ndarray, pos: np.ndarray = base_pos_w, mat: np.ndarray = base_mat_w
      ) -> np.ndarray:
        return pos + mat @ vec

      act_lin_from = local_to_world(np.array([0, 0, z_offset]) * scale)
      act_lin_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([lin_vel_b[0], lin_vel_b[1], 0])) * scale
      )
      visualizer.add_arrow(
        act_lin_from, act_lin_to, color=(0.0, 0.6, 1.0, 0.7), width=0.015
      )

      act_ang_from = act_lin_from
      act_ang_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([0, 0, ang_vel_b[2]])) * scale
      )
      visualizer.add_arrow(
        act_ang_from, act_ang_to, color=(0.0, 1.0, 0.4, 0.7), width=0.015
      )


@dataclass(kw_only=True)
class StandStillCommandCfg(CommandTermCfg):
  entity_name: str

  min_standing_height: float = 0.3
  """Base height (m) above which the robot is considered standing, used to
  gate the tracking-error metric so getting up isn't penalized as error."""

  init_velocity_prob: float = 0.0
  """Probability (per resample) of injecting nonzero root velocity, i.e.
  'robot was moving when this episode/segment starts'. Set resampling_time_
  range to match episode length if you only want this applied once at the
  start, or shorter if you want disturbances injected mid-episode too."""

  init_lin_vel_x: tuple[float, float] = (0.0, 0.0)
  init_lin_vel_y: tuple[float, float] = (0.0, 0.0)
  init_lin_vel_z: tuple[float, float] = (0.0, 0.0)
  init_ang_vel_roll: tuple[float, float] = (0.0, 0.0)
  init_ang_vel_pitch: tuple[float, float] = (0.0, 0.0)
  init_ang_vel_yaw: tuple[float, float] = (0.0, 0.0)

  @dataclass
  class VizCfg:
    z_offset: float = 0.2
    scale: float = 0.5

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> StandStillCommand:
    return StandStillCommand(self, env)