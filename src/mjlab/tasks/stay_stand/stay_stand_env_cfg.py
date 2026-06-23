"""Stay-stand task configuration.

This module provides a factory function to create a base stay-stand task config.
Robot-specific configurations call the factory and customize as needed.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def fell_over(
  env: ManagerBasedRlEnv,
  limit_angle: float,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Terminate when torso tilt from upright exceeds `limit_angle` (radians).

  Uses ``-projected_gravity_b[:, 2]`` as the "uprightness" scalar: 1.0 when
  perfectly vertical, ``cos(limit_angle)`` at the threshold tilt, 0.0 when
  horizontal. The shared ``mjlab.envs.mdp`` package does not ship an
  orientation-based termination (only ``catastrophic_state`` for sim
  explosions), so this stay-stand-specific termination lives inline rather
  than as a separate ``mdp/`` subpackage (kept flat per the task's
  agent_context.txt).
  """
  asset: Entity = env.scene[asset_cfg.name]
  uprightness = -asset.data.projected_gravity_b[:, 2]
  return uprightness < math.cos(limit_angle)


def make_stay_stand_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create base stay-stand task configuration on flat ground."""

  ##
  # Observations.
  ##

  actor_terms = {
    "base_lin_vel": ObservationTermCfg(
      func=mdp.base_lin_vel,
      noise=Unoise(n_min=-0.1, n_max=0.1),
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.base_ang_vel,
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
      terms={**actor_terms},
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  ##
  # Actions.
  ##

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.5,  # Override per-robot.
      use_default_offset=True,
    )
  }

  ##
  # Events.
  ##

  events = {
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (-0.05, 0.05),
          "y": (-0.05, 0.05),
          "z": (0.0, 0.01),
          "yaw": (-0.1, 0.1),
        },
        "velocity_range": {},
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (0.0, 0.0),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
  }

  ##
  # Rewards.
  ##

  rewards = {
    # Dominant survival signal: stay alive => gain reward every step.
    "alive": RewardTermCfg(func=mdp.is_alive, weight=3.0),
    # Stay near default joint pose. exp(-mean(error**2/std**2)); std loose enough
    # that the reward is meaningfully positive during exploration.
    "posture": RewardTermCfg(
      func=mdp.posture,
      weight=5.0,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "std": {".*": 0.4},  # Override per-robot.
      },
    ),
    # Strong direct gradient pulling torso back to vertical when tilted.
    "upright": RewardTermCfg(func=mdp.flat_orientation_l2, weight=-3.0),
  }

  ##
  # Terminations.
  ##

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    # Generous angle: 60deg terminates too aggressively during early exploration.
    "fell_over": TerminationTermCfg(
      func=fell_over,
      params={"limit_angle": math.radians(80.0)},
    ),
  }

  ##
  # Assemble.
  ##

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=1,
      extent=2.0,
    ),
    observations=observations,
    actions=actions,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",  # Set per-robot.
      distance=3.0,
      elevation=-10.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=4,
    episode_length_s=20.0,
  )
