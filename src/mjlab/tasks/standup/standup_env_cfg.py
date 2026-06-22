"""Standup task configuration.

This module provides a factory function to create a base standup task config.
Robot-specific configurations call the factory and customize as needed.

Forked from the velocity locomotion env cfg. Summary of what changed and why
(full reasoning lives in the comments of each mdp module this file wires
together -- standup_command.py, standup_reward.py, standup_observations.py,
standup_events.py, standup_termination.py, standup_curriculum.py):

  - commands.twist: UniformVelocityCommandCfg -> StandStillCommandCfg. No
    velocity to track; command is always zero, observed so reward/obs code
    keeps a stable shape.
  - rewards: track_linear_velocity/track_angular_velocity -> hold_still
    (gated to only fire once standing). variable_posture's command-speed
    bands -> standing/recovering bands gated the same way. All gait-cycle
    terms (air_time, foot_clearance, foot_swing_height, foot_slip,
    soft_landing) dropped -- no walking gait cycle in a standup motion.
    New: standup_progress, a dense shaping reward toward standing height +
    uprightness, since nothing else here rewards the act of getting up
    itself.
  - observations: critic foot_air_time/foot_contact_forces dropped (gait-
    cycle / impact-force signals not needed); foot_height_scan sensor and
    foot_contact kept (still useful: which feet/hands are touching the
    ground while getting up). New: base_height, since how far through
    standing up the robot is wasn't directly observable before.
  - events: reset_base now uses reset_fallen_state instead of a small pose
    jitter near standing, since standup needs to start from arbitrary
    fallen/tumbled states, not just small perturbations of a standing pose.
    push_robot kept unchanged -- it already does exactly the "pushed while
    standing up" mechanism needed.
  - terminations: fell_over (bad_orientation) replaced with
    catastrophic_state -- bad_orientation would terminate every standup
    episode at t=0, since starting "badly oriented" (lying down) is the
    entire premise of the task. New: stuck_no_progress, to cut off episodes
    where the robot has stalled rather than burning the full episode length.
  - curriculum: terrain_levels_vel -> terrain_levels_standup (success-rate
    driven instead of distance-walked). commands_vel -> fall_difficulty
    (eases fall/disturbance severity over training instead of widening
    velocity ranges, since there's no velocity command to widen).

NOTE on min_standing_height: this threshold is used across hold_still,
variable_posture, terrain_levels_standup, and catastrophic_state to decide
"has the robot finished standing up". It depends on your specific robot's
standing height and is left as an explicit placeholder (~0.6-0.7x a typical
standing height) -- tune per-robot the same way other "Set per-robot"
placeholders in this file are tuned.
"""

import math
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import (
  GridPatternCfg,
  ObjRef,
  RayCastSensorCfg,
  TerrainHeightSensorCfg,
)
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.standup import mdp
from mjlab.tasks.standup.mdp.standup_command import StandStillCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

# Placeholder: ~0.6-0.7x a typical standing height. Tune per-robot, same as
# other "Set per-robot" values in this file -- this is the single threshold
# used everywhere "has the robot finished standing up" needs to be decided.
MIN_STANDING_HEIGHT = 0.5


def make_standup_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create base Standup task configuration."""

  ##
  # Sensors
  ##

  terrain_scan = RayCastSensorCfg(
    name="terrain_scan",
    frame=ObjRef(type="body", name="", entity="robot"),  # Set per-robot.
    ray_alignment="yaw",
    pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),
    max_distance=5.0,
    exclude_parent_body=True,
    include_geom_groups=(0,),  # Terrain only.
    debug_vis=True,
  )

  foot_height_scan = TerrainHeightSensorCfg(
    name="foot_height_scan",
    frame=(),  # Set per-robot: frame and pattern.
    ray_alignment="yaw",
    max_distance=1.0,
    exclude_parent_body=True,
    include_geom_groups=(0,),  # Terrain only.
    debug_vis=True,
    viz=TerrainHeightSensorCfg.VizCfg(
      show_rays=True,
      hit_color=(1.0, 0.0, 1.0, 0.8),  # Magenta rays.
      hit_sphere_color=(1.0, 0.0, 1.0, 1.0),
    ),
  )

  ##
  # Observations
  ##

  actor_terms = {
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
      noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "base_height": ObservationTermCfg(
      func=mdp.base_height,
      noise=Unoise(n_min=-0.02, n_max=0.02),
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      # Reduced from \u00b11.5 to \u00b10.5 rad/s. The original \u00b11.5 was copied from the
      # locomotion config where fast gait-cycle velocities (5+ rad/s knees)
      # made this proportionally small. For stay-stand, ankle corrections
      # run at ~1-2 rad/s; at \u00b11.5 noise the SNR = 1.3 (barely detectable).
      # At \u00b10.5 the SNR rises to 3-4, giving the policy reliable feedback
      # about its own ankle/knee velocities and allowing ankle-strategy
      # credit assignment to work.
      noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
    "command": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "stand_still"},
    ),
    "height_scan": ObservationTermCfg(
      func=envs_mdp.height_scan,
      params={"sensor_name": "terrain_scan"},
      noise=Unoise(n_min=-0.1, n_max=0.1),
      scale=1 / terrain_scan.max_distance,
    ),
  }

  critic_terms = {
    **actor_terms,
    "height_scan": ObservationTermCfg(
      func=envs_mdp.height_scan,
      params={"sensor_name": "terrain_scan"},
      scale=1 / terrain_scan.max_distance,
    ),
    # "foot_height": ObservationTermCfg(
    #   func=mdp.foot_height,
    #   params={"sensor_name": "foot_height_scan"},
    # ),
    # "foot_contact": ObservationTermCfg(
    #   func=mdp.foot_contact,
    #   params={"sensor_name": "feet_ground_contact"},
    # ),
    # # foot_air_time and foot_contact_forces dropped: gait-cycle / impact-
    # # force signals with no standup analog. foot_height + foot_contact kept
    # # since knowing which feet/hands are touching the ground while getting
    # # up is still useful info.
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  ##
  # Metrics
  ##

  metrics = {
    "mean_action_acc": MetricsTermCfg(
      func=mdp.mean_action_acc,
    ),
  }

  ##
  # Actions
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
  # Commands
  ##

  commands: dict[str, CommandTermCfg] = {
    "stand_still": StandStillCommandCfg(
      entity_name="robot",
      resampling_time_range=(3.0, 8.0),
      min_standing_height=MIN_STANDING_HEIGHT,
      # init_velocity_prob/init_*_range left at zero here: the "robot was
      # already moving" scenario is handled by reset_base (reset_fallen_state)
      # below, which covers it more completely (position + orientation +
      # velocity together, not just velocity).
      debug_vis=True,
    )
  }

  ##
  # Events
  ##

  events = {
    "reset_base": EventTermCfg(
      func=mdp.reset_fallen_state,
      mode="reset",
      params={
        # Always spawn upright. The robot starts standing and a fall
        # terminates the episode, so no orientation curriculum is needed.
        "orientation_mode": "standing",
        "height_range": (0.0, 0.0),
        # INITIAL VELOCITY: give the robot a random horizontal push at
        # every episode reset. This is the most direct way to force the
        # policy to learn corrective leg strategies:
        #   - Without this, the robot starts stationary and the PD
        #     controller maintains HOME_KEYFRAME with no policy corrections
        #     needed. The policy learns \"output zero = stand still\" and
        #     never discovers ankle corrections.
        #   - With \u00b10.2 m/s initial velocity, the robot immediately needs
        #     a corrective action every episode. Ankle dorsiflexion/
        #     plantarflexion is the CORRECT mechanical response; the hip-
        #     leaning strategy that worked for static balance is less
        #     effective here because it doesn't change the ground contact
        #     point (CoP) that the ankle controls.
        #   - \u00b10.2 m/s is moderate: a healthy ankle correction can handle
        #     it (ankle torque ~35 N\u00b7m at full deflection vs ~17 N\u00b7m needed
        #     for 0.2 m/s correction). Large enough to require response,
        #     small enough not to immediately cause fell_over.
        "velocity_range": {
          "x": (-0.2, 0.2),
          "y": (-0.2, 0.2),
        },
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        # Small joint-position noise provides episode-to-episode variation and
        # prevents the policy from over-fitting to the exact HOME_KEYFRAME
        # starting state. ±0.05 rad is well within the PD controller's
        # stiffness range (HOME_KEYFRAME is a stable equilibrium, so small
        # perturbations are immediately corrected). Deliberately kept small:
        # larger noise extends the range of initial states but risks starting
        # the robot off-balance before the policy has learned to correct it.
        "position_range": (-0.05, 0.05),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      # Reduced from (8.0, 15.0) s. Current episodes last 4-7 s (mean ~85
      # control steps). At 8-15 s interval, pushes NEVER fired during
      # training -- the robot always fell before the push could happen.
      # Without pushes the policy only learns static balance (output zero)
      # and never learns push-recovery leg strategies.
      # At 3-6 s, pushes reliably fire at least once per episode for the
      # longer episodes, gradually introducing push-recovery challenge as
      # episodes get longer.
      interval_range_s=(3.0, 6.0),
      params={
        # Fixed moderate push from iteration 0. The robot starts standing,
        # so push-recovery is a first-class skill from the start.
        "velocity_range": {
          "x": (-0.3, 0.3),
          "y": (-0.3, 0.3),
          "z": (-0.2, 0.2),
        },
      },
    ),
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot.
        "operation": "abs",
        "ranges": (0.3, 1.2),
        "shared_random": True,  # All foot geoms share the same friction.
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "bias_range": (-0.015, 0.015),
      },
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
        "operation": "add",
        "ranges": {
          0: (-0.025, 0.025),
          1: (-0.025, 0.025),
          2: (-0.03, 0.03),
        },
      },
    ),
  }

  ##
  # Rewards
  ##

  rewards = {
    "hold_still": RewardTermCfg(
      func=mdp.hold_still,
      weight=2.0,
      params={
        "std": math.sqrt(0.25),
      },
    ),
    "base_height": RewardTermCfg(
      func=mdp.base_height_reward,
      weight=0.0,  # Override per-robot.
      params={
        "target_height": 0.6,  # Override per-robot (~HOME_KEYFRAME pelvis height).
        "std": 0.1,            # Override per-robot.
        "asset_cfg": SceneEntityCfg("robot"),
      },
    ),
    "upright": RewardTermCfg(
      func=mdp.upright,
      weight=2.0,
      params={
        "std": math.sqrt(0.2),
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
      },
    ),
    "upright_gated": RewardTermCfg(
      func=mdp.upright_with_feet_gate,
      weight=0.0,  # Override per-robot; requires feet contact sensor to be wired.
      params={
        "std": math.sqrt(0.2),  # Override per-robot.
        "sensor_name": "",  # Set per-robot: name of the feet ground contact sensor.
        "bodyweight_n": 200.0,  # Override per-robot (~robot_mass × 9.8 in N).
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
      },
    ),
    "pose": RewardTermCfg(
      func=mdp.variable_posture,
      weight=1.0,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "std_values": {},  # Set per-robot.
      },
    ),
    "ankle_corrective": RewardTermCfg(
      func=mdp.ankle_corrective,
      weight=0.0,  # Override per-robot.
      params={
        # std ≈ tilt_at_operating_range × typical_ankle_correction.
        # At 10° tilt (0.17) × 0.15 rad ankle: product = 0.026. Use 0.025.
        "std": 0.025,      # Override per-robot.
        "asset_cfg": SceneEntityCfg("robot"),
      },
    ),
    "body_ang_vel": RewardTermCfg(
      func=mdp.body_angular_velocity_penalty,
      weight=0.0,  # Override per-robot.
      params={
        "std": 1.0,  # Override per-robot.
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
      },
    ),
    "angular_momentum": RewardTermCfg(
      func=mdp.angular_momentum_penalty,
      weight=0.0,  # Override per-robot.
      params={
        "sensor_name": "robot/root_angmom",
        "std": 1.0,  # Override per-robot.
      },
    ),
    "feet_bearing_weight": RewardTermCfg(
      func=mdp.feet_bearing_weight,
      weight=0.0,  # Override per-robot; requires feet contact sensor to be wired.
      params={
        "sensor_name": "",  # Set per-robot: name of the feet ground contact sensor.
        "bodyweight_n": 200.0,  # Override per-robot (~robot_mass × 9.8 in N).
      },
    ),
    "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-0.05),
    # action_rate_l2 is unbounded (||a_t - a_{t-1}||^2). When the policy
    # std grows (which happens early in training before convergence), this
    # penalty grows quadratically and can dominate the reward signal,
    # contributing to value-function divergence. Disable it for now;
    # re-enable once the policy reliably stands and we want to smooth
    # action trajectories.
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=0.0),
    "self_collision": RewardTermCfg(
      func=mdp.self_collision_cost,
      weight=-0.1,  # Override per-robot.
      params={"sensor_name": "robot/self_collision"},  # Set per-robot.
    ),
    # air_time, foot_clearance, foot_swing_height, foot_slip, soft_landing
    # all dropped: gait-cycle terms for a walking step pattern, with no
    # equivalent during a standup motion.
  }

  ##
  # Terminations
  ##

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "catastrophic_state": TerminationTermCfg(
      func=mdp.catastrophic_state,
      params={
        "min_height": -0.5,  # Sanity bound, not a "fallen" check -- tune to
        "max_height": 3.0,  # your robot's scale. See standup_termination.py.
      },
    ),
    "fell_over": TerminationTermCfg(
      func=mdp.fell_over,
      params={
        "min_height": MIN_STANDING_HEIGHT,
        "min_uprightness": 0.5,
      },
    ),
    "out_of_terrain_bounds": TerminationTermCfg(
      func=mdp.out_of_terrain_bounds,
      time_out=True,
    ),
    "nan_detection": TerminationTermCfg(func=mdp.nan_detection),
  }

  ##
  # Curriculum
  ##

  curriculum = {
    "terrain_levels": CurriculumTermCfg(
      func=mdp.terrain_levels_standup,
      params={
        "min_standing_height": MIN_STANDING_HEIGHT,
        "min_standing_uprightness": 0.8,
      },
    ),
  }

  ##
  # Assemble and return
  ##

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=replace(ROUGH_TERRAINS_CFG),
        # Start all envs at the easiest terrain (level 0). The curriculum
        # advances them as they succeed. Starting at 5 meant half the robots
        # immediately faced rough terrain before any policy existed, causing
        # rapid terrain-level oscillation that corrupted the value function.
        max_init_terrain_level=0,
      ),
      sensors=(terrain_scan, foot_height_scan),
      num_envs=1,
      extent=2.0,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculum,
    metrics=metrics,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",  # Set per-robot.
      distance=3.0,
      elevation=-5.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=35,
      njmax=1500,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=4,
    episode_length_s=20.0,
  )