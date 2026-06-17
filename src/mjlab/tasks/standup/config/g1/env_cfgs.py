"""Unitree G1 standup environment configurations.

Forked from the velocity locomotion G1 env_cfgs.py. Most of this file is
robot-specific physical wiring (sensor frames, geom/site/body names, action
scale) that has nothing to do with locomotion vs. standup, so it's
unchanged. The task-coupled parts that needed conversion:

  - twist_cmd (UniformVelocityCommandCfg) -> stand_still_cmd
    (StandStillCommandCfg). Key renamed to match standup_env_cfg.py's
    commands dict ("twist" -> "stand_still").
  - cfg.rewards["pose"] std_standing/std_walking/std_running (3 bands) ->
    std_standing/std_recovering (2 bands), matching variable_posture's
    standing/recovering gate instead of a command-speed gate. The original
    per-robot std *values* are real tuning data, not locomotion-specific in
    themselves, so they're preserved: std_standing maps directly, and
    std_recovering is seeded from the old std_walking numbers (looser
    tolerance, appropriate for the large joint excursions of getting up;
    the old std_running numbers are dropped since there's no "running"
    regime in standup).
  - air_time reward weight override, foot_clearance/foot_slip site_names
    wiring: removed -- these reward terms don't exist in standup_env_cfg.py
    (dropped as gait-cycle-only terms upstream).
  - self_collisions: the original added a *second*, differently-named
    reward term ("self_collisions", plural) on top of the generic
    "self_collision" placeholder already defined in standup_env_cfg.py's
    rewards dict, effectively shadowing it without removing it. Converted
    to update the existing "self_collision" entry's params/weight instead
    of creating a duplicate key.
  - Policy runner reference: any place that imported/used
    VelocityOnPolicyRunner should use StandupOnPolicyRunner instead (see
    standup_runner.py). This file itself doesn't construct the runner, so
    there's nothing to change here directly, but flagging it since the
    leading comment in the original called it out -- check your training
    entry-point script for the actual import.

feet_ground_contact's track_air_time=True is kept even though nothing in
standup_reward.py reads air time anymore -- harmless to leave enabled, and
keeps the sensor cfg available if you reintroduce an air-time-derived
signal later.
"""

from mjlab.asset_zoo.robots import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.standup.mdp.standup_command import StandStillCommandCfg
from mjlab.tasks.standup.standup_env_cfg import make_standup_env_cfg


def unitree_g1_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 rough terrain Standup configuration."""
  cfg = make_standup_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 70

  cfg.scene.entities = {"robot": get_g1_robot_cfg()}

  # Set raycast sensor frame to G1 pelvis.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = "pelvis"

  site_names = ("left_foot", "right_foot")
  geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
  )

  # Wire foot height scan to per-foot sites.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=s, entity="robot") for s in site_names
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.03, num_samples=6)

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_ACTION_SCALE

  cfg.viewer.body_name = "torso_link"

  stand_still_cmd = cfg.commands["stand_still"]
  assert isinstance(stand_still_cmd, StandStillCommandCfg)
  stand_still_cmd.viz.z_offset = 1.15

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  # Rationale for std values:
  # - Knees/hip_pitch get the loosest std to allow natural leg bending during
  #   the get-up motion.
  # - Hip roll/yaw stay tighter to prevent excessive lateral sway once
  #   standing and keep the recovered pose stable.
  # - Ankle roll is very tight for balance; ankle pitch looser for ground
  #   contact/push-off during recovery.
  # - Waist roll/pitch stay tight to keep the torso upright and stable once
  #   standing.
  # - Shoulders/elbows get moderate freedom -- arms are often used to push
  #   off the ground or counterbalance during standup.
  # - Wrists are loose (0.3) since they don't affect balance much.
  # std_recovering values are seeded from the original walking std values
  # (~the same magnitude of motion freedom needed for a get-up motion as for
  # a walking stride); the original running std values are dropped, since
  # standup has no equivalent "running" regime.
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_recovering"] = {
    # Lower body.
    r".*hip_pitch.*": 0.3,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.35,
    r".*ankle_pitch.*": 0.25,
    r".*ankle_roll.*": 0.1,
    # Waist.
    r".*waist_yaw.*": 0.2,
    r".*waist_roll.*": 0.08,
    r".*waist_pitch.*": 0.1,
    # Arms.
    r".*shoulder_pitch.*": 0.15,
    r".*shoulder_roll.*": 0.15,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.15,
    r".*wrist.*": 0.3,
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso_link",)

  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["angular_momentum"].weight = -0.02

  # Wire the generic self_collision placeholder (defined in
  # standup_env_cfg.py) to this robot's actual sensor name and a stronger
  # weight, rather than adding a second duplicate reward term.
  cfg.rewards["self_collision"].params["sensor_name"] = self_collision_cfg.name
  cfg.rewards["self_collision"].params["force_threshold"] = 10.0
  cfg.rewards["self_collision"].weight = -1.0

  # air_time, foot_clearance, foot_slip overrides removed: these reward
  # terms don't exist in standup_env_cfg.py (dropped upstream as gait-cycle-
  # only terms with no standup analog).

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def unitree_g1_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain standup configuration."""
  cfg = unitree_g1_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensor and height scan (no terrain to scan).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  cfg.terminations.pop("out_of_terrain_bounds", None)

  # Disable terrain curriculum (not present in play mode since rough clears all).
  cfg.curriculum.pop("terrain_levels", None)

  # Note: the original here overrode twist_cmd.ranges.lin_vel_x/ang_vel_z
  # for play mode -- StandStillCommandCfg has no velocity ranges to widen
  # (command is always zero), so there is nothing to override for
  # stand_still_cmd in play mode. If you want play mode to test against
  # harder pushes/falls specifically, override cfg.events["push_robot"] or
  # cfg.events["reset_base"] params here instead.

  return cfg