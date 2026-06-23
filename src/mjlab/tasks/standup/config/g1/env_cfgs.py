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

import math

from mjlab.asset_zoo.robots import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import HOME_KEYFRAME
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

# Minimum height to be considered "standing" (~82 % of full standing height).
# Used by the terrain curriculum success check and the fell_over termination.
_G1_MIN_STANDING_HEIGHT: float = 0.65


def unitree_g1_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 rough terrain Standup configuration."""
  cfg = make_standup_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 70

  # Use HOME_KEYFRAME as the robot's default joint state instead of the
  # KNEES_BENT_KEYFRAME that get_g1_robot_cfg() returns by default.
  #
  # ROOT CAUSE of the balance failure: KNEES_BENT (hip_pitch=-0.312,
  # knee=0.669, ankle=-0.363) is a forward-leaning crouch. The
  # gravitational torque at the ankle is roughly 20 N*m (20 kg body
  # * 9.8 * 0.1 m forward CoM offset). The ankle PD stiffness is only
  # ~10.5 N*m/rad (STIFFNESS_5020 * 2 = 2 * 5.25). To resist 20 N*m the
  # ankle would need to deflect 20/10.5 = 1.9 rad from its target --
  # well past the joint limit -- so the robot falls in the first few sim
  # steps before any policy gradient exists, and spends ~95% of every
  # episode lying at ~50 degrees from vertical.
  #
  # HOME_KEYFRAME (hip_pitch=-0.1, knee=0.3, ankle=-0.2) is a near-upright
  # stance where the gravitational torque at the ankle is ~2 N*m, well
  # within the stiffness range. The robot can hold this pose with the PD
  # controller and gives the policy a standing robot to actually learn
  # balance from, not a robot that is already on the floor at step 1.
  _robot_cfg = get_g1_robot_cfg()
  _robot_cfg.init_state = HOME_KEYFRAME
  cfg.scene.entities = {"robot": _robot_cfg}

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
  # SHRINK THE ACTION SCALE UNIFORMLY. G1_ACTION_SCALE is tuned for
  # locomotion (knee ~0.35 rad per unit raw action), which is far too large
  # for a "hold default pose" balance task. At iter 0 the actor MLP outputs
  # random values ~1.0 per dim, so without shrinking every joint is
  # commanded ~0.35 rad off default at step 0 *before* any learning.
  # Multiplying by 0.25 makes the worst-case random-init action only ~0.09
  # rad off default per joint, which the actuators can hold near HOME, so
  # the upright / pose / hold_still rewards fire from iter 0 and bootstrap
  # learning. A uniform multiplier (no per-joint asymmetry) keeps the
  # exploration distribution shaped like the prior and avoids the leg-vs-
  # arm exploration imbalance that previously biased the policy toward
  # large leg motions before it had learned even basic stand-still.
  joint_pos_action.scale = {k: v * 0.25 for k, v in G1_ACTION_SCALE.items()}

  cfg.viewer.body_name = "torso_link"

  stand_still_cmd = cfg.commands["stand_still"]
  assert isinstance(stand_still_cmd, StandStillCommandCfg)
  stand_still_cmd.viz.z_offset = 1.15
  # Gate tracking-error metric to G1 standing height so it doesn't penalize
  # the velocity incurred while getting up.
  stand_still_cmd.min_standing_height = _G1_MIN_STANDING_HEIGHT

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  # =====================================================================
  # MINIMAL STAY-STAND REWARD SHAPING
  # =====================================================================
  # Goal: the simplest reward structure that teaches the robot to stay
  # standing from a stable upright start. No prescriptive constraints on
  # HOW to balance -- arms, torso, hips, knees, ankles are all valid
  # tools. Earlier iterations of this file added many task-specific
  # signals (upright_gated, feet_bearing_weight, ankle_corrective,
  # body_ang_vel, angular_momentum) to "force" a particular leg-based
  # strategy. Every one of them either competed with the basic balance
  # gradient or directly penalised legitimate corrective motions, so the
  # policy collapsed to either lock-down or torso-swing failure modes.
  # The minimal set below has FOUR positive shaping rewards (upright,
  # base_height, pose, hold_still) and the two small standard penalties
  # (dof_pos_limits, self_collision). That's it.
  #
  # Per-step reward at perfect HOME_KEYFRAME standing:
  #   upright         +5.0 × 1.0  =  5.0   (primary: torso vertical)
  #   base_height     +2.0 × 1.0  =  2.0   (pelvis at ~0.72 m)
  #   pose            +1.0 × ~1.0 = ~1.0   (joints near HOME)
  #   hold_still      +1.0 × ~1.0 = ~1.0   (base at rest)
  #   dof_pos_limits  -0.05 × 0   =  0.0
  #   self_collision  -0.1 × 0    =  0.0
  #   Total per-step  ≈ 9.0   (cumulative over 20s × 50Hz episode ≈ 9000)
  # =====================================================================

  # UPRIGHT: primary balance signal. Track torso_link uprightness (NOT
  # pelvis) -- torso is what visually defines "standing", and tracking it
  # allows the natural human-like strategy of hip flex to keep the torso
  # upright while the pelvis shifts slightly. The user explicitly wants
  # any body part to be usable for balance.
  # std=sqrt(0.2)≈0.45 gives exp(-1)=0.37 at ~26° tilt and exp(-0.25)=0.78
  # at ~13° tilt -- a smooth gradient across the full operating range
  # without collapsing to zero at the tilts the policy actually visits.
  cfg.rewards["upright"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["upright"].params["std"] = math.sqrt(0.2)
  cfg.rewards["upright"].weight = 5.0

  # POSE: gentle attractor toward HOME_KEYFRAME on every joint with a
  # single uniform std. Per-joint std partitions are over-engineering for
  # this task -- the only role of pose here is to discourage drift away
  # from the default arm/leg configuration when the policy has no other
  # reason to move. std=0.3 is loose enough that small corrective motions
  # (knee flex, ankle dorsiflexion, hip lean) barely affect the kernel
  # while large arm flailing or full squatting noticeably reduces it.
  cfg.rewards["pose"].params["std_values"] = {".*": 0.3}
  cfg.rewards["pose"].weight = 1.0

  # HOLD_STILL: gentle preference for low base velocity. Weight 1.0 (not
  # 0.5 from session 5, not 2.0 from earlier) is the lightest version
  # that still rewards stillness without out-weighing a corrective
  # motion. With no pushes in the base task, this is a free reward most
  # of the time -- the policy just needs to keep the pelvis still.
  cfg.rewards["hold_still"].weight = 1.0

  # BASE_HEIGHT: keep pelvis at HOME_KEYFRAME actual height (~0.72 m
  # with bent knees). Without this, the policy can earn full upright +
  # pose by slowly sinking into a deep squat. Gaussian centred at 0.72 m
  # with std=0.10 m gives exp(-1)≈0.37 at 10 cm below target.
  cfg.rewards["base_height"].params["target_height"] = 0.72
  cfg.rewards["base_height"].params["std"] = 0.10
  cfg.rewards["base_height"].weight = 2.0

  # All prescriptive "use legs / don't swing torso" terms DISABLED. The
  # user's brief explicitly allows balance via arms, torso, hips, knees,
  # ankles or feet. The right place for the policy to learn HOW to
  # balance is the policy gradient, not a hand-engineered reward menu.
  cfg.rewards["upright_gated"].params["asset_cfg"].body_names = ()
  cfg.rewards["upright_gated"].params["std"] = math.sqrt(0.2)
  cfg.rewards["upright_gated"].params["sensor_name"] = feet_ground_cfg.name
  cfg.rewards["upright_gated"].params["bodyweight_n"] = 225.0
  cfg.rewards["upright_gated"].weight = 0.0

  cfg.rewards["feet_bearing_weight"].params["sensor_name"] = feet_ground_cfg.name
  cfg.rewards["feet_bearing_weight"].params["bodyweight_n"] = 225.0
  cfg.rewards["feet_bearing_weight"].weight = 0.0

  cfg.rewards["ankle_corrective"].params["std"] = 0.025
  cfg.rewards["ankle_corrective"].weight = 0.0

  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["body_ang_vel"].params["std"] = 1.0
  cfg.rewards["body_ang_vel"].weight = 0.0

  cfg.rewards["angular_momentum"].params["std"] = 2.0
  cfg.rewards["angular_momentum"].weight = 0.0

  cfg.rewards["action_rate_l2"].weight = 0.0

  # Wire the generic self_collision placeholder to this robot's sensor.
  cfg.rewards["self_collision"].params["sensor_name"] = self_collision_cfg.name
  cfg.rewards["self_collision"].params["force_threshold"] = 10.0
  cfg.rewards["self_collision"].weight = -0.1

  # =====================================================================
  # TERMINATION + CURRICULUM
  # =====================================================================
  # fell_over: loose thresholds so the policy has room to recover from
  # large tilts before the episode ends. min_height=0.40 m is 55% of
  # standing height -- clearly fallen, not a squat. min_uprightness=0.2
  # allows up to ~80° of tilt before termination, giving the policy time
  # to discover that "tilting + corrective motion" still earns reward.
  # Tighten once Episode_Reward/upright > 3.0 sustained.
  cfg.terminations["fell_over"].params["min_height"] = 0.40
  cfg.terminations["fell_over"].params["min_uprightness"] = 0.2

  # Terrain curriculum: loose promotion criterion. 0.7 = cos45° -- the
  # robot only needs to end the episode with reasonable uprightness to
  # advance, not perfect uprightness. Tighten only when the easier
  # terrains are consistently solved. (For the flat task this whole
  # curriculum is removed below.)
  cfg.curriculum["terrain_levels"].params["min_standing_height"] = 0.55
  cfg.curriculum["terrain_levels"].params["min_standing_uprightness"] = 0.7

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