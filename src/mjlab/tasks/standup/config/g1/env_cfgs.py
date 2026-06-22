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
  # SHRINK THE ACTION SCALE FOR STANDUP. G1_ACTION_SCALE is tuned for
  # locomotion (knee ~0.35 rad per unit raw action), which is fine for a
  # walking gait. But for standup -- especially stage 0 where the robot
  # spawns standing and the optimal policy is "output ~0 to hold default
  # pose" -- this scale is way too large for bootstrapping. At iter 0
  # the actor MLP outputs random values with magnitude ~1.0 per dim, so
  # every joint is commanded ~0.35 rad off default at step 0 *before* any
  # learning has happened. The robot is jerked out of the default pose
  # immediately and the policy has to first un-learn this random offset
  # before any pose/upright reward can fire. Multiplying by 0.25 makes
  # the worst-case random-init action only ~0.09 rad off default per
  # joint, which the actuators can hold near default pose, so pose/
  # upright rewards fire from iter 0 and bootstrap learning. Scale this
  # back up once the policy can reliably hold default pose, since harder
  # curriculum stages (recovery from prone) need larger action ranges.
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

  # POSE STD UNIFORM: Tight std=0.25 for all joints stabilises early training
  # by keeping the policy near HOME before the value function converges.
  # The gated upright reward (upright_gated) provides a stronger gradient for
  # corrective leg movements than any pose std relaxation could, because the
  # gate turns off the entire 10-unit upright signal when feet leave the ground
  # -- this overwhelms the 2-unit pose gradient opposing leg corrections.
  # Loose leg std (tried previously) removed the stabilising pose gradient
  # for legs, causing random policy perturbations to destabilise the robot
  # immediately (fell_over=1.0 from iter 0).
  cfg.rewards["pose"].params["std_values"] = {".*": 0.25}

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso_link",)

  # UPRIGHT STD calibrated to the robot's 20° operating tilt. See the long
  # comment in the session 3 notes for the gradient analysis. std=sqrt(0.117)
  # gives exp(-1)=0.37 kernel at 20° vs exp(-2.3)≈0.1 with old std=sqrt(0.05).
  # Applied to BOTH upright (weight 0, kept for reference) and upright_gated.
  _upright_std = 0.117 ** 0.5  # sin²(20°) = 0.117
  cfg.rewards["upright"].params["std"] = _upright_std
  cfg.rewards["upright"].weight = 0.0  # Replaced by upright_gated below.

  # UPRIGHT GATED BY FEET CONTACT: the primary balance signal (weight 10).
  # Returns upright_score × feet_gate where:
  #   upright_score = exp(-xy²/std²) ∈ [0, 1] -- PELVIS uprightness
  #   feet_gate     = clamp(foot_force / bodyweight_n, 0, 1) ∈ [0, 1]
  #
  # KEY DESIGN DECISIONS vs previous iteration:
  #
  # 1. Track PELVIS (root link), NOT torso_link.
  #    Torso tracking allowed the "waist-compensation" local optimum:
  #    robot bends at the waist (torso stays upright) while the pelvis falls,
  #    earning full upright reward without using any leg joints.
  #    Pelvis tracking makes this impossible: only leg corrections
  #    (ankle/knee/hip) can keep the pelvis upright.
  #    body_names=() → asset_cfg.body_ids is falsy → uses root_link_quat_w.
  #
  # 2. bodyweight_n = G1_MASS × 9.8 / 2 = 112.5 N (half-bodyweight threshold).
  #    Using full bodyweight (225 N) means the gate drops when one foot lifts
  #    to step, penalising stepping and discouraging the stepping reflex.
  #    Using half-bodyweight: gate = 1.0 whenever either foot bears full load.
  #      both feet standing:    total_force ≈ 225 N → gate = clamp(2.0, 0, 1) = 1.0
  #      stepping (one foot):   total_force ≈ 112 N → gate = clamp(1.0, 0, 1) = 1.0
  #      airborne:              total_force = 0 N   → gate = 0.0
  #    This allows free stepping to catch pushes without any reward penalty.
  cfg.rewards["upright_gated"].params["asset_cfg"].body_names = ()  # pelvis / root link
  cfg.rewards["upright_gated"].params["std"] = _upright_std
  cfg.rewards["upright_gated"].params["sensor_name"] = feet_ground_cfg.name
  cfg.rewards["upright_gated"].params["bodyweight_n"] = 112.5  # half-bodyweight
  cfg.rewards["upright_gated"].weight = 10.0

  # Reward balance (per-step max when standing with both feet down):
  #   upright_gated: 10.0 × 1.0 × 1.0 = 10.0  (PRIMARY — pelvis + legs required)
  #   feet_bearing:   5.0 × 0.76     =  3.8  (extra incentive for symmetric loading)
  #   pose:           2.0 × 1.0      =  2.0  (regularization, prevents extreme joints)
  #   hold_still:     2.0 × ~0.5     = ~1.0  (penalises large base velocity / falls)
  #   Total max per step ≈ 16.8
  cfg.rewards["pose"].weight = 2.0

  # body_ang_vel and angular_momentum now use bounded exp(-x²/std²) kernels
  # (see rewards.py), but they are stability-shaping signals for a policy
  # that already stands. Enable them once the robot can stand reliably;
  # leaving them active from iteration 0 adds value-function noise that
  # fights bootstrapping. Same for action_rate_l2.
  cfg.rewards["body_ang_vel"].params["std"] = 1.0
  cfg.rewards["body_ang_vel"].weight = 0.0  # Re-enable (e.g. -0.05) once robot can stand.
  cfg.rewards["angular_momentum"].params["std"] = 1.0
  cfg.rewards["angular_momentum"].weight = 0.0  # Re-enable (e.g. -0.01) once robot can stand.
  cfg.rewards["action_rate_l2"].weight = 0.0  # Re-enable (e.g. -0.01) once robot can stand.

  # G1 standing height overrides -- terrain curriculum and fell_over
  # termination both use the same "is standing" threshold.
  cfg.curriculum["terrain_levels"].params["min_standing_height"] = _G1_MIN_STANDING_HEIGHT
  # TIGHTEN PROMOTION CRITERION: default min_standing_uprightness=0.8 (cos37°)
  # is too loose -- the robot got promoted at only 37° average tilt, immediately
  # failed on harder terrain, and repeated in a ~600-step oscillation cycle that
  # prevents consistent learning. Setting 0.95 (cos18°) requires near-perfect
  # uprightness for promotion so only a genuinely stable policy advances.
  cfg.curriculum["terrain_levels"].params["min_standing_uprightness"] = 0.95

  # fell_over thresholds: these must NOT be close to the HOME_KEYFRAME height
  # (~0.72-0.75 m with bent knees). min_height=0.65 is only 7-10 cm below
  # standing height -- random policy actions perturb the pelvis enough to
  # cross that threshold in a handful of steps, collapsing every episode.
  # 0.45 m (57% of standing height) is clearly fallen, not just crouching.
  # min_uprightness=0.2 allows up to ~80 degrees of tilt before terminating;
  # the robot needs this room to discover the standing attractor before it
  # learns to correct itself. Tighten both once the robot reliably stands.
  cfg.terminations["fell_over"].params["min_height"] = 0.45
  cfg.terminations["fell_over"].params["min_uprightness"] = 0.2

  # Wire the generic self_collision placeholder (defined in
  # standup_env_cfg.py) to this robot's actual sensor name and a stronger
  # weight, rather than adding a second duplicate reward term.
  cfg.rewards["self_collision"].params["sensor_name"] = self_collision_cfg.name
  cfg.rewards["self_collision"].params["force_threshold"] = 10.0
  cfg.rewards["self_collision"].weight = -0.2

  # FEET BEARING WEIGHT: reward ground-reaction force through feet.
  # When both feet are properly planted, total force ≈ bodyweight (225 N for
  # G1 ~23 kg), giving tanh(1) ≈ 0.76 per step × weight=5 ≈ 3.8.
  # When the robot abandons legs for arm-only balance, foot forces drop to 0
  # and this reward collapses, directly opposing that failure mode.
  # The feet_ground_contact sensor exposes two primaries (left and right
  # ankle subtrees) with reduce="netforce", so data.force is [B, 2, 3].
  cfg.rewards["feet_bearing_weight"].params["sensor_name"] = feet_ground_cfg.name
  cfg.rewards["feet_bearing_weight"].params["bodyweight_n"] = 225.0  # ~23 kg × 9.8
  cfg.rewards["feet_bearing_weight"].weight = 5.0

  # BASE HEIGHT REWARD: continuous height gradient from standing (0.72 m) down
  # to the fell_over floor (0.45 m). Without this, the robot can earn full
  # upright_gated while slowly squatting because pelvis orientation alone does
  # not penalise downward displacement. Gaussian centred at HOME_KEYFRAME actual
  # pelvis height (~0.72 m with bent knees); std=0.10 m gives exp(-1)≈0.37 at
  # 10 cm below target and ≈0 near the 0.45 m fell_over threshold.
  # This adds a direct gradient for legs to support body weight at the right height.
  cfg.rewards["base_height"].params["target_height"] = 0.72  # HOME actual pelvis z
  cfg.rewards["base_height"].params["std"] = 0.10
  cfg.rewards["base_height"].weight = 3.0

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