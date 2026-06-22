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
  # LEG SCALE INCREASE for the joints that actually do balance work.
  # G1_ACTION_SCALE is effort/stiffness based; at the global ×0.25 multiplier
  # the per-step random exploration on knee/hip_pitch is ≈0.026 rad
  # (init_std=0.4 × 0.25 × ≈0.35 rad/unit), so the random walk over a 24-step
  # rollout reaches only ≈0.13 rad of leg flexion -- far below the ≈0.3 rad
  # of knee bend a real corrective step requires. The policy therefore never
  # samples a successful step during exploration, never sees the value of a
  # step, and collapses to "output zero". Bumping ankle, hip_pitch and knee
  # to ×0.5 doubles per-step exploration and lets a multi-step random walk
  # cover the ±0.3 rad range needed for stepping/squatting corrections.
  # Arms, waist, hip_yaw, hip_roll stay at ×0.25 (no need for large excursions;
  # tight scale keeps upper-body cheats and lateral leg wobble suppressed).
  joint_pos_action.scale[".*_ankle_pitch_joint"] = (
    G1_ACTION_SCALE[".*_ankle_pitch_joint"] * 0.5
  )
  joint_pos_action.scale[".*_ankle_roll_joint"] = (
    G1_ACTION_SCALE[".*_ankle_roll_joint"] * 0.5
  )
  joint_pos_action.scale[".*_hip_pitch_joint"] = (
    G1_ACTION_SCALE[".*_hip_pitch_joint"] * 0.5
  )
  joint_pos_action.scale[".*_knee_joint"] = (
    G1_ACTION_SCALE[".*_knee_joint"] * 0.5
  )

  cfg.viewer.body_name = "torso_link"

  stand_still_cmd = cfg.commands["stand_still"]
  assert isinstance(stand_still_cmd, StandStillCommandCfg)
  stand_still_cmd.viz.z_offset = 1.15
  # Gate tracking-error metric to G1 standing height so it doesn't penalize
  # the velocity incurred while getting up.
  stand_still_cmd.min_standing_height = _G1_MIN_STANDING_HEIGHT

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  # POSE STD PER-JOINT PARTITION (session 5 fix).
  # The previous uniform std=0.25 was killing dynamic balance: any knee
  # flexion past ≈0.45 rad and any hip swing past ≈0.45 rad collapsed the
  # pose kernel, so the policy converged to "lock all joints near HOME".
  # The robot ended up using only static stiffness to balance and never
  # learned stepping or knee-flexion corrections.
  #
  # New scheme: keep arms and waist tight (suppresses arm-flailing and
  # waist-bending balance cheats that previously emerged), but allow legs
  # to move freely. Knee std=0.6 makes a full ±0.5 rad squat barely affect
  # the kernel; hip_pitch std=0.5 allows full stepping range. The
  # task-level rewards (upright_gated, base_height, feet_bearing_weight,
  # ankle_corrective) provide the actual "return to standing pose" signal
  # for legs -- much stronger and more correctly oriented than a pose
  # attractor toward HOME could be.
  cfg.rewards["pose"].params["std_values"] = {
    # Arms: tight -- arm flailing should never be a balance strategy.
    ".*_shoulder_pitch_joint": 0.15,
    ".*_shoulder_roll_joint": 0.15,
    ".*_shoulder_yaw_joint": 0.15,
    ".*_elbow_joint": 0.15,
    ".*_wrist_roll_joint": 0.15,
    ".*_wrist_pitch_joint": 0.15,
    ".*_wrist_yaw_joint": 0.15,
    # Waist: very tight -- waist-bending is the "upper-body compensation"
    # cheat that lets the torso stay vertical while the pelvis falls.
    "waist_yaw_joint": 0.10,
    "waist_pitch_joint": 0.10,
    "waist_roll_joint": 0.10,
    # Hips: loose on pitch (stepping), moderate on roll (lateral), tight
    # on yaw (don't want legs twisting in/out for no reason).
    ".*_hip_pitch_joint": 0.5,
    ".*_hip_roll_joint": 0.4,
    ".*_hip_yaw_joint": 0.2,
    # Knees: very loose -- full bend range is a legitimate balance tool.
    ".*_knee_joint": 0.6,
    # Ankles: moderate -- corrections needed, but joint range itself is
    # narrow (±0.26 rad on roll, ±0.87/0.52 on pitch).
    ".*_ankle_pitch_joint": 0.4,
    ".*_ankle_roll_joint": 0.3,
  }

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

  # Reward balance (per-step values at perfect standing, both feet bearing load):
  #   upright_gated:  +10.0 × 1.0 × 1.0  = 10.0  (PRIMARY: pelvis upright + feet planted)
  #   base_height:     +3.0 × 1.0        =  3.0  (pelvis at ~0.72 m above terrain)
  #   feet_bearing:    +5.0 × 0.76       =  3.8  (both feet bearing full bodyweight)
  #   pose:            +2.0 × ~0.8       = ~1.6  (joints near HOME_KEYFRAME)
  #   hold_still:      +2.0 × ~0.7       = ~1.4  (low base velocity when stable)
  #   body_ang_vel:    -0.05 × 1.0       ≈  0.0  (still torso → no penalty)
  #   action_rate_l2:  -0.01 × ~0        ≈  0.0  (smooth actions → no penalty)
  #   self_collision:  -0.2 × 0          =  0.0  (no collisions when stable)
  #   Total positive max per step ≈ 19.8
  #
  # Arm-only / upper-body-only balance (feet off ground):
  #   upright_gated gate = 0 → reward 0; feet_bearing = 0; base_height = varies
  #   Total ≈ 1-3 vs 19.8 for proper balance (6-20× incentive for leg use)
  cfg.rewards["pose"].weight = 2.0

  # STABILITY-SHAPING SIGNALS — penalise torso-swing balance strategy.
  #
  #   body_ang_vel (-0.5): penalises torso roll/pitch angular velocity, tracking
  #     torso_link's own ang_vel_xy. TORSO ROTATION was the policy's preferred
  #     balance shortcut — swinging the upper body shifts angular momentum and
  #     momentarily rights the pelvis faster than leg corrections can. This
  #     term taxes that strategy directly.
  #
  #     CRITICAL HISTORY: in earlier runs this term had its KERNEL INVERTED in
  #     rewards.py. The function returned exp(-xy²/std²) (1 at rest, 0 at chaos),
  #     and with negative weight that meant "more spinning = less penalty",
  #     i.e. the term ACTIVELY REWARDED the torso-swing failure mode it was
  #     meant to suppress. The kernel is now (1 - exp(-xy²/std²)) (0 at rest,
  #     saturates to 1 at chaos), and the weight magnitude is bumped from -0.05
  #     to -0.5 so the corrected penalty is large enough (~0.4/step at 2 rad/s
  #     torso swing) to outweigh whatever short-term upright_gated boost the
  #     swing strategy buys.
  #
  #   angular_momentum (-0.5): bounded penalty on whole-body angular momentum.
  #     Had the same inverted-kernel bug in rewards.py. Now fixed and enabled —
  #     gives an additional brake on any rotation-based balance strategy
  #     (not just torso pitch/roll, but also waist twist / arm flailing).
  #
  #   action_rate_l2 (-0.01): penalises ||a_t - a_{t-1}||². Kept small; the
  #     mean_action_acc=6.5 chaos pattern earlier in training was largely
  #     driven by the same torso-swing strategy, so fixing the angular-velocity
  #     penalties should reduce action chaos as a side effect.
  #
  # SESSION 5 TUNING: angular_momentum.std bumped 1.0 -> 2.0. The old std=1.0
  # saturated at ≈1 kg·m²/s which is roughly the angular momentum of ONE
  # SWINGING LEG during a corrective step. That meant the penalty fired
  # full-strength against legitimate stepping motions, not just torso swings.
  # body_ang_vel std unchanged because it only tracks TORSO link pitch/roll
  # angular velocity -- legs swinging do not rotate the torso, so it does not
  # need to be loosened to permit stepping.
  cfg.rewards["body_ang_vel"].params["std"] = 1.0
  cfg.rewards["body_ang_vel"].weight = -0.5
  cfg.rewards["angular_momentum"].params["std"] = 2.0
  cfg.rewards["angular_momentum"].weight = -0.5
  cfg.rewards["action_rate_l2"].weight = -0.01

  # SESSION 5: hold_still weight 2.0 -> 0.5. hold_still penalises pelvis
  # linear+angular velocity -- exactly what stepping motions produce. At
  # weight 2.0 a corrective 0.5 m/s step cost ≈1.3/step (kernel drops from
  # 1.0 to 0.37), which over a 5-step stepping window outweighed the value
  # of avoiding a small push. At 0.5 the cost is ≈0.3/step, soft enough to
  # not block stepping but still preferring stillness when no push is
  # present. upright_gated + base_height + pose still pin the robot to its
  # initial position when nothing is disturbing it.
  cfg.rewards["hold_still"].weight = 0.5

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
  # min_uprightness=0.35 (70° max tilt): was 0.2 (80°) during early bootstrap
  # when the robot needed maximum room to discover the standing attractor.
  # Now that the robot can stand (episode length 85+, upright_gated 0.8+),
  # tightening prevents the policy from accumulating rewards while nearly
  # horizontal, which corrupts the value function with late-fall states.
  cfg.terminations["fell_over"].params["min_height"] = 0.45
  cfg.terminations["fell_over"].params["min_uprightness"] = 0.35

  # Wire the generic self_collision placeholder (defined in
  # standup_env_cfg.py) to this robot's actual sensor name and a stronger
  # weight, rather than adding a second duplicate reward term.
  cfg.rewards["self_collision"].params["sensor_name"] = self_collision_cfg.name
  cfg.rewards["self_collision"].params["force_threshold"] = 10.0
  cfg.rewards["self_collision"].weight = -0.2

  # FEET BEARING WEIGHT: reward ground-reaction force through feet.
  # When the robot abandons legs for arm-only balance, foot forces drop to 0
  # and this reward collapses, directly opposing that failure mode.
  # The feet_ground_contact sensor exposes two primaries (left and right
  # ankle subtrees) with reduce="netforce", so data.force is [B, 2, 3].
  #
  # SESSION 5: bodyweight_n 225 -> 112.5 (half-bodyweight). At 225 N
  # (full bodyweight), tanh(force/225) gives 0.76 when both feet bear full
  # bodyweight but only 0.46 when one foot lifts to step (single-foot stance
  # at 112 N). That ≈0.3/step drop × weight=5 = 1.5/step penalty on stepping
  # was actively training the policy NOT to step. At 112.5 N, tanh saturates
  # near 1.0 for both-feet stance (~0.96) AND for single-foot stance bearing
  # full bodyweight (~0.76), so stepping no longer costs reward. Only true
  # leg-abandonment (both feet airborne) collapses the term.
  cfg.rewards["feet_bearing_weight"].params["sensor_name"] = feet_ground_cfg.name
  cfg.rewards["feet_bearing_weight"].params["bodyweight_n"] = 112.5  # half-bodyweight
  cfg.rewards["feet_bearing_weight"].weight = 5.0
  # ANKLE CORRECTIVE: direct, same-step reward for ankle_pitch/roll joints
  # being in the corrective direction for the current pelvis tilt.
  # WHY THIS FIXES THE HIP-STRATEGY PROBLEM:
  #   Hip corrections show up in upright_gated in 1 step (direct kinematics).
  #   Ankle corrections take 3-5 steps (CoP shift \u2192 GRF \u2192 pelvis acceleration).
  #   PPO correctly prefers the faster signal, so it learns hips over ankles.
  #   This reward fires in the SAME STEP as the ankle moves, breaking the
  #   timing asymmetry: ankle in correct position now earns reward NOW,
  #   regardless of whether the pelvis has had time to respond.
  # SIGNAL: reward = clamp(-tilt_x \u00d7 mean(ankle_pitch - HOME) / std, 0, 1)
  #   Forward tilt + dorsiflexion: positive \u2192 rewarded  \u2713
  #   Standing + ankle at HOME: zero \u2192 no spurious incentive  \u2713
  #   Wrong direction: zero (clamped, not penalised)  \u2713
  cfg.rewards["ankle_corrective"].params["std"] = 0.025
  cfg.rewards["ankle_corrective"].weight = 3.0
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