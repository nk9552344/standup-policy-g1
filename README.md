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

python -m mjlab.scripts.train Mjlab-Standup-Flat-Unitree-G1 --gpu-ids None

 uv run play Mjlab-StayStand-Flat-Unitree-G1 \
  --checkpoint-file logs/rsl_rl/g1_staystand/2026-06-21_21-58-46/model_4000.pt \ 
  --num-envs 2

https://github.com/InternRobotics/HoST

https://wandb.ai/nk9552344-infosys/mjlab/runs/rawno0s9?nw=nwusernk9552344


## The core conceptual shift

The recovery policy's central idea is "start fallen, get up, then hold still." The stay-stand policy's central idea is "start standing, don't fall, and if pushed, recover balance fast." This changes three things structurally: (1) resets must always start from standing, (2) the dense `standup_progress` shaping reward disappears entirely because there is no "getting up" arc to shape, and (3) a fall is now a *termination*, not a *starting condition*.

---

## Phase 1 — `rewards.py` and `terminations.py`

These two files are the heart of the change.

**`rewards.py`** — the `standup_progress` function should be deleted outright. It only exists to give gradient during the get-up motion, and that motion no longer exists. More importantly, the `_is_standing` gate that wraps `hold_still` and `variable_posture` needs to be removed or made always-true. Currently those rewards return zero if the robot isn't at standing height, which makes sense when the robot spends half an episode lying down. In stay-stand, the robot is always at standing height (or the episode terminates), so the gate is either always-true (harmless but confusing) or absent. Remove it. The `body_angular_velocity_penalty` and `angular_momentum_penalty` were disabled with `weight=0` specifically because "wild flailing during tumbling" was punished too harshly. In stay-stand there is no tumbling phase, so both can be re-enabled — but use a bounded kernel (`exp(-x²/σ²)` style) rather than the raw squared magnitude, to avoid the value-function divergence risk noted in `env_cfgs.py`. `hold_still` and `upright` become the primary dense reward signal; they should always fire, not be gated.

**`terminations.py`** — add a `fell_over` term back. The original velocity policy had `bad_orientation` which terminated when tilt exceeded a threshold; that's exactly what's needed here. A simple height-below-threshold OR uprightness-below-threshold check, something like `height < 0.5 OR uprightness < 0.5`, ends the episode the moment the robot can no longer be considered standing. Remove `stuck_no_progress` entirely — it was designed to cut off episodes where the robot stalled on the ground, which cannot happen now since any ground-contact terminates immediately via `fell_over`. Keep `time_out` and `catastrophic_state`.

---

## Phase 2 — `standup_env_cfg.py` and `curriculums.py`

**`standup_env_cfg.py`** — three targeted changes. First, `reset_base` switches from `reset_fallen_state` to a simple standing-pose reset (either keep `reset_fallen_state` with `orientation_mode="standing"` permanently, or switch to `reset_joints_by_offset` with small pose jitter around `HOME_KEYFRAME`). Second, remove the entire `fall_difficulty` entry from the curriculum dict — there are no fall stages to progress through. Third, remove `stuck_no_progress` from the terminations dict and add `fell_over` in its place. The `standup_progress` reward entry in the rewards dict should be deleted. The `push_robot` event can stay, but its curriculum escalation is gone — set a fixed moderate push range from the start (Stage 1 strength, e.g. `±0.3 m/s`), since the robot starts standing and push-recovery is a first-class skill from iteration 0.

**`curriculums.py`** — delete the `fall_difficulty` function entirely. Keep `terrain_levels_standup`, but its success signal already uses the same height+uprightness gate, which now means "held standing through the full episode" rather than "recovered from a fall." The docstring should update to reflect this, but the code itself is correct as-is. Remove the `min_step_counter` gate in `terrain_levels_standup` usage — the robot starts standing and can succeed at terrain advancement from iteration 0.

---

## Phase 3 — `env_cfgs.py`, `rl_cfg.py`, `__init__.py`

**`env_cfgs.py`** — this is param retuning rather than logic change. Reset `push_robot`'s `velocity_range` to a non-empty fixed range (remove its `fall_difficulty` escalation). Re-enable `body_ang_vel` and `angular_momentum` with non-zero weights (start small, e.g. `-0.05` and `-0.01`). Re-enable `action_rate_l2` with a small weight (e.g. `-0.01`) — the reason it was disabled was the exploding-std feedback loop, which you should also fix in `rl_cfg.py`. You can tighten `std_recovering` in `variable_posture` since there is no longer a recovery phase with large joint excursions — or collapse `std_recovering` and `std_standing` to the same value if you remove the gate entirely. Remove the `fall_difficulty` curriculum param wiring. Remove `stuck_no_progress` param overrides, add `fell_over` param wiring instead. Remove `terrain_levels`'s `min_step_counter` override.

**`rl_cfg.py`** — rename `experiment_name` to `"g1_staystand"`. Consider re-enabling `learn_std` (set back to `True`) now that the reward landscape is tighter and less noisy — the fixed-std workaround was a response to the chaotic recovery episodes, which no longer exist. If you re-enable it, watch `Policy/mean_std` closely in the first few hundred iterations. The `entropy_coef=0.0` can stay initially; the standing task needs less exploration than recovery did.

**`__init__.py`** — rename the task IDs from `Mjlab-Standup-*` to `Mjlab-StayStand-*`. No logic change.

---

## Files with zero edits

`standup_command.py`, `observations.py`, `terrain_utils.py`, and `runner.py` need no changes. `StandStillCommand` is already correct — it always outputs zero, which is exactly the right command for stay-stand. All observations (projected gravity, base height, joint state, IMU) are equally relevant for balancing as for getting up.

---

## Summary of what's deleted vs added

**Deleted:** `standup_progress` reward function and config entry, `fall_difficulty` curriculum function and config entry, `stuck_no_progress` termination function and config entry, `_is_standing` gates on `hold_still` and `variable_posture`.

**Added:** `fell_over` termination term (simple height/uprightness threshold check, very few lines).

**No new files.** The whole migration is surgical edits within the existing six files.

Ready to start coding whenever you are — Phase 1 (`rewards.py` + `terminations.py`) is the natural first step since everything else depends on getting the reward and termination semantics right.


Let's start with Phase 1. I'll do `rewards.py` first since it's the most fundamental change.Good. Now I have a complete picture of every line. Let me write the new `rewards.py`.**Change 1 — delete `standup_progress` (lines 58–94) and update the module docstring.****Change 2 — delete `_is_standing` helper (no longer referenced by any reward term).****Change 3 — delete `standup_progress` entirely.****Change 4 — rewrite `hold_still` to remove the `_is_standing` gate.****Change 5 — update `body_angular_velocity_penalty` to use a bounded exp kernel and rename it.****Change 6 — update `angular_momentum_penalty` to use a bounded exp kernel.****Change 7 — rewrite `variable_posture` to use a single std, removing the two-band logic.**Now verify the final file looks exactly right end-to-end:`rewards.py` is clean. Now `terminations.py`:Now add the `fell_over` function and remove `stuck_no_progress`:Now add the missing `quat_apply_inverse` import (needed by `fell_over`):Now remove `stuck_no_progress` entirely:Now clean up the unused `TerminationTermCfg` import (it was only used by `stuck_no_progress`):`TerminationTermCfg` is in the `TYPE_CHECKING` block, so it's import-only and causes no runtime error even if unused. Leave it — it's harmless and may be useful if someone adds a new stateful term later. Now do a final view of the whole file:Both Phase 1 files are complete and correct. Copy to outputs:Phase 1 is done. Here's a precise summary of every change made:

---

## `rewards.py` — what changed

**Deleted:**
- `_is_standing()` helper function — no longer referenced by anything
- `standup_progress()` function entirely — there is no get-up arc to shape

**Rewritten:**
- `hold_still` — removed `min_standing_height` and `min_standing_uprightness` params and the `_is_standing` gate. Now fires unconditionally every step. The signature shrank to just `(env, std, asset_cfg)`
- `body_angular_velocity_penalty` — added a `std` param, replaced `sum(square(ang_vel_xy))` with `exp(-xy_sq / std²)`. Bounded in [0,1]; safe to re-enable with a non-zero weight
- `angular_momentum_penalty` — added a `std` param, replaced raw `angmom_magnitude_sq` return with `exp(-angmom_magnitude_sq / std²)`. Same reason
- `variable_posture` — collapsed two-band logic (`std_recovering` / `std_standing`) into a single `std_values` tensor. Removed the `_is_standing` gate from `__call__`. Config must now pass `std_values` instead of two separate dicts
