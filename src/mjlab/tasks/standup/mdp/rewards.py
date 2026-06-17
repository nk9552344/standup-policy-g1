# src/mjlab/tasks/standup/mdp/rewards.py
"""HoST reward terms — mjlab port.

Source: legged_gym/envs/g1/g1_config_ground*.py, g1_config_slope.py,
g1_config_wall.py, g1_config_platform.py (scales, groups, sigmas).

Groups (separate critic heads in HoST):
  task   (2.5 / 1.0 for prone)
  regu   (0.1)
  style  (1.0)
  target (1.0)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _projected_gravity_b(asset: Entity) -> torch.Tensor:
    gravity_w = asset.data.gravity_vec_w
    return quat_apply_inverse(asset.data.root_link_quat_w, gravity_w)


def _gaussian(x: torch.Tensor, sigma: float) -> torch.Tensor:
    return torch.exp(-x / sigma**2)


# ── GROUP 1: task ──────────────────────────────────────────────────────────

def task_orientation(
    env: ManagerBasedRlEnv,
    std: float = 1.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Upright reward. orientation_sigma=1 in all g1 configs."""
    asset: Entity = env.scene[asset_cfg.name]
    grav_b = _projected_gravity_b(asset)
    xy_sq = torch.sum(torch.square(grav_b[:, :2]), dim=1)
    return torch.exp(-xy_sq / std**2)


def task_head_height(
    env: ManagerBasedRlEnv,
    target_height: float = 1.0,
    margin: float = 1.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """target_head_height=1, target_head_margin=1. Point asset_cfg.body_ids at head/torso."""
    asset: Entity = env.scene[asset_cfg.name]
    if asset_cfg.body_ids:
        head_z = asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2].squeeze(1)
    else:
        head_z = asset.data.root_link_pos_w[:, 2]
    err_sq = torch.square(head_z - target_height)
    return _gaussian(err_sq, margin)


# ── GROUP 2: regu ───────────────────────────────────────────────────────────

def regu_dof_acc(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """regu_dof_acc = -2.5e-7."""
    asset: Entity = env.scene[asset_cfg.name]
    acc = asset.data.joint_acc[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(acc), dim=1)


def regu_action_rate(env: ManagerBasedRlEnv) -> torch.Tensor:
    """regu_action_rate = -0.01."""
    a, a1 = env.action_manager.action, env.action_manager.prev_action
    return torch.sum(torch.square(a - a1), dim=1)


class regu_smoothness:
    """Jerk penalty: ||(a_t-a_{t-1}) - (a_{t-1}-a_{t-2})||^2. regu_smoothness = -0.01.

    Stateful: buffers action from two steps back.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        del cfg
        n_actions = env.action_manager.action.shape[1]
        self._prev_prev_action = torch.zeros(env.num_envs, n_actions, device=env.device)

    def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        a, a1 = env.action_manager.action, env.action_manager.prev_action
        delta_t = a - a1
        delta_t1 = a1 - self._prev_prev_action
        self._prev_prev_action = a1.clone()
        return torch.sum(torch.square(delta_t - delta_t1), dim=1)

    def reset(self, env_ids: torch.Tensor) -> None:
        self._prev_prev_action[env_ids] = 0.0


def regu_torques(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """regu_torques = -2.5e-6. Uses actuator_force, indexed by actuator_ids."""
    asset: Entity = env.scene[asset_cfg.name]
    force = asset.data.actuator_force[:, asset_cfg.actuator_ids]
    return torch.sum(torch.square(force), dim=1)


def regu_joint_power(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """regu_joint_power = -2.5e-5."""
    asset: Entity = env.scene[asset_cfg.name]
    force = asset.data.actuator_force[:, asset_cfg.actuator_ids]
    vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(force * vel), dim=1)


def regu_dof_vel(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """regu_dof_vel = -1e-3."""
    asset: Entity = env.scene[asset_cfg.name]
    vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(vel), dim=1)


def regu_joint_tracking_error(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """regu_joint_tracking_error = -0.00025."""
    asset: Entity = env.scene[asset_cfg.name]
    pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    default = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(pos - default), dim=1)


def regu_dof_pos_limits(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """regu_dof_pos_limits = -100.0. Uses soft_joint_pos_limits (already includes the 0.9 soft ratio)."""
    asset: Entity = env.scene[asset_cfg.name]
    pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    limits = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, :]
    violation = torch.clamp(limits[..., 0] - pos, min=0.0) + torch.clamp(
        pos - limits[..., 1], min=0.0
    )
    return torch.sum(torch.square(violation), dim=1)


def regu_dof_vel_limits(
    env: ManagerBasedRlEnv,
    limit: float = 30.0,
    soft_ratio: float = 0.9,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """regu_dof_vel_limits = -1.0. mjlab has no per-joint joint_vel_limits field;
    pass an explicit `limit` (rad/s) per call site instead."""
    asset: Entity = env.scene[asset_cfg.name]
    vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    violation = torch.clamp(torch.abs(vel) - limit * soft_ratio, min=0.0)
    return torch.sum(torch.square(violation), dim=1)


# ── GROUP 3: style ──────────────────────────────────────────────────────────

def style_joint_deviation(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Generic |joint_pos|^2 penalty. Covers waist/hip_yaw/hip_roll deviation
    (-10 each) — register per joint group via asset_cfg."""
    asset: Entity = env.scene[asset_cfg.name]
    pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(pos), dim=1)


def style_joint_pose_deviation(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Generic deviation-from-default penalty. Covers shoulder_roll (-2.5),
    knee (-0.25 ground / -10 platform-slope-wall)."""
    asset: Entity = env.scene[asset_cfg.name]
    pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    default = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(pos - default), dim=1)


def style_joint_pose_match(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Generic exp(-err^2) reward. Covers shank_orientation (+10) and
    ground_parallel (+20)."""
    asset: Entity = env.scene[asset_cfg.name]
    pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    default = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    err_sq = torch.sum(torch.square(pos - default), dim=1)
    return torch.exp(-err_sq)


def style_ang_vel_xy(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """style_style_ang_vel_xy = 1 (ground) / 25 (prone)."""
    asset: Entity = env.scene[asset_cfg.name]
    ang_vel_xy = asset.data.root_link_ang_vel_b[:, :2]
    return torch.exp(-torch.sum(torch.square(ang_vel_xy), dim=1))


def style_feet_distance(
    env: ManagerBasedRlEnv,
    target_distance: float = 0.25,
    std: float = 2.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """style_feet_distance = -10. asset_cfg.body_ids = [left_foot, right_foot]."""
    asset: Entity = env.scene[asset_cfg.name]
    feet_xy = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :2]
    dist = torch.norm(feet_xy[:, 0, :] - feet_xy[:, 1, :], dim=-1)
    return _gaussian(torch.square(dist - target_distance), std)


def style_foot_displacement(
    env: ManagerBasedRlEnv,
    std: float = 2.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """style_left/right_foot_displacement = +2.5. asset_cfg.body_ids = [foot]."""
    asset: Entity = env.scene[asset_cfg.name]
    pelvis_xy = asset.data.root_link_pos_w[:, :2]
    foot_xy = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :2].squeeze(1)
    disp_sq = torch.sum(torch.square(foot_xy - pelvis_xy), dim=1)
    return _gaussian(disp_sq, std)


def style_feet_stumble(
    env: ManagerBasedRlEnv,
    threshold: float = 0.1,
    sensor_name: str = "feet_contact_sensor",
) -> torch.Tensor:
    """style_feet_stumble = -25 (platform/slope/wall)."""
    sensor: ContactSensor = env.scene[sensor_name]
    force = sensor.data.force
    assert force is not None
    horiz = torch.norm(force[..., :2], dim=-1)
    vert = torch.abs(force[..., 2])
    stumble = (horiz > threshold) & (horiz > vert)
    return stumble.float().sum(dim=1)


# ── GROUP 4: target ─────────────────────────────────────────────────────────

def target_ang_vel_xy(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """target_ang_vel_xy = 10."""
    asset: Entity = env.scene[asset_cfg.name]
    ang_vel_xy = asset.data.root_link_ang_vel_b[:, :2]
    return torch.exp(-torch.sum(torch.square(ang_vel_xy), dim=1))


def target_lin_vel_xy(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """target_lin_vel_xy = 10."""
    asset: Entity = env.scene[asset_cfg.name]
    lin_vel_xy = asset.data.root_link_lin_vel_b[:, :2]
    return torch.exp(-torch.sum(torch.square(lin_vel_xy), dim=1))


def target_feet_height_var(
    env: ManagerBasedRlEnv,
    std: float = 0.05,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """target_feet_height_var = 2.5. asset_cfg.body_ids = [left_foot, right_foot]."""
    asset: Entity = env.scene[asset_cfg.name]
    feet_z = asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2]
    return _gaussian(torch.square(feet_z[:, 0] - feet_z[:, 1]), std)


def target_upper_dof_pos(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """target_target_upper_dof_pos = 10."""
    asset: Entity = env.scene[asset_cfg.name]
    pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    default = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    err_sq = torch.sum(torch.square(pos - default), dim=1)
    return torch.exp(-err_sq)


def target_orientation(
    env: ManagerBasedRlEnv,
    std: float = 1.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """target_target_orientation = 10."""
    asset: Entity = env.scene[asset_cfg.name]
    grav_b = _projected_gravity_b(asset)
    xy_sq = torch.sum(torch.square(grav_b[:, :2]), dim=1)
    return torch.exp(-xy_sq / std**2)


def target_base_height(
    env: ManagerBasedRlEnv,
    target_height: float = 0.75,
    std: float = 0.25,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """target_target_base_height = 10. target_height=base_height_target=0.75."""
    asset: Entity = env.scene[asset_cfg.name]
    root_z = asset.data.root_link_pos_w[:, 2]
    return _gaussian(torch.square(root_z - target_height), std)