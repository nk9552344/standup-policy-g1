# src/mjlab/tasks/standup/mdp/terminations.py

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_DEFAULT_SENSOR_CFG = SceneEntityCfg("contact_sensor")


def time_out(
    env: ManagerBasedRlEnv,
) -> torch.Tensor:
    """
    Episode timeout termination.

    MJLab already tracks episode length internally.
    This function simply returns the timeout buffer.

    Returns:
        Tensor[num_envs] of bools.
    """
    return env.termination_manager.time_outs


def head_contact(
    env: ManagerBasedRlEnv,
    threshold: float = 1.0,
    sensor_cfg: SceneEntityCfg = _DEFAULT_SENSOR_CFG,
) -> torch.Tensor:
    """
    Terminate when the head collides with terrain.

    HoST uses keyframe_head/head body contact as a hard failure.

    Args:
        threshold: Contact force threshold.
        sensor_cfg: Contact sensor configuration.

    Returns:
        Tensor[num_envs] bool.
    """
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]

    forces = contact_sensor.data.force
    if forces is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    head_body_ids = sensor_cfg.body_ids

    head_force = torch.norm(
        forces[:, head_body_ids, :],
        dim=-1,
    )

    return torch.any(head_force > threshold, dim=1)


def bad_orientation(
    env: ManagerBasedRlEnv,
    orientation_threshold: float = 0.2,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """
    Terminate when robot becomes heavily inverted.

    Uses projected gravity in base frame.

    Upright:
        gravity_b = [0,0,-1]

    Flipped:
        gravity_b = [0,0,+1]

    Args:
        orientation_threshold:
            Minimum acceptable gravity z projection.

    Returns:
        Tensor[num_envs] bool.
    """
    robot: Entity = env.scene[asset_cfg.name]

    gravity_b = robot.data.projected_gravity_b
    return gravity_b[:, 2] > -orientation_threshold


def torso_contact(
    env: ManagerBasedRlEnv,
    threshold: float = 5.0,
    sensor_cfg: SceneEntityCfg = _DEFAULT_SENSOR_CFG,
) -> torch.Tensor:
    """
    Optional termination when torso hits terrain.

    Not enabled by default in HoST but useful during debugging.

    Args:
        threshold: Contact force threshold.

    Returns:
        Tensor[num_envs] bool.
    """
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]

    forces = contact_sensor.data.force
    if forces is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    torso_force = torch.norm(
        forces[:, sensor_cfg.body_ids, :],
        dim=-1,
    )

    return torch.any(
        torso_force > threshold,
        dim=1,
    )


def fallen(
    env: ManagerBasedRlEnv,
    min_height: float = 0.15,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """
    Generic fall detector.

    Terminates when base height becomes too low.

    Useful as a safety fallback across
    ground / prone / slope / wall tasks.

    Args:
        min_height: Minimum allowed root height.

    Returns:
        Tensor[num_envs] bool.
    """
    robot: Entity = env.scene[asset_cfg.name]

    root_height = robot.data.root_link_pos_w[:, 2]
    return root_height < min_height