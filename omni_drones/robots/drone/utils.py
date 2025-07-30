import torch

def split_state(state: torch.Tensor):
    pos = state[..., :3]
    rot = state[..., 3:7]
    lin_vel = state[..., 7:10]
    ang_vel = state[..., 10:13]
    heading = state[..., 13:16]
    up = state[..., 16:19]
    throttle = state[..., 19:]
    return pos, rot, lin_vel, ang_vel, heading, up, throttle


def get_position_from_state(state: torch.Tensor):
    pos, _, _, _, _, _, _ = split_state(state)
    return pos

def get_velocity_from_state(state: torch.Tensor):
    _, _, lin_vel, ang_vel, _, _, _ = split_state(state)
    return lin_vel, ang_vel

