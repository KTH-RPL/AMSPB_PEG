import abc

import torch
import torch.nn as nn
import torch.distributions as D
import functools
from typing import Sequence, Union, Dict
from contextlib import contextmanager
from tensordict import TensorDict
import numpy as np
from torch.func import vmap
from omni_drones.utils.torch import euler_to_quaternion, quat_rotate


class Trajectory(nn.Module):
    """
    This class is used to define a trajectory for the drone to follow.
    """
    REGISTRY: Dict[str, "Trajectory"] = {}

    @abc.abstractmethod
    def __init_subclass__(cls, **kwargs):
        if cls.__name__ in Trajectory.REGISTRY:
            raise ValueError(f"{cls.__name__} already registered.")
        super().__init_subclass__(**kwargs)
        Trajectory.REGISTRY[cls.__name__] = cls
        Trajectory.REGISTRY[cls.__name__.lower()] = cls

    @abc.abstractmethod
    def reset(self, env_ids: torch.Tensor):
        """
        Abstract method to reset the trajectory.

        Args:
            env_ids (torch.Tensor): The environment IDs to reset.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abc.abstractmethod
    def compute_ref(self, *args, **kwargs) -> TensorDict:
        """
        Abstract method to compute the reference state of the drone at a given time.

        Args:

        Returns:
            TensorDict: The reference state of the drone at time t.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def estimate_bounds(self, dt:float, max_timesteps:float) -> Dict[str, torch.Tensor]:
        t_vector = torch.arange(0, (max_timesteps - 1)*dt, dt, device=self.device).unsqueeze(0)
        pos = self.compute_ref(t_vector, torch.arange(0., self.n_envs, device=self.device).int())
        return trajectory_stats(pos, dt)
    
    
    def show(self, device: str = "cpu"):
        import matplotlib.pyplot as plt

        time = torch.linspace(0, 20, 500, device=device)

        # Compute the reference state at each time step
        # using vmap for vectorized computation
        compute_ref_vmap = vmap(self.compute_ref)
        pos = compute_ref_vmap(time)
        pos = pos.squeeze(1).cpu().numpy()
        plt.figure()
        plt.plot(pos[...,0], pos[...,1], label=self.__class__.__name__)
        plt.axis('equal')
        plt.title(f"{self.__class__.__name__} Trajectory")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.grid()
        plt.show()


class Hover(Trajectory):
    """
    This class is used to define a hover trajectory for the drone.
    """
    def __init__(self, n_envs, device, arena_min, arena_max, difficulty):
        super().__init__()

        self.device = device
        self.n_envs = n_envs

        self.arena_min = arena_min + 0.5
        self.arena_max = arena_max - 0.5

        self.pos_dist = D.Uniform(self.arena_min, self.arena_max)

        self.pos = self.pos_dist.sample(torch.Size([n_envs], device=self.device))

    def reset(self, env_ids: torch.Tensor):
        self.pos[env_ids] = self.pos_dist.sample(env_ids.shape)

    def compute_ref(self, t: Union[float, torch.Tensor], env_ids) -> TensorDict:
        pos = self.pos[env_ids].unsqueeze(1).expand(-1, t.shape[1], -1)
        return pos


class Circular(Trajectory):

    def __init__(self, n_envs, device, arena_min, arena_max, difficulty):
        super().__init__()

        self.device = device
        self.n_envs = n_envs

        R_MAX = torch.min(arena_max[:2]) - 0.5
        R_MIN = R_MAX/1.25
        
        V_TARGET = 8

        OMEGA_MAX = V_TARGET / R_MAX
        OMEGA_MIN = OMEGA_MAX/1.25

        Z_OFFSET_MIN = arena_min[2] + 1.0 + difficulty/2.0
        Z_OFFSET_MAX = arena_min[2] + 1.0 + 1.5*difficulty

        self.radius_dist = D.Uniform(R_MIN, R_MAX)
        self.omega_dist = D.Uniform(
            torch.tensor(OMEGA_MIN, device=self.device),
            torch.tensor(OMEGA_MAX, device=self.device)
        )
        self.phase_dist = D.Uniform(
            torch.tensor(0., device=self.device),
            torch.tensor(2 * torch.pi, device=self.device)
        )
        self.traj_scale_dist = D.Uniform(
            torch.tensor([0.85, 0.85, 0.7], device=self.device),
            torch.tensor([1.0, 1.0, 1.0], device=self.device)
        )
        self.traj_rpy_dist = D.Uniform(
            torch.tensor([0., 0., 0.], device=self.device) * torch.pi,
            torch.tensor([0.05, 0.05, 2.], device=self.device) * torch.pi
        )

        self.Z_OFFSET = D.Uniform(
            torch.tensor(Z_OFFSET_MIN, device=self.device),
            torch.tensor(Z_OFFSET_MAX, device=self.device)
        )

        self.radius = self.radius_dist.sample(torch.Size([n_envs], device=self.device)).unsqueeze(1)
        self.omega = self.omega_dist.sample(torch.Size([n_envs], device=self.device)).unsqueeze(1)
        self.phase = self.phase_dist.sample(torch.Size([n_envs], device=self.device)).unsqueeze(1)
        self.traj_scale = self.traj_scale_dist.sample(torch.Size([n_envs], device=self.device)).unsqueeze(1)
        self.traj_rot = euler_to_quaternion(self.traj_rpy_dist.sample(torch.Size([n_envs], device=self.device))).unsqueeze(1)
        self.z_offset = self.Z_OFFSET.sample(torch.Size([n_envs], device=self.device)).unsqueeze(1)


    def reset(self, env_ids: torch.Tensor):
        self.radius[env_ids] = self.radius_dist.sample(env_ids.shape).unsqueeze(1)
        self.omega[env_ids] = self.omega_dist.sample(env_ids.shape).unsqueeze(1)
        self.phase[env_ids] = self.phase_dist.sample(env_ids.shape).unsqueeze(1)
        self.traj_scale[env_ids] = self.traj_scale_dist.sample(env_ids.shape).unsqueeze(1)
        self.traj_rot[env_ids] = euler_to_quaternion(self.traj_rpy_dist.sample(env_ids.shape)).unsqueeze(1)
        self.z_offset[env_ids] = self.Z_OFFSET.sample(env_ids.shape).unsqueeze(1)
    
    def compute_ref(self, t: Union[float, torch.Tensor], env_ids) -> TensorDict:
        _t = self.phase[env_ids] + scale_time(t * self.omega[env_ids])
        
        pos = torch.stack([
            torch.cos(_t) * self.radius[env_ids],
            torch.sin(_t) * self.radius[env_ids],
            torch.ones_like(_t, device=self.device)*self.z_offset[env_ids]
        ], dim=-1)
        pos = quat_rotate(self.traj_rot[env_ids].expand(-1,_t.shape[1],-1), pos) * self.traj_scale[env_ids]
        return pos
    
class Lemniscate(Trajectory):

    def __init__(self, n_envs, device, arena_min, arena_max, difficulty):
        super().__init__()

        self.device = device
        self.n_envs = n_envs

        R_MAX = torch.min(arena_max[:2]) - 0.5
        R_MIN = R_MAX/1.5

        OMEGA_MIN = 0.8
        OMEGA_MAX = 1.1 * (1 + (1 - difficulty)/2.0)

        self.Z_OFFSET = 0.5 + 1 + difficulty

        self.radius_dist = D.Uniform(R_MIN, R_MAX)
        self.omega_dist = D.Uniform(
            torch.tensor(OMEGA_MIN, device=self.device),
            torch.tensor(OMEGA_MAX, device=self.device)
        )
        self.phase_dist = D.Uniform(
            torch.tensor(0., device=self.device),
            torch.tensor(2 * torch.pi, device=self.device)
        )
        self.traj_scale_dist = D.Uniform(
            torch.tensor([0.5, 0.5, 0.7], device=self.device),
            torch.tensor([1.0, 1.0, 1.0], device=self.device)
        )
        self.traj_rpy_dist = D.Uniform(
            torch.tensor([0., 0., 0.], device=self.device) * torch.pi,
            torch.tensor([0.05, 0.05, 2.], device=self.device) * torch.pi
        )

        self.radius = self.radius_dist.sample(torch.Size([n_envs], device=self.device)).unsqueeze(1)
        self.omega = self.omega_dist.sample(torch.Size([n_envs], device=self.device)).unsqueeze(1)
        self.phase = self.phase_dist.sample(torch.Size([n_envs], device=self.device)).unsqueeze(1)
        self.traj_scale = self.traj_scale_dist.sample(torch.Size([n_envs], device=self.device)).unsqueeze(1)
        self.traj_rot = euler_to_quaternion(self.traj_rpy_dist.sample(torch.Size([n_envs], device=self.device))).unsqueeze(1)
        
    def reset(self, env_ids: torch.Tensor):
        self.radius[env_ids] = self.radius_dist.sample(env_ids.shape).unsqueeze(1)
        self.omega[env_ids] = self.omega_dist.sample(env_ids.shape).unsqueeze(1)
        self.phase[env_ids] = self.phase_dist.sample(env_ids.shape).unsqueeze(1)
        self.traj_scale[env_ids] = self.traj_scale_dist.sample(env_ids.shape).unsqueeze(1)
        self.traj_rot[env_ids] = euler_to_quaternion(self.traj_rpy_dist.sample(env_ids.shape)).unsqueeze(1)

    def compute_ref(self, t: Union[float, torch.Tensor], env_ids) -> TensorDict:
        _t = self.phase[env_ids] + scale_time(t * self.omega[env_ids])

        sin_t = torch.sin(_t)
        cos_t = torch.cos(_t)
        sin2p1 = torch.square(sin_t) + 1

        pos = torch.stack([
            (self.radius[env_ids] * cos_t / sin2p1),
            (self.radius[env_ids] * sin_t * cos_t / sin2p1),
            torch.ones_like(_t, device=self.device)*self.Z_OFFSET
        ], dim=-1)
        pos = quat_rotate(self.traj_rot[env_ids].expand(-1,_t.shape[1],-1), pos) * self.traj_scale[env_ids]
        return pos
    

def trajectory_stats(trajectory:torch.Tensor, dt:float):
    
    vel = torch.gradient(trajectory, spacing=dt, dim=1)[0]
    acc = torch.gradient(vel, spacing=dt, dim=1)[0]
    jerk = torch.gradient(acc, spacing=dt, dim=1)[0]

    # curvature: https://en.wikipedia.org/wiki/Curvature
    cross_va = torch.cross(vel, acc)
    curvature = torch.norm(cross_va, dim=-1) / (torch.norm(vel, dim=-1)**3 + 1e-6)
    
    #torsion: https://en.wikipedia.org/wiki/Torsion_of_a_curve
    torsion = torch.norm(cross_va*jerk, dim=-1) / (torch.norm(cross_va, dim=-1)**2 + 1e-6)

    trajectory = trajectory.reshape((trajectory.shape[0]*trajectory.shape[1], 3))
    vel = vel.reshape((vel.shape[0]*vel.shape[1], 3))
    acc = acc.reshape((acc.shape[0]*acc.shape[1], 3))
    jerk = jerk.reshape((jerk.shape[0]*jerk.shape[1], 3))

    max_p, min_p, mean_p = trajectory.max(dim=0)[0], trajectory.min(dim=0)[0], trajectory.mean(dim=0)
    max_v, min_v, mean_v = vel.max(dim=0)[0], vel.min(dim=0)[0], vel.mean(dim=0)
    max_a, min_a, mean_a = acc.max(dim=0)[0], acc.min(dim=0)[0], acc.mean(dim=0)
    max_j, min_j, mean_j = jerk.max(dim=0)[0], jerk.min(dim=0)[0], jerk.mean(dim=0)

    stats = {
        "position": {
            "max": max_p,
            "min": min_p,
            "mean": mean_p,
        },
        
        "velocity": {
            "max": max_v,
            "min": min_v,
            "mean": mean_v,
        },
        "acceleration": {
            "max": max_a,
            "min": min_a,
            "mean": mean_a,
        },
        "jerk": {
            "max": max_j,
            "min": min_j,
            "mean": mean_j,
        },
        "curvature": {
            "max": curvature.max(dim=-1)[0],
            "min": curvature.min(dim=-1)[0],
            "mean": curvature.mean(dim=-1),
        },
        "torsion": {
            "max": torsion.max(dim=-1)[0],
            "min": torsion.min(dim=-1)[0],
            "mean": torsion.mean(dim=-1),
        },
    }
    return stats


def check_feasibility(pos, vel, acc, drone_params):
    g = torch.tensor([0, 0, -9.81], device=pos.device)
    m = drone_params['mass']
    max_thrust = drone_params['max_thrust']
    max_angle = drone_params['max_angle']  # in radians

    f = m * (acc + g)  # Required force
    thrust_mag = torch.norm(f, dim=-1)
    z_b = f / (thrust_mag.unsqueeze(-1) + 1e-6)

    # compute tilt angle
    tilt = torch.acos(z_b[..., 2].clamp(-1, 1))  # angle from z-axis

    feasible_thrust = (thrust_mag < max_thrust)
    feasible_tilt = (tilt < max_angle)

    feasible = feasible_thrust & feasible_tilt
    return feasible, thrust_mag, tilt



def scale_time(t, a: float=1.0):
    return t / (1 + 1/(a*torch.abs(t)))