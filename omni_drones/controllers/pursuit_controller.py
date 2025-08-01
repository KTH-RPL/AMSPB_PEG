# MIT License
#
# Copyright (c) 2025 Your Name
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all    
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
from tensordict import TensorDict
from .controller import ControllerBase  
from .lee_position_controller import LeePositionController  

from omni_drones.robots.drone.utils import split_state, get_velocity_from_state, get_position_from_state

import yaml
import os.path as osp

class PID_Pursuit(ControllerBase):
    def __init__(
        self,
        g: float,
        uav_params: dict,
        controller_params: dict = None,
        env_params: dict = None,
    ) -> None:
        super().__init__()

        self.lee = LeePositionController(g, uav_params)

        if controller_params is None or controller_params == "None":
            controller_params_path = osp.join(
                osp.dirname(__file__), "cfg", f"pid_pursuit_{uav_params['name']}.yaml"
            )
            with open(controller_params_path, "r") as f:
                controller_params = yaml.safe_load(f)
        self.controller_params = controller_params
        self.dt = env_params["dt"]

        self.kp = torch.tensor(controller_params["kp"])
        self.ki = torch.tensor(controller_params["ki"])
        self.kd = torch.tensor(controller_params["kd"])
        self.filter_alpha = torch.tensor(controller_params["filter_alpha"])
        self.integral_limit = torch.tensor(controller_params["integral_limit"])
        self.derivative_limit = torch.tensor(controller_params["derivative_limit"])

        self.max_speed = torch.tensor(controller_params["max_speed"])
        self.max_acceleration = torch.tensor(controller_params["max_acceleration"])
        
        min_frames_max_speed = 1000000
        self.total_frames = env_params["total_frames"] - min_frames_max_speed if env_params["total_frames"] > min_frames_max_speed else env_params["total_frames"]
        self.start_speed = self.max_speed # just initialization

    def do_curriculum_speed(self, bool: bool):
        self.curriculum_pursuer_speed = bool
        if self.curriculum_pursuer_speed:
            self.start_speed = self.max_speed * 0.1
        else:
            self.start_speed = self.max_speed

    def allocate_arrays(self, n_envs:int, device: torch.device):
        self.e_p = torch.zeros(n_envs, 1, 3, device=device)
        self.e_i = torch.zeros(n_envs, 1, 3, device=device)
        self.e_v = torch.zeros(n_envs, 1, 3, device=device)

        self.speed_limit = torch.zeros(n_envs, 1, 3, device=device)
        
    def reset(self, env_ids: torch.Tensor, frame: int = 0):
        self.e_p[env_ids] = torch.zeros_like(self.e_p[env_ids])
        self.e_i[env_ids] = torch.zeros_like(self.e_i[env_ids])
        self.e_v[env_ids] = torch.zeros_like(self.e_v[env_ids])

        current_speed = self.start_speed + (self.max_speed - self.start_speed) * (frame / self.total_frames)
        current_speed = torch.min(current_speed, self.max_speed).unsqueeze(-1).unsqueeze(-1).expand(env_ids.shape[0], -1, 3)
        self.speed_limit[env_ids] = current_speed.to(self.speed_limit.device)

    def to(self, device: torch.device):
        self.lee = self.lee.to(device)
        self.start_speed = self.start_speed.to(device)

        for attr in self.controller_params:
            setattr(self, attr, getattr(self, attr).to(device))

        return self
 
    def forward(
        self,
        pursuer_state: torch.Tensor,  
        evader_state: torch.Tensor, 
    ) -> torch.Tensor:
        
        u_cmd, yaw_cmd = self._pid(pursuer_state, evader_state)
        # Since RL is not considering yaw, we should ignore it here as well
        return self.lee(
            root_state=pursuer_state,
            target_pos=None,
            target_vel=u_cmd,
            target_acc=None,
            target_yaw=None,
        )

    def _pid(self, p_state, e_state):
        pos_p = get_position_from_state(p_state)
        pos_e = get_position_from_state(e_state)
        prev_e_p = self.e_p
        self.e_p = pos_e - pos_p

        self.e_i += self.e_p * self.dt
        self.e_i = torch.clamp(self.e_i, -self.integral_limit, self.integral_limit)

        derivative = (self.e_p - prev_e_p) / self.dt
        derivative = torch.clamp(derivative, -self.derivative_limit, self.derivative_limit)
        self.e_v = self.e_v.lerp(derivative, self.filter_alpha)

        u = self.kp*self.e_p + self.ki*self.e_i + self.kd*self.e_v
        u = torch.clamp(u, -self.speed_limit, self.speed_limit)

        yaw = torch.atan2(self.e_p[..., 1], self.e_p[..., 0]).unsqueeze(-1)
        return u, yaw
    
    
class FRPN_Pursuit(ControllerBase):
    """
    Fast–Response Proportional Navigation (FRPN) pursuit controller.
    "Towards Safe Mid-Air Drone Interception: Strategies for Tracking & Capture"
    """
    def __init__(
        self,
        g: float,
        uav_params: dict,
        controller_params: dict = None,
        env_params: dict = None,
    ) -> None:
        super().__init__()

        self.lee = LeePositionController(g, uav_params)

        if controller_params is None or controller_params == "None":
            cfg_path = osp.join(
                osp.dirname(__file__),
                "cfg",
                f"frpn_pursuit_{uav_params['name']}.yaml",
            )
            with open(cfg_path, "r") as f:
                controller_params = yaml.safe_load(f)

        self.controller_params = controller_params

        self.G = torch.tensor(controller_params["G"])
        self.W = torch.tensor(controller_params["W"])
        self.max_speed = torch.tensor([controller_params["max_speed"]])

        self.dt = env_params["dt"]
        min_frames_max_speed = 1000000
        self.total_frames = env_params["total_frames"] - min_frames_max_speed if env_params["total_frames"] > min_frames_max_speed else env_params["total_frames"]
        self.start_speed = self.max_speed # just initialization

    def do_curriculum_speed(self, bool: bool):
        self.curriculum_pursuer_speed = bool
        if self.curriculum_pursuer_speed:
            self.start_speed = self.max_speed * 0.1
        else:
            self.start_speed = self.max_speed

    def reset(self, env_ids: torch.Tensor, frame: int = 0):
        current_speed = self.start_speed + (self.max_speed - self.start_speed) * (frame / self.total_frames)
        current_speed = torch.min(current_speed, self.max_speed).unsqueeze(-1).unsqueeze(-1).expand(env_ids.shape[0], -1, 3)
        self.speed_limit[env_ids] = current_speed.to(self.speed_limit.device)
    
    def allocate_arrays(self, n_envs:int, device: torch.device):
        self.speed_limit = torch.zeros(n_envs, 1, 3, device=device)

    def to(self, device: torch.device):
        self.lee = self.lee.to(device)
        self.start_speed = self.start_speed.to(device)

        for attr in self.controller_params:
            setattr(self, attr, getattr(self, attr).to(device))

        return self

    def forward(
        self,
        pursuer_state: torch.Tensor, 
        evader_state: torch.Tensor,   
    ) -> torch.Tensor:
        
        pos_p = get_position_from_state(pursuer_state)
        vel_p, _ = get_velocity_from_state(pursuer_state)
        pos_e = get_position_from_state(evader_state)
        vel_e, _ = get_velocity_from_state(evader_state)

        dp = pos_e - pos_p              
        dv = vel_e - vel_p           

        dp_norm = dp.norm(dim=-1, keepdim=True)   
        dv_norm = dv.norm(dim=-1, keepdim=True)   

        eps = 1e-6
        t_go = dp_norm / (dv_norm + eps)         

        inv_t2 = 1.0 / (t_go * t_go + eps)       
        term_PN = (dp + dv * t_go) * inv_t2      
        term_P  = dp                              

        u_cmd = self.G*(self.W*term_P + (1.0 - self.W)*term_PN)      
        u_cmd = torch.clamp(u_cmd, -self.speed_limit, self.speed_limit)

        yaw = torch.atan2(dp[..., 1], dp[..., 0]).unsqueeze(-1)  

        return self.lee(
            root_state=pursuer_state,
            target_pos=None,
            target_vel=u_cmd,
            target_acc=None,
            target_yaw=yaw,
        )
    