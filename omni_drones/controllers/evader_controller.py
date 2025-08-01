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
from .controller import ControllerBase
from .lee_position_controller import LeePositionController
import yaml
import os.path as osp

from omni_drones.robots.drone.utils import split_state, get_velocity_from_state, get_position_from_state

class Repel_Evade(ControllerBase):
    r"""
    An evader that generates repulsive forces from the pursuer and arena boundaries
    to avoid capture while staying within a rectangular arena.
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
                osp.dirname(__file__), 'cfg', f'repel_evade_{uav_params["name"]}.yaml'
            )
            with open(cfg_path, 'r') as f:
                controller_params = yaml.safe_load(f)

        self.dt = env_params['dt']
        self.controller_params = controller_params

        self.k_pursuer = torch.tensor(controller_params['k_pursuer'])
        self.k_wall = torch.tensor(controller_params['k_wall'])
        self.order_den_pursuer = torch.tensor(controller_params['order_den_pursuer'])
        self.order_den_wall = torch.tensor(controller_params['order_den_wall'])
        self.min_wall_dist = torch.tensor(controller_params['min_wall_dist'])
        self.min_pursuer_dist = torch.tensor(controller_params['min_pursuer_dist'])
        self.max_speed = torch.tensor(controller_params['max_speed'])

        self.arena_min = torch.tensor(env_params['arena_min']) 
        self.arena_max =torch.tensor(env_params['arena_max'])

        # prepare wall normals and offsets for vectorized computation
        normals = torch.tensor([
            [1.0,  0.0,  0.0],  
            [-1.0, 0.0,  0.0], 
            [0.0,  1.0,  0.0],  
            [0.0, -1.0,  0.0], 
            [0.0,  0.0,  1.0], 
            [0.0,  0.0, -1.0], 
        ])
        normal_ground = torch.tensor([0.0, 0.0, 1.0])  

        offsets = torch.tensor([
            self.arena_min[0],      
            -self.arena_max[0],    
            self.arena_min[1],    
            -self.arena_max[1],     
            self.arena_min[2],      
            -self.arena_max[2],     
        ])
        offset_ground = torch.tensor(self.arena_min[2])  

        self.wall_normals = normals
        self.wall_offsets = offsets

        self.normal_ground = normal_ground
        self.offset_ground = offset_ground


    def reset(self, env_ids: torch.Tensor):
        # nothing to reset for this stateless evader
        return

    def to(self, device: torch.device):
        # move all parameters and lee controller to device
        self.lee = self.lee.to(device)

        for attr in self.controller_params:
            setattr(self, attr, getattr(self, attr).to(device))

        self.arena_min = self.arena_min.to(device)
        self.arena_max = self.arena_max.to(device)

        self.wall_normals = self.wall_normals.to(device)
        self.wall_offsets = self.wall_offsets.to(device)

        self.normal_ground = self.normal_ground.to(device)
        self.offset_ground = self.offset_ground.to(device)

        return self

    def forward(
        self,
        pursuer_state: torch.Tensor,  # (...,13)
        evader_state: torch.Tensor,   # (...,13)
    ) -> torch.Tensor:
        pos_p = get_position_from_state(pursuer_state)
        pos_e = get_position_from_state(evader_state)

        e_p = pos_e - pos_p
        dist_pe = e_p.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        
        mask_near = dist_pe <= self.min_pursuer_dist
        f_pursuer = torch.zeros_like(e_p)
        f_pursuer = torch.where(
            mask_near,
            self.k_pursuer*e_p/(dist_pe**self.order_den_pursuer),
            f_pursuer
        )
        dist_w = (pos_e * self.wall_normals).sum(dim=-1) - self.wall_offsets
        dist_w = dist_w.clamp(min=1e-6)

        mask_walls = dist_w < self.min_wall_dist
        f_wall = (self.k_wall*self.wall_normals*(1.0 / (dist_w**self.order_den_wall)).unsqueeze(-1))
        f_wall = f_wall[mask_walls,:].reshape(pos_e.shape[0],-1,3)
        f_wall = f_wall.sum(dim=-2, keepdim=True)

        F_total = f_pursuer + f_wall

        u_cmd = F_total / F_total.norm(dim=-1, keepdim=True).clamp(min=1e-6) * self.max_speed
        yaw = torch.atan2(u_cmd[..., 1], u_cmd[..., 0]).unsqueeze(-1)

        return self.lee(
            root_state=evader_state,
            target_pos=None,
            target_vel=u_cmd,
            target_acc=None,
            target_yaw=None,
        )
