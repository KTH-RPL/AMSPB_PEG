# MIT License
#
# Copyright (c) 2023 Botian Xu, Tsinghua University
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
import torch.distributions as D


import omni_drones.utils.kit as kit_utils
import omni.isaac.core.utils.prims as prim_utils

from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.robots.drone import MultirotorBase
from omni_drones.views import ArticulationView, RigidPrimView
from omni_drones.utils.torch import euler_to_quaternion, quat_axis, quaternion_to_rotation_matrix, quat_rotate_inverse
from omni_drones.utils.sampling import rectangular_ring_sampling, min_separation_sampling, policy_sampling

from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import (UnboundedContinuousTensorSpec, 
                          CompositeSpec, 
                          DiscreteTensorSpec, 
                          BinaryDiscreteTensorSpec)

from omni.isaac.debug_draw import _debug_draw

from omni_drones.utils.trajectory import Trajectory

from omni_drones.robots.drone.utils import split_state, get_velocity_from_state

import numpy as np

class Pursuit(IsaacEnv):
    def __init__(self, cfg, headless):
        cfg = self.parse_cfg(cfg)
        super().__init__(cfg, headless)

        self.pursuer.initialize()
        self.evader.initialize()
        if self.cfg.task.do_randomization:
            self.pursuer.setup_randomization(self.randomization["drone"])
            self.evader.setup_randomization(self.randomization["drone"])

        self.init_rpy_dist = D.Uniform(
            torch.tensor([-.2, -.2, 0.], device=self.device) * torch.pi,
            torch.tensor([0.2, 0.2, 2.], device=self.device) * torch.pi
        )

        self.arena_min = torch.tensor(self.cfg.task.env.arena_min, device=self.device)
        self.arena_max = torch.tensor(self.cfg.task.env.arena_max, device=self.device)
        self.arena_center = (self.arena_max + self.arena_min)/2.0
        self.arena_half = (self.arena_max - self.arena_min)/2.0 

        self.K_P = self.arena_half
        self.K_V = torch.tensor([15, 15, 5], device=self.device)
        self.K_OMEGA = torch.tensor([15, 15, 5], device=self.device)
        self.K_RP = self.K_P*2
        self.K_D = self.K_RP.norm(dim=-1, keepdim=True)
        self.K_EFFORT = 4.0  # max effort

        self.alpha = 0.8

        self.draw = _debug_draw.acquire_debug_draw_interface()

        if self.cfg.task.env.render_grid_arenas:
            self.render_grids_arenas(torch.arange(self.num_envs, device=self.device))

        if self.prev_traj_steps > 0:
            self.prev_pos = torch.zeros(
                self.num_envs, self.prev_traj_steps, 3,
                device=self.device
            )

        # hot fix: always log a minimum of 4 previous steps
        self.prev_evader_steps_fix = torch.max(torch.tensor([self.prev_evader_traj_steps, 4]))
        self.prev_error_pos = torch.zeros(
            self.num_envs, self.prev_evader_steps_fix, 3,
            device=self.device
        )

        self.pos_evaders = torch.zeros(self.num_envs, 3, device=self.device)
        self.vel_evaders = torch.zeros(self.num_envs, 6, device=self.device)
        self.evader_actions = torch.zeros(self.num_envs, 1, 4, device=self.device)

        self.prev_action = torch.zeros(
            self.num_envs, 1, self.pursuer.action_spec.shape[-1],
            device=self.device
        )

        self.action_difference = torch.zeros(
            self.num_envs, 1, self.pursuer.action_spec.shape[-1],
            device=self.device
        )

        if self.cfg.task.include_effort:
            self.effort = torch.zeros(
                self.num_envs, 1,
                device=self.device
            )

            self.effort_evader = torch.zeros(
                self.num_envs, 1,
                device=self.device
            )

        self._sample_pursuer_controllers()
        self._sample_evader_controllers()

        if self.do_interception:
            self.W_INT = torch.ones(self.num_envs, 1, device=self.device)*1.0
        else:
            self.W_INT = torch.ones(self.num_envs, 1, device=self.device)*(-1.0)


    def parse_cfg(self, cfg):
        pursuer_policy = cfg.task.pursuer_model.policy.lower()
        if "fixed_path" in pursuer_policy:
            cfg.task.pursuer_model.controller = ["LeePositionController"]
            cfg.task.action_transform = None
        elif "pid" in pursuer_policy:
            cfg.task.pursuer_model.controller = ["PID_Pursuit"]
            cfg.task.action_transform = None
        elif "frpn" in pursuer_policy:
            cfg.task.pursuer_model.controller = ["FRPN_Pursuit"]
            cfg.task.action_transform = None
        elif "velocity" in pursuer_policy:
            cfg.task.pursuer_model.controller = ["LeePositionController"]
            cfg.task.action_transform = "velocity"
        elif "rate" in pursuer_policy:
            cfg.task.pursuer_model.controller = ["RateController"]
            cfg.task.action_transform = "rate"
        elif "thrust" in pursuer_policy:
            cfg.task.pursuer_model.controller = [None]
            cfg.task.action_transform = None
        else:
            raise NotImplementedError(f"Policy {pursuer_policy} not implemented")

        evader_controllers = []
        for evader_policy in cfg.task.evader_model.policy:
            if "fixed_path" in evader_policy:
                evader_controllers.append("LeePositionController")
            elif "repel" in evader_policy:
                evader_controllers.append("Repel_Evade")
            elif "thrust" in evader_policy:
                evader_controllers.append(None)
            elif "velocity" in evader_policy:
                evader_controllers.append("LeePositionController")
            elif "rate" in evader_policy:
                evader_controllers.append("RateController")
            else:
                Warning(f"Policy {evader_policy} not implemented, using None")
                evader_controllers.append(None)
        cfg.task.evader_model.controller = evader_controllers
        
        if cfg.task.use_timestep_reward:
            cfg.task.reward_approach_weight = 0.0
        else:
            cfg.task.reward_timestep = 0.0

        self.prev_traj_steps = int(cfg.task.prev_traj_steps)
        self.prev_evader_traj_steps = int(cfg.task.prev_evader_traj_steps)

        self.arena_size = cfg.task.env.arena_size
        self.min_initial_separation = 1.5*(1.0 + self.arena_size)
        self.final_separation = cfg.task.env.final_separation

        xy_half_size = 2.0*(1.0 + self.arena_size)
        z_size = 2.0*(1.0 + self.arena_size)

        z_min = cfg.task.env.z_min # below z_min, pursuer gets a negative reward
        self.z_stop = cfg.task.env.z_stop # below z_stop, episode is terminated

        cfg.task.env.arena_min = [-xy_half_size, -xy_half_size, z_min]
        cfg.task.env.arena_max = [xy_half_size, xy_half_size, z_size + z_min]

        if cfg.task.env.leave_spacing_between_envs:
            cfg.task.env.env_spacing = int(xy_half_size*2)
        else:
            cfg.task.env.env_spacing = 0.0

        if cfg.viewer.auto:
            cfg.viewer.eye = (np.array(cfg.task.env.arena_max)*[2,2,2]).tolist()
        else:
            cfg.viewer.eye = (np.array(cfg.task.env.arena_max)*[1.5,1.5,1.5]).tolist()
            # cfg.viewer.lookat = (np.array(cfg.task.env.arena_max)*[0.0,0.0,0.5]).tolist()

        self.time_encoding = cfg.task.time_encoding
        self.randomization = cfg.task.get("randomization", {})

        self.reward_timestep = cfg.task.reward_timestep
        self.reward_approach_weight = cfg.task.reward_approach_weight
        self.reward_evader_caught = cfg.task.reward_evader_caught
        self.reward_body_rates_weight = cfg.task.reward_body_rates_weight
        self.reward_out_of_bounds = cfg.task.reward_out_of_bounds
        self.reward_action_smoothness = cfg.task.reward_action_smoothness
        self.reward_heading_weight = cfg.task.reward_heading_weight
        self.reward_effort_weight = cfg.task.reward_effort_weight

        self.do_interception = cfg.task.do_interception

        return cfg

    def _design_scene(self):
        pursuer_model_cfg = self.cfg.task.pursuer_model
        env_cfg = self.cfg.task.env
        env_cfg.dt = self.dt
        env_cfg.total_frames = self.cfg.total_frames

        MultirotorBase.reset_registry()

        self.pursuer, self.cont_pursuer = MultirotorBase.make(
            pursuer_model_cfg.name, 
            pursuer_model_cfg.controller, 
            env_params=env_cfg,
            controller_params=self.cfg.task.pursuer_model.controller_params,
            device=self.device,
            drone_id="pursuer",
            drone_color="blue"
        )
        self.cont_pursuer = self.cont_pursuer[0]

        evader_model_cfg = self.cfg.task.evader_model
        self.evader, self.cont_evader = MultirotorBase.make(
            evader_model_cfg.name, 
            evader_model_cfg.controller, 
            env_params=env_cfg,
            controller_params=self.cfg.task.evader_model.controller_params,
            device=self.device,
            drone_id="evader",
            drone_color="red"
        )

        kit_utils.create_ground_plane(
            "/World/defaultGroundPlane",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
            color=(0.5,0.5,0.5),
            collision=False
            )

        # kit_utils.load_template_scene()

        self.pursuer.spawn(translations=[(1.0, 1.0, 2.0)], scale=self.cfg.task.env.scale_drones)
        self.evader.spawn(translations=[(0.0, 0.0, 1.5)], scale=self.cfg.task.env.scale_drones)

        return ["/World/defaultGroundPlane"]

    def _set_specs(self):
        pursuer_state_dim = self.pursuer.state_spec.shape[-1]
        observation_dim = 0

        if self.cfg.task.time_encoding:
            self.time_encoding_dim = 1
            observation_dim += self.time_encoding_dim

        # relative position to evader
        dim_pos = 3
        if self.cfg.task.include_distances:
            dim_pos += 1

        if self.prev_evader_traj_steps > 0:
            observation_dim += dim_pos * self.prev_evader_traj_steps
        observation_dim += dim_pos  

        if self.cfg.task.include_closing_velocity:
            observation_dim += 3

        # heading to target
        if self.cfg.task.include_heading_to_target:
            observation_dim += 3
    
        # z coordinate
        if self.prev_traj_steps > 0:
            observation_dim += self.prev_traj_steps*1
        observation_dim += 1

        # orientation, velocity, angular velocity
        observation_dim += 9 + 3 + 3

        # previous action
        if self.cfg.task.include_last_action:
            observation_dim += self.pursuer.action_spec.shape[-1]

        # thrust
        if self.cfg.task.include_effort:
            observation_dim += 1

        self.observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "observation": UnboundedContinuousTensorSpec((1, observation_dim), device=self.device),
                "pursuer_state": UnboundedContinuousTensorSpec((1, pursuer_state_dim), device=self.device),
                "intrinsics": self.pursuer.intrinsics_spec_flattened.unsqueeze(0)
            })
        }).expand(self.num_envs).to(self.device)

        self.action_spec = CompositeSpec({
            "agents": CompositeSpec({
                "action": self.pursuer.action_spec.unsqueeze(0),
            })
        }).expand(self.num_envs).to(self.device)

        self.reward_spec = CompositeSpec({
            "agents": CompositeSpec({
                "reward": UnboundedContinuousTensorSpec((1, 1))
            })
        }).expand(self.num_envs).to(self.device)

        self.agent_spec["drone"] = AgentSpec(
            "drone", 1,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
            state_key=("agents", "intrinsics"),
        )

        stats_spec = CompositeSpec({
            "return": UnboundedContinuousTensorSpec(1),
            "reward_timestep": UnboundedContinuousTensorSpec(1),
            "reward_approach": UnboundedContinuousTensorSpec(1),
            "reward_catch": UnboundedContinuousTensorSpec(1),
            "reward_body_rates": UnboundedContinuousTensorSpec(1),
            "reward_bounds": UnboundedContinuousTensorSpec(1),
            "reward_escape": UnboundedContinuousTensorSpec(1),
            "reward_action_smoothness": UnboundedContinuousTensorSpec(1),
            "reward_effort": UnboundedContinuousTensorSpec(1),
            "episode_len": UnboundedContinuousTensorSpec(1),
            "reward_heading": UnboundedContinuousTensorSpec(1),
            "error_pos": UnboundedContinuousTensorSpec(1),
            "heading_alignment": UnboundedContinuousTensorSpec(1),
            "evader_caught": BinaryDiscreteTensorSpec(1, dtype=bool),
            "max_lin_vel": UnboundedContinuousTensorSpec(1),
            "max_ang_vel": UnboundedContinuousTensorSpec(1),
            "avg_lin_vel": UnboundedContinuousTensorSpec(1),
            "avg_ang_vel": UnboundedContinuousTensorSpec(1),
            "effort": UnboundedContinuousTensorSpec(1),
        }).expand(self.num_envs).to(self.device)

        self.stats = stats_spec.zero()

    def _reset_idx(self, env_ids: torch.Tensor):
        for info in self.evaders_map.values():
            idxs = info["idxs"]
            mask = torch.isin(env_ids, idxs)
            if not mask.any() or len(idxs) == 0:
                continue
            group_envs = env_ids[mask]
            p, v = info["reset_method"](group_envs, info)

            self.pos_evaders[group_envs] = p
            self.vel_evaders[group_envs] = v

        rot_evaders = euler_to_quaternion(self.init_rpy_dist.sample(env_ids.shape))
        self.evader._reset_idx(env_ids, self.training)
        self.evader.set_world_poses(
            self.pos_evaders[env_ids,...] + self.envs_positions[env_ids], rot_evaders, env_ids
        )
        self.evader.set_velocities(self.vel_evaders[env_ids,...], env_ids)
        
        if self.pursuer_reset is not None:
            pos_pursuers, vel_pursuers = self.pursuer_reset(env_ids)
        else:
            pos_pursuers, vel_pursuers = self._reset_within_arena_min_separation(
                env_ids, self.pos_evaders[env_ids,...], min_sep=self.min_initial_separation
            )

            if self.cfg.task.env.start_with_initial_velocity:
                vel_pursuers = D.Normal(
                    loc=torch.zeros_like(vel_pursuers[0,:]),
                    scale=torch.ones_like(vel_pursuers[0,:])*self.cfg.task.env.vel0_std
                ).sample((vel_pursuers.shape[0],)) 
                
                vel_pursuers[..., :3]*=self.K_V
                vel_pursuers[..., 3:]*=self.K_OMEGA

                time_to_collision_proxy = pos_pursuers[..., 2]/vel_pursuers[..., 2] 
                too_close_to_ground = (-1 < time_to_collision_proxy) & (time_to_collision_proxy <= 0)
                vel_pursuers[too_close_to_ground, 2] = torch.zeros_like(vel_pursuers[too_close_to_ground, 2])

        rot_pursuers = euler_to_quaternion(self.init_rpy_dist.sample(env_ids.shape))
        self.pursuer._reset_idx(env_ids, self.training)
        self.pursuer.set_world_poses(
            pos_pursuers + self.envs_positions[env_ids], rot_pursuers, env_ids
        )
        self.pursuer.set_velocities(vel_pursuers, env_ids)

        if self.prev_evader_steps_fix > 0:
            rel0  = (self.pos_evaders[env_ids,...] - pos_pursuers).unsqueeze(1)  # shape [len(env_ids),1,3]
            self.prev_error_pos[env_ids] = rel0.expand(-1, self.prev_evader_steps_fix, -1)

        if self.prev_traj_steps > 0:
            pos0 = pos_pursuers.unsqueeze(1)
            self.prev_pos[env_ids] = pos0.expand(-1, self.prev_traj_steps, -1)

        self.prev_action[env_ids] = 0.0
        self.action_difference[env_ids] = 0.0

        if self.cfg.task.include_effort:
            self.effort[env_ids] = 0.0
            self.effort_evader[env_ids] = 0.0

        self.stats.exclude("evader_caught")[env_ids] = 0.
        self.stats["evader_caught"][env_ids] = False

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("agents", "action")]

        if self.pursuer_step is not None:
            env_ids = torch.arange(self.num_envs, device=self.device)
            self.effort = self.pursuer_step(env_ids)
        else:
            self.effort = self.pursuer.apply_action(actions)
            self.action_difference = actions - self.prev_action
            self.prev_action = actions.clone()

        # For self-play, try to integrate it better
        # pursuer_actions = self.cont_pursuer(self.pursuer.get_state()[...,:13], 
        #                 target_vel= actions[..., :3],
        #                   )
        # self.effort = self.pursuer.apply_action(pursuer_actions)

        for info in self.evaders_map.values():
            idxs = info["idxs"]
            step_fn = info["step_method"]
            if len(idxs) == 0:
                continue
            step_actions = step_fn(idxs, info)
            self.evader_actions[idxs, ...] = step_actions
        self.effort_evader = self.evader.apply_action(self.evader_actions)


    def _compute_state_and_obs(self):
        self.pursuer_state = self.pursuer.get_state()
        self.pos, self.rot, self.lin_vel, self.ang_vel, self.heading, self.up, self.throttle = split_state(self.pursuer_state)  
        self.body_rates = quat_rotate_inverse(self.rot, self.ang_vel)

        evader_pos = self.evader.get_state()[..., :3]

        if self.cfg.task.env.include_noise_in_observations:
            pos_noise = D.Normal(
                loc=torch.zeros_like(evader_pos[0,:]),
                scale=torch.ones_like(evader_pos[0,:])*0.05
            ).sample((evader_pos.shape[0],))

            evader_pos += pos_noise

        obs = []
        # Time
        if self.time_encoding:
            t = (self.progress_buf / self.max_episode_length).unsqueeze(-1)
            obs.append(t.expand(-1, self.time_encoding_dim).unsqueeze(1))

        # Evader position
        self.error_pos = evader_pos - self.pos
        if self.prev_evader_traj_steps > 0:
            offset = self.prev_evader_steps_fix - self.prev_evader_traj_steps
            hist = self.prev_error_pos[:,offset:,:] /self.K_RP
            obs.append(hist.view(self.num_envs, -1).unsqueeze(1))
        error_pos_norm = self.error_pos/self.K_RP + 0
        obs.append(error_pos_norm)

        if self.cfg.task.include_distances:
            if self.prev_evader_traj_steps > 0:
                hist_dist = hist.norm(dim=-1, keepdim=True)
                obs.append(hist_dist.view(self.num_envs, -1).unsqueeze(1))
            current_dist = error_pos_norm.norm(dim=-1, keepdim=True) + 0
            obs.append(current_dist)
            # obs.append(torch.cat([hist_dist, current_dist], dim=1).view(self.num_envs, -1).unsqueeze(1))

        if self.cfg.task.include_closing_velocity:
            v_evader, _ = get_velocity_from_state(self.evader.get_state())
            closing_velocity = -(self.lin_vel - v_evader) / self.K_V
            obs.append(closing_velocity)

        # Heading to target
        error_pos_unit = self.error_pos / self.error_pos.norm(dim=-1, keepdim=True)
        self.heading_tt = (self.heading * error_pos_unit).sum(dim=-1)
        if self.cfg.task.include_heading_to_target:
            obs.append(self.heading_tt.unsqueeze(-1))

        # z coordinate
        if self.prev_traj_steps > 0:
            z_hist = (self.prev_pos - self.arena_center)/self.K_RP
            z_hist = z_hist[..., 2]
            obs.append(z_hist.view(self.num_envs, -1).unsqueeze(1))

        pos = (self.pos - self.arena_center)/self.K_P
        obs.append(pos[..., 2].unsqueeze(-1))

        # Orientation, velocity, angular velocity
        rot_matrix = quaternion_to_rotation_matrix(self.rot).flatten(start_dim=-2)
        obs.append(rot_matrix)
        obs.append(self.lin_vel/self.K_V)

        if self.cfg.task.use_body_rates:
            obs.append(self.body_rates/self.K_OMEGA)
        else:
            obs.append(self.ang_vel/self.K_OMEGA)

        if self.cfg.task.include_last_action:
            obs.append(self.prev_action)

        if self.cfg.task.include_effort:
            obs.append(self.effort.unsqueeze(-1)/self.K_EFFORT)

        obs = torch.cat(obs, dim=-1)

        return TensorDict(
            {
                "agents": {
                    "observation": obs,
                    "pursuer_state": self.pursuer_state,
                    "intrinsics": self.pursuer.intrinsics,
                },
            },
            self.batch_size,
        )
    
    def _compute_reward_and_done(self):
        prev_error_pos_norm = self.stats["error_pos"]
        error_pos_norm = self.error_pos.norm(dim=-1)

        reward_timestep = -self.reward_timestep 

        reward_approach = self.reward_approach_weight*(prev_error_pos_norm - error_pos_norm)

        # distance
        # reward_approach = -error_pos_norm/self.K_RP.norm(dim=-1) * 1

        evader_caught = (error_pos_norm < self.final_separation)
        reward_catch  = self.reward_evader_caught*evader_caught.float()*self.W_INT

        if self.cfg.task.use_body_rates:
            reward_body_rates = -self.reward_body_rates_weight*torch.norm(self.body_rates, dim=-1)
        else:
            reward_body_rates = -self.reward_body_rates_weight*torch.norm(self.ang_vel, dim=-1)

        if self.cfg.task.include_last_action:
            reward_action_smoothness = -self.reward_action_smoothness*torch.norm(self.action_difference, dim=-1)
        else:
            reward_action_smoothness = 0.0

        if self.cfg.task.include_effort:
            reward_effort = self.reward_effort_weight * torch.exp(-self.effort)
        else:
            reward_effort = 0.0

        if self.cfg.task.include_heading_to_target:
            reward_heading = self.reward_heading_weight*self.heading_tt
        else:
            reward_heading = 0.0
        
        z_violation = torch.relu(self.arena_min[2] - self.pos[..., 2])  
        reward_bounds = -self.reward_out_of_bounds * (z_violation > 0).float()

        truncated  = (self.progress_buf >= self.max_episode_length - 1).unsqueeze(-1)
        misbehave_pursuer = (self.pos[..., 2] < self.z_stop_pursuer) # collision -> evader is free, thus negative reward
        hasnan_pursuer = torch.isnan(self.pursuer_state).any(-1)
        if self.do_interception:
            w_reward = self.W_INT.clone()
        else:
            w_reward = self.W_INT.clone()
            large_distance = error_pos_norm > self.cfg.task.env.max_final_separation_track
            w_reward[misbehave_pursuer | hasnan_pursuer | large_distance] *= -1.0
        cond_escape = truncated | misbehave_pursuer | hasnan_pursuer
        reward_escape = -self.reward_evader_caught*cond_escape.float()*w_reward

        reward = (
            reward_timestep
            + reward_approach
            + reward_catch
            + reward_body_rates
            + reward_action_smoothness
            + reward_bounds
            + reward_escape
            + reward_heading
            + reward_effort
        )

        evader_state = self.evader.get_state()
        pos_e = evader_state[..., :3]
        misbehave_evader = (pos_e[..., 2] < self.z_stop_evader)
        misbehave = misbehave_pursuer | misbehave_evader
        
        hasnan_evader = torch.isnan(evader_state).any(-1)
        hasnan = hasnan_pursuer | hasnan_evader
        
        terminated = misbehave | hasnan | evader_caught

        evader_caught = evader_caught | misbehave_evader | hasnan_evader

        if terminated.any():
            a=2

        self.stats["error_pos"].lerp_(error_pos_norm, (1-self.alpha))
        self.stats["heading_alignment"].lerp_(self.heading_tt, (1-self.alpha))
        self.stats["return"] += reward
        self.stats["reward_timestep"] += reward_timestep
        self.stats["reward_approach"] += reward_approach
        self.stats["reward_catch"] += reward_catch
        self.stats["reward_body_rates"] += reward_body_rates
        self.stats["reward_bounds"] += reward_bounds
        self.stats["reward_escape"] += reward_escape
        self.stats["reward_heading"] += reward_heading
        self.stats["reward_effort"] += reward_effort
        self.stats["reward_action_smoothness"] += reward_action_smoothness
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)
        self.stats["evader_caught"].bitwise_or_(evader_caught)

        self.stats["max_lin_vel"] = torch.max(self.stats["max_lin_vel"], self.lin_vel.norm(dim=-1))
        self.stats["max_ang_vel"] = torch.max(self.stats["max_ang_vel"], self.ang_vel.norm(dim=-1))

        n_steps = self.progress_buf.unsqueeze(1)
        self.stats["avg_lin_vel"] = (self.stats["avg_lin_vel"]*(n_steps - 1) + self.lin_vel.norm(dim=-1))/ n_steps 
        self.stats["avg_ang_vel"] = (self.stats["avg_ang_vel"]*(n_steps - 1) + self.ang_vel.norm(dim=-1))/ n_steps

        self.stats["effort"] += self.effort

        return TensorDict(
            {
                "agents": {
                    "reward": reward.unsqueeze(-1),
                },
                "done":       terminated | truncated,
                "terminated": terminated,
                "truncated":  truncated,
                "stats": self.stats.clone()
            },
            self.batch_size,
        )
    
    def _update_history(self):
        if self.prev_evader_steps_fix > 0:
            self.prev_error_pos = torch.cat([self.prev_error_pos[:, 1:, :], self.error_pos], dim=1)

        if self.prev_traj_steps > 0:
            self.prev_pos = torch.cat([self.prev_pos[:, 1:, :], self.pos], dim=1)

    def _sample_pursuer_controllers(self):
        pursuer_policy = self.cfg.task.pursuer_model.policy
    
        if "fixed_path" in pursuer_policy:
            traj_type = pursuer_policy.split(":")[1]
            n_trajs = self.num_envs
            self.pursuer_trajectory = Trajectory.REGISTRY[traj_type](n_trajs, self.device,
                                                  self.arena_min, self.arena_max,
                                                  self.arena_size)
            self.pursuer_step = self._step_fixed_path_pursuer
            self.pursuer_reset = self._reset_fixed_path_pursuer

            self.z_stop_pursuer = -2.0

        elif "pid" in pursuer_policy or "frpn" in pursuer_policy:
            self.pursuer_step = self._step_geom_pursuit
            self.pursuer_reset  = self._reset_pid_pn
            n_envs = self.num_envs
            self.cont_pursuer.allocate_arrays(n_envs, self.device)
            self.pursuer_trajectory = None

            self.z_stop_pursuer = -2.0

        else:
            self.pursuer_trajectory = None
            self.pursuer_step = None
            self.pursuer_reset = None

            self.z_stop_pursuer = self.cfg.task.env.z_stop


    def _sample_evader_controllers(self):
        pool_evader_policies = self.cfg.task.evader_model.policy
        cont_evaders = self.cont_evader
        samples = self.num_envs
        prob_old_policy = self.cfg.task.prob_old_policy

        if len(pool_evader_policies) == 1:
            probs = {pool_evader_policies[0]: 1.0}
        else:
            probs = {policy: prob_old_policy / (len(pool_evader_policies) - 1) for policy in pool_evader_policies[0:]}
            probs[pool_evader_policies[-1]] = 1.0 - prob_old_policy
            
        assignment = policy_sampling(probs, samples)
        print(f"Policy sampling for {pool_evader_policies} with {samples} samples:")
        print(f"Probabilities: {probs}")
        for policy, idxs in assignment.items():
            print(f"{policy:>8s}: ({len(idxs)}): {idxs}")

        self.evaders_map = {}
        for policy, ii in zip(pool_evader_policies, range(len(pool_evader_policies))):
            if "fixed_path" in policy:
                traj_type = policy.split(":")[1]
                n_trajs = len(assignment[policy])
                traj = Trajectory.REGISTRY[traj_type](n_trajs, self.device,
                                                    self.arena_min, self.arena_max,
                                                    self.arena_size)
                step = self._step_fixed_path
                reset = self._reset_fixed_path
                extra = None
            elif "repel" in policy:
                step = self._step_repel
                reset = self._reset_within_arena
                traj = None
                extra = None
            elif "thrust" in policy or "velocity" in policy or "rate" in policy:
                step = self._step_policy
                reset = self._reset_within_arena
                traj = None
                extra = policy
            else:
                Warning(f"Policy {policy} not implemented, using None")
                step = None
                reset = None
                traj = None
                extra = None
            
            idxs = torch.tensor(assignment[policy], dtype=torch.long, device=self.device)
            self.evaders_map[policy] = {
                "idxs": idxs,
                "controller": cont_evaders[ii],
                "step_method": step,
                "reset_method": reset,
                "trajectory": traj,
                "extra": extra
            }

        self.z_stop_evader = self.cfg.task.env.z_stop # always the same
    
    def _compute_traj(self, steps: int, env_ids=None, step_size: float=1., trajectory=None):
        if env_ids is None:
            env_ids = ...

        t = self.progress_buf[env_ids].unsqueeze(1) + step_size * torch.arange(steps, device=self.device)
        t *= self.dt
        env_ids_centered = env_ids - env_ids[0]  # center the env_ids to start from 0
        target_pos = trajectory.compute_ref(t, env_ids_centered)

        return target_pos
    
    
    def _step_policy(self, env_ids:torch.Tensor, info):
        evader_state = self.evader.get_state()[env_ids, ..., :23]
        pos, rot, lin_vel, ang_vel, heading, up, _  = split_state(evader_state)  
        body_rates = quat_rotate_inverse(rot, ang_vel)

        pursuer_pos = self.pursuer.get_state()[env_ids, ..., :3]

        if self.cfg.task.env.include_noise_in_observations:
            pos_noise = D.Normal(
                loc=torch.zeros_like(pursuer_pos[0,:]),
                scale=torch.ones_like(pursuer_pos[0,:])*0.05
            ).sample((pursuer_pos.shape[0],))

            pursuer_pos += pos_noise

        obs = []
        # Time
        if self.time_encoding:
            t = (self.progress_buf[env_ids] / self.max_episode_length).unsqueeze(-1)
            obs.append(t.expand(-1, self.time_encoding_dim).unsqueeze(1))

        # Relative position to pursuer
        # Hot fix, use the same as this environment
        if self.prev_evader_traj_steps > 0:
            hist = -self.prev_error_pos/self.K_RP
            offset = self.prev_evader_steps_fix - self.prev_evader_traj_steps
            hist = hist[env_ids, offset:, :]
            obs.append(hist.view(env_ids.shape[0], -1).unsqueeze(1))

        error_pos = pursuer_pos - pos
        error_pos_norm = error_pos/self.K_RP 
        obs.append(error_pos_norm)

        if self.cfg.task.include_distances:
            if self.prev_evader_traj_steps > 0:
                hist_dist = hist.norm(dim=-1, keepdim=True)
                obs.append(hist_dist.view(env_ids.shape[0], -1).unsqueeze(1))

            current_dist = error_pos_norm.norm(dim=-1, keepdim=True) + 0
            obs.append(current_dist)
            # obs.append(torch.cat([hist_dist, current_dist], dim=1).view(env_ids.shape[0], -1).unsqueeze(1))

        if self.cfg.task.include_closing_velocity:
            v_pursuer = self.lin_vel[env_ids,...]
            v_evader = lin_vel
            closing_velocity = -(v_evader - v_pursuer) / self.K_V
            obs.append(closing_velocity)

        # Heading to pursuer
        if self.cfg.task.include_heading_to_target:
            error_pos_unit = error_pos / error_pos.norm(dim=-1, keepdim=True)
            heading_tt = (heading * error_pos_unit).sum(dim=-1)
            obs.append(heading_tt.unsqueeze(-1))

        # Self position
        obs.append((pos - self.arena_center)/self.K_P)

        # Orientation, velocity, angular velocity
        rot_matrix = quaternion_to_rotation_matrix(rot).flatten(start_dim=-2)
        obs.append(rot_matrix)
        obs.append(lin_vel/self.K_V)

        if self.cfg.task.use_body_rates:
            obs.append(body_rates/self.K_OMEGA)
        else:
            obs.append(ang_vel/self.K_OMEGA)

        if self.cfg.task.include_effort:
            effort = self.effort_evader[env_ids, ...].unsqueeze(-1) / self.K_EFFORT
            obs.append(effort)

        obs = torch.cat(obs, dim=-1)

        td = TensorDict(
            {"agents": {"observation": obs}},  
            batch_size=env_ids.shape[0],
        ).to(self.device)

        with torch.no_grad():
            out_td = info["policy"](td)
        action_evader = out_td.get(("agents","action"))

        if "rate" in info["extra"]:
            target_rate, target_thrust = action_evader.split([3, 1], dim=-1)
            max_thrust = info["controller"].max_thrusts.sum(-1)
            target_thrust = ((target_thrust + 1) / 2).clip(0.) * max_thrust
            action_evader = info["controller"](
                evader_state[..., :13],
                target_rate=target_rate*torch.pi,
                target_thrust=target_thrust
            )
            torch.nan_to_num_(action_evader, 0.)

        elif "velocity" in info["extra"]:
            target_vel, target_yaw_rate = action_evader.split([3,1], dim=-1)
            target_vel = target_vel * self.K_V
            action_evader = info["controller"](
                root_state=evader_state[..., :13],
                target_pos=None,
                target_vel=target_vel,
                target_acc=None,
                target_yaw=None, 
                target_yaw_rate=target_yaw_rate * torch.pi
            )
            torch.nan_to_num_(action_evader, 0.)

        elif "thrust" in info["extra"]:
            pass # no action needed to transform the actions here
        else:
            raise NotImplementedError

        return action_evader
    
    def _step_fixed_path(self, env_ids: torch.Tensor, info):
        evader_state = self.evader.get_state()[env_ids, ..., :13]  
        target_pos = self._compute_traj(steps=1, step_size=5, env_ids=env_ids, trajectory=info["trajectory"])
        action_evader = info["controller"](evader_state, target_pos)
        return action_evader
        # self.evader.apply_action(action_evader)

    def _step_geom_pursuit(self, env_ids: torch.Tensor):
        state_p = self.pursuer.get_state()[env_ids, ..., :13]
        state_e = self.evader.get_state()[env_ids, ..., :13]
        
        try:
            action_pursuer = self.cont_pursuer(state_p, state_e)
        except Exception as e:
            print(f"Error in controller: {e}")
            action_pursuer = torch.zeros_like(state_p[..., :4])
        return self.pursuer.apply_action(action_pursuer)

    def _step_fixed_path_pursuer(self, env_ids: torch.Tensor):
        pursuer_state = self.pursuer.get_state()[env_ids, ..., :13]  
        target_pos = self._compute_traj(steps=1, step_size=5, env_ids=env_ids, trajectory=self.pursuer_trajectory)
        action_pursuer = self.cont_pursuer(pursuer_state, target_pos)
        return self.pursuer.apply_action(action_pursuer)

    def _reset_fixed_path(self, env_ids: torch.Tensor, info):
        trajectory = info["trajectory"]
        env_ids_centered = env_ids - info["idxs"][0]  
        trajectory.reset(env_ids_centered) 
        t0 = torch.zeros([len(env_ids_centered),1], device=self.device)
        t1 = torch.full((len(env_ids_centered), 1), self.dt, device=self.device)
        pos0 = trajectory.compute_ref(t0, env_ids_centered).squeeze(1)  
        pos1 = trajectory.compute_ref(t1, env_ids_centered).squeeze(1)   
        lin_vel0 = (pos1 - pos0) / self.dt     
        vel0 = torch.zeros(len(env_ids_centered), 6, device=self.device)
        vel0[..., :3] = lin_vel0
        return pos0, vel0
    
    def _reset_fixed_path_pursuer(self, env_ids: torch.Tensor):
        trajectory = self.pursuer_trajectory  
        trajectory.reset(env_ids) 
        t0 = torch.zeros([len(env_ids),1], device=self.device)
        t1 = torch.full((len(env_ids), 1), self.dt, device=self.device)
        pos0 = trajectory.compute_ref(t0, env_ids).squeeze(1)  
        pos1 = trajectory.compute_ref(t1, env_ids).squeeze(1)   
        lin_vel0 = (pos1 - pos0) / self.dt     
        vel0 = torch.zeros(len(env_ids), 6, device=self.device)
        vel0[..., :3] = lin_vel0
        return pos0, vel0

    def _reset_within_arena(self, env_ids: torch.Tensor, info=None):
        pos = D.Uniform(
            self.arena_min,
            self.arena_max
        ).sample([len(env_ids),])
        vel = torch.zeros(len(env_ids), 6, device=self.device)
        return pos, vel
    
    def _reset_pid_pn(self, env_ids: torch.Tensor):
        pos, vel = self._reset_within_arena(env_ids, None)
        self.cont_pursuer.reset(env_ids)
        return pos, vel

    def _reset_within_arena_min_separation(self, env_ids: torch.Tensor, pos_others: torch.Tensor, min_sep: float=0.5, info=None):
        pos = min_separation_sampling(self.arena_min, self.arena_max, pos_others, min_sep)
        vel = torch.zeros(len(env_ids), 6, device=self.device)
        return pos, vel
    
    def _step_repel(self, env_ids:torch.Tensor, info):
        state_p = self.pursuer.get_state()[env_ids, ..., :13]
        state_e = self.evader.get_state()[env_ids, ..., :13]

        action_evader = info["controller"](state_p, state_e)
        return action_evader
    
    def render_grids_arenas(self, env_ids: torch.Tensor):

        min_xyz = self.arena_min.tolist()
        max_xyz = self.arena_max.tolist()

        size_lines = 4

        def _draw_rect(min_xy, max_xy, z, offset):
            corners = [
                [min_xy[0] + offset[0], min_xy[1] + offset[1], z + offset[2]],
                [max_xy[0] + offset[0], min_xy[1] + offset[1], z + offset[2]],
                [max_xy[0] + offset[0], max_xy[1] + offset[1], z + offset[2]],
                [min_xy[0] + offset[0], max_xy[1] + offset[1], z + offset[2]],
                [min_xy[0] + offset[0], min_xy[1] + offset[1], z + offset[2]],
            ]
            p0 = corners[:-1]
            p1 = corners[1:]
            colors = [(0.0, 0.0, 0.0, 1.0)] * len(p0)
            sizes  = [size_lines] * len(p0)
            self.draw.draw_lines(p0, p1, colors, sizes)

        for env_id in env_ids.tolist():

            offset = self.envs_positions[env_id].tolist()

            _draw_rect(min_xyz[0:2], max_xyz[0:2], min_xyz[2], offset)
            _draw_rect(min_xyz[0:2], max_xyz[0:2], max_xyz[2], offset)

            # draw vertical edges between floor & ceiling (4 corners)
            floor_corners = [
                [min_xyz[0] + offset[0], min_xyz[1] + offset[1], min_xyz[2] + offset[2]],
                [max_xyz[0] + offset[0], min_xyz[1] + offset[1], min_xyz[2] + offset[2]],
                [max_xyz[0] + offset[0], max_xyz[1] + offset[1], min_xyz[2] + offset[2]],
                [min_xyz[0] + offset[0], max_xyz[1] + offset[1], min_xyz[2] + offset[2]],
            ]
            ceil_corners = [
                [x, y, max_xyz[2] + offset[2]] for x, y, _ in floor_corners
            ]
            # draw a line from each floor corner up to its ceiling corner
            colors = [(0.0, 0.0, 0.0, 1.0)] * 4
            sizes  = [size_lines] * 4
            self.draw.draw_lines(floor_corners, ceil_corners, colors, sizes)
