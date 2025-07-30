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


import datetime
import logging
import os

import wandb
from omegaconf import OmegaConf


def dict_flatten(a: dict, delim="."):
    """Flatten a dict recursively.
    Examples:
        >>> a = {
                "a": 1,
                "b":{
                    "c": 3,
                    "d": 4,
                    "e": {
                        "f": 5
                    }
                }
            }
        >>> dict_flatten(a)
        {'a': 1, 'b.c': 3, 'b.d': 4, 'b.e.f': 5}
    """
    result = {}
    for k, v in a.items():
        if isinstance(v, dict):
            result.update({k + delim + kk: vv for kk, vv in dict_flatten(v).items()})
        else:
            result[k] = v
    return result


def init_wandb(cfg):
    """Initialize WandB.

    If only `run_id` is given, resume from the run specified by `run_id`.
    If only `run_path` is given, start a new run from that specified by `run_path`,
        possibly restoring trained models.

    Otherwise, start a fresh new run.

    """
    wandb_cfg = cfg.wandb
    time_str = datetime.datetime.now().strftime("%m-%d_%H-%M")
    run_name = f"{wandb_cfg.run_name}/{time_str}"
    kwargs = dict(
        project=wandb_cfg.project,
        group=wandb_cfg.group,
        entity=wandb_cfg.entity,
        name=run_name,
        mode=wandb_cfg.mode,
        tags=wandb_cfg.tags,
    )
    if wandb_cfg.run_id is not None:
        kwargs["id"] = wandb_cfg.run_id
        kwargs["resume"] = "must"
    else:
        kwargs["id"] = wandb.util.generate_id()
    run = wandb.init(**kwargs)
    cfg_dict = dict_flatten(OmegaConf.to_container(cfg))
    run.config.update(cfg_dict)
    return run


class ConfigList:
    """A class to manage a list of configurations for WandB overrides."""
    def __init__(self, cfg=None, pursuer=False):
        self.cfg = cfg
        self.pursuer = pursuer
        if cfg is not None:
            self.create_override_list()


    def create_override_list(self):
        self.overrides = []

        self.overrides.append(f"seed={self.cfg.seed}")

        self.overrides.append(f"use_model={self.cfg.use_model}")

        self.overrides.append(f"total_frames={self.cfg.total_frames}")
        self.overrides.append(f"render_interval={self.cfg.render_interval}")

        self.overrides.append(f"task.prob_old_policy={self.cfg.task.prob_old_policy}")

        self.overrides.append(f"task.env.num_envs={self.cfg.task.env.num_envs}")
        self.overrides.append(f"task.env.arena_size={self.cfg.task.env.arena_size}")
        self.overrides.append(f"task.env.max_episode_length={self.cfg.task.env.max_episode_length}")
        self.overrides.append(f"task.env.start_with_initial_velocity={self.cfg.task.env.start_with_initial_velocity}")

        if self.pursuer:
            self.overrides.append(f"task.pursuer_model.name={self.cfg.task.pursuer_model.name}")
            self.overrides.append(f"task.pursuer_model.policy={self.cfg.task.pursuer_model.policy}")

            self.overrides.append(f"task.evader_model.name={self.cfg.task.evader_model.name}")                                                   
        else:
            self.overrides.append(f"task.evader_model.name={self.cfg.task.evader_model.name}")
            self.overrides.append(f"task.evader_model.policy={self.cfg.task.evader_model.policy}")

            self.overrides.append(f"task.pursuer_model.name={self.cfg.task.pursuer_model.name}") 
        
        self.overrides.append(f"task.include_distances={self.cfg.task.include_distances}")
        self.overrides.append(f"task.include_last_action={self.cfg.task.include_last_action}")
        self.overrides.append(f"task.include_heading_to_target={self.cfg.task.include_heading_to_target}")

        self.overrides.append(f"task.use_body_rates={self.cfg.task.use_body_rates}")


    def add_replace_override(self, override):
        override_key, override_value = override.split("=")
        for (item, idx) in zip(self.overrides, range(len(self.overrides))):
            if override_key in item:
                self.overrides[idx] = override
                return
            
        print(f"Adding new override: {override}")
        self.overrides.append(override)

    def get_list(self):
        return self.overrides
    
class ConfigListAdversarial:
    """A class to manage a list of configurations for WandB overrides."""
    def __init__(self, cfg=None):
        self.cfg = cfg
        if cfg is not None:
            self.create_override_list()


    def create_override_list(self):
        self.overrides = []

        self.overrides.append(f"seed={self.cfg.seed}")

        self.overrides.append(f"optional_tag={self.cfg.optional_tag}")
        self.overrides.append(f"use_model={self.cfg.use_model}")

        self.overrides.append(f"task.prob_old_policy={self.cfg.task.prob_old_policy}")

        self.overrides.append(f"task.env.num_envs={self.cfg.task.env.num_envs}")
        self.overrides.append(f"task.env.arena_size={self.cfg.task.env.arena_size}")
        self.overrides.append(f"task.env.max_episode_length={self.cfg.task.env.max_episode_length}")
        self.overrides.append(f"task.env.start_with_initial_velocity={self.cfg.task.env.start_with_initial_velocity}")
        
        self.overrides.append(f"task.pursuer_model.name={self.cfg.task.pursuer_model.name}")
        self.overrides.append(f"task.evader_model.name={self.cfg.task.evader_model.name}")
        
        self.overrides.append(f"task.include_distances={self.cfg.task.include_distances}")
        self.overrides.append(f"task.include_last_action={self.cfg.task.include_last_action}")
        self.overrides.append(f"task.include_heading_to_target={self.cfg.task.include_heading_to_target}")

        self.overrides.append(f"wandb.job_type={self.cfg.wandb.job_type}")

    def add_replace_override(self, override):
        override_key, override_value = override.split("=")
        for (item, idx) in zip(self.overrides, range(len(self.overrides))):
            if override_key in item:
                self.overrides[idx] = override
                return
            
        print(f"Adding new override: {override}")
        self.overrides.append(override)

    def get_list(self):
        return self.overrides
    

def get_variable_from_out_text(run_out, variable_name):
    line = next(l for l in run_out.splitlines() if l.startswith(f"{variable_name}="))
    return line.split("=",1)[1]

def fetch_run_summary(run_id):
    run = wandb.Api().run(run_id)

    returns = run.summary.get("eval/stats.return")
    evader_caught = run.summary.get("eval/stats.evader_caught")
    episode_len = run.summary.get("eval/stats.episode_len")
    max_lin_vel = run.summary.get("eval/stats.max_lin_vel")
    max_ang_vel = run.summary.get("eval/stats.max_ang_vel")
    effort = run.summary.get("eval/stats.effort")

    stats = {
        "return": returns,
        "evader_caught": evader_caught,
        "episode_len": episode_len,
        "max_lin_vel": max_lin_vel,
        "max_ang_vel": max_ang_vel,
        "effort": effort,
    }
    return stats