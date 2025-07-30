import logging
import os
import time

import hydra
import torch
import numpy as np
import pandas as pd
import wandb
import matplotlib.pyplot as plt

from torch.func import vmap
from tqdm import tqdm
from omegaconf import OmegaConf

from omni_drones import init_simulation_app
from torchrl.data import CompositeSpec
from torchrl.envs.utils import set_exploration_type, ExplorationType
from omni_drones.utils.torchrl import SyncDataCollector
from omni_drones.utils.torchrl.transforms import (
    FromMultiDiscreteAction,
    FromDiscreteAction,
    ravel_composite,
    AttitudeController,
    RateController,
    VelocityController
)
from omni_drones.utils.wandb import init_wandb
from omni_drones.utils.torchrl import RenderCallback, EpisodeStats
from omni_drones.learning import ALGOS

from omni_drones.utils.sampling import get_seeds_list

from setproctitle import setproctitle
from torchrl.envs.transforms import TransformedEnv, InitTracker, Compose

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path=".", config_name="benchmark_pursuer")
def main(cfg, simulation_app=None):
    if cfg.wandb.job_type == "debug":
        cfg.headless = False
        cfg.render_interval = 300
        cfg.task.env.num_envs = 3
        cfg.task.env.max_episode_length = 10000
        cfg.task.env.leave_spacing_between_envs = False

    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    simulation_app = init_simulation_app(cfg)

    run = init_wandb(cfg)
    setproctitle(run.name)
    print(OmegaConf.to_yaml(cfg))
    log.info(OmegaConf.to_yaml(cfg))

    from omni_drones.envs.isaac_env import IsaacEnv

    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=cfg.headless)

    transforms = [InitTracker()]

    action_transform: str = cfg.task.get("action_transform", None)
    if action_transform is not None and not action_transform.startswith("None"):
        if action_transform.startswith("multidiscrete"):
            nbins = int(action_transform.split(":")[1])
            transform = FromMultiDiscreteAction(nbins=nbins)
            transforms.append(transform)

        elif action_transform.startswith("discrete"):
            nbins = int(action_transform.split(":")[1])
            transform = FromDiscreteAction(nbins=nbins)
            transforms.append(transform)

        elif action_transform.startswith("velocity"):
            transform = VelocityController(controller=base_env.cont_pursuer,
                                           max_vel=base_env.K_V,
                                           in_keys_inv=("agents", "pursuer_state"),
                                           )
            transforms.append(transform)

        elif action_transform.startswith("rate"):
            transform = RateController(controller=base_env.cont_pursuer,
                                       in_keys_inv=("agents", "pursuer_state"),
                                       )
            transforms.append(transform)

        elif action_transform.startswith("thrust"):
            pass

        else:
            raise NotImplementedError(f"Unknown action transform: {action_transform}")

    env = TransformedEnv(base_env, Compose(*transforms)).train()
    try:
        policy = ALGOS[cfg.algo.name.lower()](
            cfg.algo,
            env.observation_spec,
            env.action_spec,
            env.reward_spec,
            device=base_env.device
        )

        if cfg.task.pursuer_model.source_policy is not None and "None" not in cfg.task.pursuer_model.source_policy:
            source_model_path = cfg.task.pursuer_model.source_policy
            source_model = run.use_artifact(f"{source_model_path}:{cfg.use_model}")
            source_model.download()
            folder_root = list(source_model._download_roots)[0]
            files = os.listdir(folder_root)  # Ensure the folder is downloaded
            ckpt_path = folder_root + f"/{files[-1]}"
            # ckpt_path = list(source_model._download_roots)[0] + "/checkpoint_final.pt"
            weights = torch.load(ckpt_path, map_location=base_env.device)
            policy.load_state_dict(weights)

        for policy_name, policy_src, idx in zip(cfg.task.evader_model.policy, 
                                                cfg.task.evader_model.source_policy, 
                                                range(len(cfg.task.evader_model.source_policy))):
            if "None" not in policy_src:
                from torchrl.data import UnboundedContinuousTensorSpec
                # Hot fix: we assume pursuer and evader ppo share the same features
                observation_spec_evader = env.observation_spec.clone()
                shape = torch.tensor(observation_spec_evader["agents", "observation"].shape, device=base_env.device)
                shape[2] += 2
                observation_spec_evader["agents", "observation"] = UnboundedContinuousTensorSpec(shape, device=base_env.device)
                cfg.algo.priv_critic = False
                env.evaders_map[policy_name]["policy"] = ALGOS[cfg.algo.name.lower()](
                    cfg.algo,
                    observation_spec_evader,
                    env.action_spec,
                    env.reward_spec,
                    device=base_env.device
                )
                source_model_path = policy_src
                source_model = run.use_artifact(f"{source_model_path}:{cfg.use_model}")
                source_model.download()
                folder_root = list(source_model._download_roots)[0]
                files = os.listdir(folder_root)  # Ensure the folder is downloaded
                ckpt_path = folder_root + f"/{files[-1]}"
                # ckpt_path = list(source_model._download_roots)[0] + "/checkpoint_final.pt"
                weights = torch.load(ckpt_path, map_location=base_env.device)
                env.evaders_map[policy_name]["policy"].load_state_dict(weights)
                env.evaders_map[policy_name]["policy"].eval()
    except KeyError:
        raise NotImplementedError(f"Unknown algorithm: {cfg.algo.name}")

    @torch.no_grad()
    def evaluate(
        seed: int=0,
        exploration_type: ExplorationType=ExplorationType.MODE,
        render: bool=False,
    ):
        if render:
            base_env.enable_render(True)
            render_callback = RenderCallback(interval=2)
        else:
            base_env.enable_render(False)
            render_callback = None
        base_env.eval()
        env.eval()
        env.set_seed(seed)

        with set_exploration_type(exploration_type):
            trajs = env.rollout(
                max_steps=base_env.max_episode_length,
                policy=policy,
                callback=render_callback,
                auto_reset=True,
                break_when_any_done=False,
                return_contiguous=False,
            )
        base_env.enable_render(not cfg.headless)
        env.reset()

        done = trajs.get(("next", "done"))
        first_done = torch.argmax(done.long(), dim=1).cpu()

        def take_first_episode(tensor: torch.Tensor):
            indices = first_done.reshape(first_done.shape+(1,)*(tensor.ndim-2))
            return torch.take_along_dim(tensor, indices, dim=1).reshape(-1)

        traj_stats = {
            k: take_first_episode(v)
            for k, v in trajs[("next", "stats")].cpu().items()
        }

        info = {
            "eval/stats." + k: torch.mean(v.float()).item()
            for k, v in traj_stats.items()
        }

        for k, v in env.evaders_map.items():
            for k2, v2 in traj_stats.items():
                key_name = f"eval/stats.{k2}.{k}"
                counter = 0
                while key_name in info:
                    key_name = f"{key_name}_{counter}"
                    counter += 1
                info[key_name] = v2[v["idxs"].cpu()].float().mean().item()

        if render:
            info["recording"] = wandb.Video(
                render_callback.get_video_array(axes="t c h w"),
                fps=0.5 / (cfg.sim.dt * cfg.sim.substeps),
                format="mp4"
            )

        return info

    info = {"env_frames": 0}
    render_flag = cfg.render_interval > 0
    seeds_list = get_seeds_list(cfg.n_seeds)
    info_per_seed = []
    for seed in tqdm(seeds_list, desc="Running seeds"):
        info_eval = evaluate(seed=seed, render=render_flag)
        info.update(info_eval)
        info_per_seed.append(info_eval)
    run.log(info)

    main_metrics_keys = [
        "evader_caught",
        "effort",
        "episode_len",
        "heading_alignment",
        "max_ang_vel",
        "max_lin_vel",
        "avg_lin_vel",
        "avg_ang_vel",
        "return",
    ]

    today_date = time.strftime("%Y-%m-%d")
    today_hour = time.strftime("%H-%M-%S")
    data_name = f"csv/benchmark_pursuer_{today_date}_{today_hour}.csv"

    info_per_seed = [
        {k: v for k, v in d.items() if any(metric in k for metric in main_metrics_keys)}
        for d in info_per_seed
    ]
    runs_data = pd.DataFrame(info_per_seed)
    avg_row = pd.DataFrame([runs_data.mean(numeric_only=True)])
    df = pd.concat([runs_data, avg_row], ignore_index=True)
    df.to_csv(data_name, index=False)
    wandb.finish()

    print(f"BENCH_CSV_PATH={data_name}")
    log.info(f"BENCH_CSV_PATH={data_name}")

    print(f"BENCH_RUN_PATH={run.path}")  
    log.info(f"BENCH_RUN_PATH={run.path}")

    simulation_app.close()


if __name__ == "__main__":
    main()
