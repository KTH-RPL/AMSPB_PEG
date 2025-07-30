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

from setproctitle import setproctitle
from torchrl.envs.transforms import TransformedEnv, InitTracker, Compose


@hydra.main(version_base=None, config_path=".", config_name="train_evader")
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    simulation_app = init_simulation_app(cfg)

    run = init_wandb(cfg)
    setproctitle(run.name)
    print(OmegaConf.to_yaml(cfg))

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
            transform = VelocityController(controller=base_env.cont_evader,
                                           max_vel=base_env.K_V,
                                           in_keys_inv=("agents", "evader_state"),
                                           )
            transforms.append(transform)

        elif action_transform.startswith("rate"):
            transform = RateController(controller=base_env.cont_evader,
                                       in_keys_inv=("agents", "evader_state"),
                                       )
            transforms.append(transform)

        elif action_transform.startswith("thrust"):
            pass

        else:
            raise NotImplementedError(f"Unknown action transform: {action_transform}")

    env = TransformedEnv(base_env, Compose(*transforms)).train()
    env.set_seed(cfg.seed)

    try:
        policy = ALGOS[cfg.algo.name.lower()](
            cfg.algo,
            env.observation_spec,
            env.action_spec,
            env.reward_spec,
            device=base_env.device
        )

        if "None" not in cfg.task.evader_model.source_policy:
            source_model_path = cfg.task.evader_model.source_policy
            source_model = run.use_artifact(f"{source_model_path}:{cfg.use_model}")
            source_model.download()
            folder_root = list(source_model._download_roots)[0]
            files = os.listdir(folder_root)  # Ensure the folder is downloaded
            ckpt_path = folder_root + f"/{files[-1]}"
            # ckpt_path = list(source_model._download_roots)[0] + "/checkpoint_final.pt"
            weights = torch.load(ckpt_path, map_location=base_env.device)
            policy.load_state_dict(weights)

        for policy_name, policy_src, idx in zip(cfg.task.pursuer_model.policy, 
                                                cfg.task.pursuer_model.source_policy, 
                                                range(len(cfg.task.pursuer_model.source_policy))):
            if "None" not in policy_src:
                from torchrl.data import UnboundedContinuousTensorSpec
                observation_spec_pursuer = env.observation_spec.clone()
                shape = torch.tensor(observation_spec_pursuer["agents", "observation"].shape, device=base_env.device)
                shape[2] -= 2
                observation_spec_pursuer["agents", "observation"] = UnboundedContinuousTensorSpec(shape, device=base_env.device)
                env.pursuers_map[policy_name]["policy"] = ALGOS[cfg.algo.name.lower()](
                    cfg.algo,
                    observation_spec_pursuer,
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
                env.pursuers_map[policy_name]["policy"].load_state_dict(weights)
                env.pursuers_map[policy_name]["policy"].eval()
    except KeyError:
        raise NotImplementedError(f"Unknown algorithm: {cfg.algo.name}")

    frames_per_batch = env.num_envs * int(cfg.algo.train_every)
    total_frames = cfg.get("total_frames", -1) // frames_per_batch * frames_per_batch
    max_iters = cfg.get("max_iters", -1)
    eval_interval = cfg.get("eval_interval", -1)
    save_interval = cfg.get("save_interval", -1)
    render_interval = cfg.get("render_interval", -1)

    stats_keys = [
        k for k in base_env.observation_spec.keys(True, True)
        if isinstance(k, tuple) and k[0]=="stats"
    ]
    episode_stats = EpisodeStats(stats_keys)
    collector = SyncDataCollector(
        env,
        policy=policy,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=cfg.sim.device,
        return_same_td=True,
    )

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

        if render:
            info["recording"] = wandb.Video(
                render_callback.get_video_array(axes="t c h w"),
                fps=0.5 / (cfg.sim.dt * cfg.sim.substeps),
                format="mp4"
            )

        return info

    consec_success = 0
    early_stop_threshold = 1.0 - cfg.get("early_stop_threshold", 0.95)
    early_stop_patience = cfg.get("early_stop_patience", 3)
    flag_early_stop = False
    model_name = cfg.wandb.run_name.lower().replace(":", "_") 
    best_caught_rate = float("inf")

    pbar = tqdm(collector, total=total_frames//frames_per_batch)
    env.train()
    for i, data in enumerate(pbar):
        info = {"env_frames": collector._frames, "rollout_fps": collector._fps}
        episode_stats.add(data.to_tensordict())

        if len(episode_stats) >= base_env.num_envs:
            stats = {
                "train/" + (".".join(k) if isinstance(k, tuple) else k): torch.mean(v.float()).item()
                for k, v in episode_stats.pop().items(True, True)
            }
            info.update(stats)

        info.update(policy.train_op(data.to_tensordict()))

        if eval_interval > 0 and i % eval_interval == 0:
            logging.info(f"Eval at {collector._frames} steps.")
            render_flag = False
            if render_interval > 0 and i % render_interval == 0:
                render_flag = True
            info.update(evaluate(render=render_flag))
            env.train()
            base_env.train()

            caught_rate = info.get("eval/stats.evader_caught", -1)
            if early_stop_patience > 0:
                if caught_rate <= early_stop_threshold:
                    consec_success += 1
                else:
                    consec_success = 0
                if consec_success >= early_stop_patience:
                    logging.info(
                        f"Early stopping: evader_caught ≥ {early_stop_threshold} "
                        f"{early_stop_patience}× in a row."
                    )
                    flag_early_stop = True

            if save_interval > 0 and i % save_interval == 0:
                try:
                    ckpt_path = os.path.join(run.dir, f"checkpoint_{collector._frames}.pt")
                    torch.save(policy.state_dict(), ckpt_path)
                    logging.info(f"Saved checkpoint to {str(ckpt_path)}")

                    model_artifact = wandb.Artifact(
                        model_name,
                        type="model",
                        description=f"Snapshot at {collector._frames} frames",
                        metadata=dict(cfg),
                    )
                    model_artifact.add_file(ckpt_path)

                    alias = [f"frame-{collector._frames}", "latest"]
                    if caught_rate <= best_caught_rate:
                        best_caught_rate = caught_rate
                        alias.append("best")

                    run.log_artifact(model_artifact, aliases=alias)

                except AttributeError:
                    logging.warning(f"Policy {policy} does not implement `.state_dict()`")

            elif save_interval > 0 and caught_rate <= best_caught_rate:
                best_caught_rate = caught_rate
                try:
                    ckpt_path = os.path.join(run.dir, "checkpoint_best.pt")
                    torch.save(policy.state_dict(), ckpt_path)
                    logging.info(f"Saved best checkpoint to {str(ckpt_path)}")

                    model_artifact = wandb.Artifact(
                        model_name,
                        type="model",
                        description=f"Best caught rate at {collector._frames} frames",
                        metadata=dict(cfg),
                    )
                    model_artifact.add_file(ckpt_path)
                    run.log_artifact(model_artifact, aliases=["best"])

                except AttributeError:
                    logging.warning(f"Policy {policy} does not implement `.state_dict()`")

        run.log(info)
        print(OmegaConf.to_yaml({k: v for k, v in info.items() if isinstance(v, float)}))

        pbar.set_postfix({"rollout_fps": collector._fps, "frames": collector._frames})

        if (max_iters > 0 and i >= max_iters - 1) or flag_early_stop:
            break

    logging.info(f"Final Eval at {collector._frames} steps.")
    info = {"env_frames": collector._frames}
    render_flag = render_interval > 0
    info.update(evaluate(render=render_flag))
    run.log(info)

    try:
        ckpt_path = os.path.join(run.dir, "checkpoint_final.pt")
        torch.save(policy.state_dict(), ckpt_path)

        model_artifact = wandb.Artifact(
            model_name,
            type="model",
            description=f"{cfg.task.name}-{cfg.algo.name.lower()}",
            metadata=dict(cfg))

        model_artifact.add_file(ckpt_path)
        wandb.save(ckpt_path)
        run.log_artifact(model_artifact, aliases=["latest"])

        logging.info(f"Saved checkpoint to {str(ckpt_path)}")
    except AttributeError:
        logging.warning(f"Policy {policy} does not implement `.state_dict()`")

    wandb.finish()

    simulation_app.close()


if __name__ == "__main__":
    main()
