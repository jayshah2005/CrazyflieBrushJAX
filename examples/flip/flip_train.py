from datetime import datetime
import functools
import os
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_triton_gemm_any=True '
    '--xla_gpu_enable_latency_hiding_scheduler=true '
)

from utils.reward_utils import get_reward_function, interval_constraint_reward
from pathlib import Path
from functools import partial
import numpy as np
import random

import jax
import jax.numpy as jnp

from brax.training.agents.ppo import train as ppo
from brax.training.agents.sac import train as sac
from brax.training.agents.sac.networks import SACNetworks
from brax.io import model

from typing import Sequence, Tuple

from brax.training import distribution
from brax.training import networks
from brax.training import types
from brax.training.types import PRNGKey
import flax
from flax import linen

from flax.training import orbax_utils
from orbax import checkpoint as ocp

import hydra
import omegaconf

import mediapy as media

from utils.utils import create_model_dir
from brax.training.agents.ppo import networks as ppo_networks

from environment.quadcopter import Quadcopter, domain_randomize, quat_to_rot_mat

from utils.execution_guard import execution_guard


class QuadcopterFlip(Quadcopter):

    def __init__(self, cfg, config=None, **kwargs):
        super().__init__(cfg, config, **kwargs)

        # init reward functions
        self.__vel_reward_function = get_reward_function(cfg.reward.vel_reward_shape)
        self.__quat_reward_function = get_reward_function(cfg.reward.quat_reward_shape)
        self.__pitch_reward_function = get_reward_function(cfg.reward.pitch_reward_shape)

    def init_metrics(self):
        return {
            "pos_reward": 0.0,
            "quat_reward": 0.0,
            "pitch_reward": 0.0,
            "vel_reward": 0.0,
        }
    
    def reset_aux_state(self, data, rng: jnp.ndarray) -> dict:
        # sample target position
        low_pos, hi_pos = jnp.array([self.cfg.system.reset_pos_x_range[0], self.cfg.system.reset_pos_y_range[0], self.cfg.system.reset_pos_z_range[0]]), jnp.array([self.cfg.system.reset_pos_x_range[1], self.cfg.system.reset_pos_y_range[1], self.cfg.system.reset_pos_z_range[1]])
        key1, key2, key3 = jax.random.split(rng, 3)
        target_state = jax.random.uniform(key1, (3,), minval=low_pos, maxval=hi_pos)

        return {
            "my_current_time": jnp.zeros((1,)),
            "target_pitch_speed": jax.random.uniform(key2, (1,), minval=-self.cfg.reward.rot_speed * 1.0 * 2 * jnp.pi, maxval=self.cfg.reward.rot_speed * 1.0 * 2 * jnp.pi),
            "target_vel": jax.random.uniform(key3, (1,), minval=0, maxval=self.cfg.reward.target_vel_max)
        }
    
    def get_aux_observations(self, aux_state: dict) -> jnp.ndarray:
        return jnp.concatenate([aux_state["target_pitch_speed"], aux_state["target_vel"]])
    
    def step_aux_state(self, data, action, aux_states, current_time, rng: jnp.ndarray) -> dict:
        aux_states["my_current_time"] += self.cfg.system.simulation_dt
        return aux_states

    def get_reward(self, data, thrust: jnp.ndarray, state_info, metrics):
        parsed_state = self.parse_state(data)
        pos = parsed_state["pos"]
        vel = parsed_state["vel"]
        quat = parsed_state["quat"]
        ang_rate = parsed_state["ang_rate"]
        reward = 0

        reward_parts = {}

        # position reward
        reward_parts["pos"] = interval_constraint_reward(pos[0], 
                                                    delta=self.cfg.reward.pos_x_range[1] - self.cfg.reward.pos_x_range[0], 
                                                    epsilon=self.cfg.reward.max_pos_diff) * self.cfg.reward.pos_coef
        reward_parts["pos"] += interval_constraint_reward(pos[1],
                                                    delta=self.cfg.reward.pos_y_range[1] - self.cfg.reward.pos_y_range[0],
                                                    epsilon=self.cfg.reward.max_pos_diff) * self.cfg.reward.pos_coef
        middle = (self.cfg.reward.pos_z_range[1] + self.cfg.reward.pos_z_range[0]) / 2
        reward_parts["pos"] += interval_constraint_reward(pos[2] - middle,
                                                    delta=self.cfg.reward.pos_z_range[1] - self.cfg.reward.pos_z_range[0],
                                                    epsilon=self.cfg.reward.max_pos_diff) * self.cfg.reward.pos_coef

        # quaternion reward
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        rot_matrix = jnp.array([
            [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w),   2 * (x * z + y * w)],
            [2 * (x * y + z * w),   1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w),   2 * (y * z + x * w),   1 - 2 * (x**2 + y**2)]
        ])

        reward_parts["quat"] = self.__quat_reward_function(
            jnp.linalg.norm(rot_matrix[:, 0] - jnp.array([0, 1, 0])),
            self.cfg.reward.max_quat_diff) * self.cfg.reward.quat_coef
        

        # velocity reward
        vel_fixed_frame = rot_matrix.T @ vel
        reward_parts["vel"] = self.__vel_reward_function(
            jnp.linalg.norm(vel_fixed_frame - jnp.array([0, 0, state_info["target_vel"][0]])), 
            self.cfg.reward.max_vel_diff) * self.cfg.reward.vel_coef
        reward_parts["vel"] += self.__vel_reward_function(
            jnp.linalg.norm(vel_fixed_frame - jnp.array([0, 0, state_info["target_vel"][0]])), 
            self.cfg.reward.max_vel_diff2) * self.cfg.reward.vel_coef2
        

        # pitch reward
        # make such that the integral over total_rot_time is 2*pi
        target_rot_speed = state_info["target_pitch_speed"][0] 
        e = jnp.linalg.norm(ang_rate - jnp.array([target_rot_speed, 0, 0]))
        reward_parts["pitch"] = self.__pitch_reward_function(
            e, 
            self.cfg.reward.max_pitch_diff) * self.cfg.reward.pitch_coef
        reward_parts["pitch"] += self.__pitch_reward_function(
            e, 
            self.cfg.reward.max_pitch_diff2) * self.cfg.reward.pitch_coef

        
        done = self.__done(pos, quat, vel, ang_rate)

        reward = 0
        for key, value in reward_parts.items():
            reward += value

        reward = jax.lax.cond(
            done,
            lambda x: x - self.cfg.reward.fail_penalty,
            lambda x: x,
            reward
        )
        
        metrics.update(
            pos_reward=reward_parts["pos"],
            vel_reward=reward_parts["vel"],
            quat_reward=reward_parts["quat"],
            pitch_reward=reward_parts["pitch"],
        )

        return reward, jnp.astype(done, jnp.float32), metrics

    def __done(self, pos, quat, vel, ang_rate):
        failed = jnp.linalg.norm(vel) > self.cfg.reward.max_vel_fail
        failed |= jnp.linalg.norm(ang_rate) > self.cfg.reward.max_ang_rate_fail
        failed |= pos[2] < -10.0

        return failed


@hydra.main(config_path="../../parameters", config_name="main", version_base="1.1")
@partial(execution_guard, force_overwrite=True)
def main(cfg: omegaconf.DictConfig):
    # set seeds
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    key = jax.random.PRNGKey(cfg.seed)

    # create log directory
    create_model_dir()

    # WANDB INIT
    wandb = None

    # create weights and biases logger
    if cfg.wandb_log:
        import wandb

        wnb_cfg = omegaconf.OmegaConf.to_container(cfg, resolve=True)
        wandb.init(
            # set the wandb project where this run will be logged
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            # track hyperparameters and run metadata
            config=wnb_cfg,
            job_type="train",
            group="mjx",
            name=cfg.experiment_name_wandb,
        )
        wandb.define_metric("env_steps")
        wandb.define_metric("eval/episode_reward", step_metric="env_steps")
        wandb.define_metric("eval/episode_pos_reward", step_metric="env_steps")

    train_fn = functools.partial(
        ppo.train, 
        num_timesteps=300_000_000, 
        num_evals=10, 
        reward_scaling=0.1, 
        episode_length=int(1 / cfg.reward.rot_speed * 250) * 2 + int(cfg.reward.warmup_time * 250), 
        normalize_observations=True, 
        action_repeat=1, 
        unroll_length=100, 
        num_minibatches=24 * 2, 
        num_updates_per_batch=8,
        discounting=0.99, 
        learning_rate=3e-4, 
        entropy_cost=1e-3, 
        num_envs=cfg.num_systems,
        batch_size=512, 
        network_factory=functools.partial(ppo_networks.make_ppo_networks),
        seed=0)

    env = QuadcopterFlip(cfg)
    eval_env = QuadcopterFlip(cfg)

    times = [datetime.now()]

    def progress(num_steps, metrics):
        print(num_steps)
        times.append(datetime.now())

        logs = {k: v for k, v in metrics.items()}
        logs["env_steps"] = num_steps

        wandb.log(logs)

    def policy_params_fn(current_step, make_policy, params):
        Path("checkpoints").mkdir(parents=True, exist_ok=True)
        model.save_params(f"checkpoints/{current_step}", params)

    make_inference_fn, params, _= train_fn(environment=env, 
                                           eval_env=eval_env, 
                                           progress_fn=progress, 
                                           policy_params_fn=policy_params_fn, 
                                           randomization_fn=env.domain_randomize)

    model.save_params("brax_save", params)


if __name__=="__main__":
    main()