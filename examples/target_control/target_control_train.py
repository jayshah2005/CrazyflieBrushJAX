from datetime import datetime
import functools
import os
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_triton_gemm_any=True '
    '--xla_gpu_enable_latency_hiding_scheduler=true '
)

from utils.reward_utils import get_reward_function

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

from environment.quadcopter import Quadcopter, domain_randomize

from utils.execution_guard import execution_guard


class QuadcopterTargetControl(Quadcopter):

    def __init__(self, cfg, config=None, custom_omega_to_thrust=None, custom_omega_to_torque=None, **kwargs):
        super().__init__(cfg, config, custom_omega_to_thrust=custom_omega_to_thrust, custom_omega_to_torque=custom_omega_to_torque, **kwargs)

        # init reward functions
        self.__pos_reward_function = get_reward_function(cfg.reward.pos_reward_shape)
        self.__pos_reward_function3 = get_reward_function(cfg.reward.pos_reward_shape3)
        self.__i_term_reward_function = get_reward_function(cfg.reward.i_term_reward_shape)
        self.__action_reward_function = get_reward_function(cfg.reward.action_reward_shape)
        self.__vel_reward_function = get_reward_function(cfg.reward.vel_reward_shape)
        self.__ang_rate_reward_function = get_reward_function(cfg.reward.ang_rate_reward_shape)
        self.__quat_reward_function = get_reward_function(cfg.reward.quat_reward_shape)
        self.__yaw_reward_function = get_reward_function(cfg.reward.yaw_reward_shape)
        self.__pos_ang_rate_reward_function = get_reward_function(cfg.reward.pos_ang_rate_reward_shape)
        self.__torque_reward_function = get_reward_function(cfg.reward.torque_reward_shape)
        self.__action_diff_reward_function = get_reward_function(cfg.reward.action_diff_reward_shape)
        self.__rolling_reward_function = get_reward_function(cfg.reward.rolling_reward_shape)

    def get_reward(self, data, thrust: jnp.ndarray, state_info, metrics):
        parsed_state = self.parse_state(data)
        pos = parsed_state["pos"]
        vel = parsed_state["vel"]
        quat = parsed_state["quat"]
        ang_rate = parsed_state["ang_rate"]
        reward = 0

        reward_parts = {}

        # position reward
        p = jnp.linalg.norm(pos - state_info["target_state"][0:3])
        reward_parts["pos"] =  self.__pos_reward_function(
            p, 
            self.cfg.reward.max_pos_diff) * self.cfg.reward.pos_coef
        reward_parts["pos"] += self.__pos_reward_function(
            p, 
            self.cfg.reward.max_pos_diff2) * self.cfg.reward.pos_coef2
        reward_parts["pos"] += self.__pos_reward_function3(
            p, 
            self.cfg.reward.max_pos_diff3) * self.cfg.reward.pos_coef3
        
        # action reward
        reward_parts["thrust"] = jnp.sum(self.__action_reward_function(
                jnp.abs(thrust[0:4] - self.hover_command),
                self.cfg.reward.max_action_diff)) * self.cfg.reward.action_coef
        
        # velocity reward
        reward_parts["vel"] = self.__vel_reward_function(
            jnp.linalg.norm(vel),
            self.cfg.reward.max_vel_diff) * self.cfg.reward.vel_coef
        reward_parts["vel"] += self.__vel_reward_function(
            jnp.abs(vel[2]),
            self.cfg.reward.max_vertical_vel_diff) * self.cfg.reward.vertical_vel_coef


        # quaternion reward
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        rot_matrix = jnp.array([
            [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w),   2 * (x * z + y * w)],
            [2 * (x * y + z * w),   1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w),   2 * (y * z + x * w),   1 - 2 * (x**2 + y**2)]
        ])

        rot_matrix = rot_matrix - jnp.eye(3)
        reward_parts["quat"] = self.__quat_reward_function(
            jnp.linalg.norm(rot_matrix.flatten()),
            self.cfg.reward.max_quat_diff) * self.cfg.reward.quat_coef
        
        # yaw reward
        reward_parts["yaw"] = self.__yaw_reward_function(
            jnp.abs(jnp.arctan2(rot_matrix[1, 0], rot_matrix[0, 0] + 1e-7 + 1.0)),
            self.cfg.reward.max_yaw_diff) * self.cfg.reward.yaw_coef

        # ang_rate reward
        reward_parts["ang_rate"] = (self.__ang_rate_reward_function(
            jnp.linalg.norm(ang_rate),
            self.cfg.reward.max_ang_rate_diff)) * self.cfg.reward.ang_rate_coef
        
        # torque_reward
        torques = jnp.array([[1, 1, -1, -1],
                             [1, -1, -1, 1],
                             [1, -1, 1, -1]]) @ thrust[0:4]
        reward_parts["torque"] = jnp.sum(self.__torque_reward_function(jnp.abs(torques), self.cfg.reward.torque_max_diff)) * self.cfg.reward.torque_coef

        # action_diff reward
        action_diff = jnp.linalg.norm(thrust[0:4] / self.hover_command - state_info["old_action"] / self.hover_command)
        reward_parts["action_diff"] = self.__action_diff_reward_function(
            action_diff, 
            self.cfg.reward.max_action_diff_diff) * self.cfg.reward.action_diff_coef
        

        # pos_ang_rate reward
        pos_ang_rate = jnp.linalg.norm(pos - state_info["target_state"][0:3]) + jnp.linalg.norm(ang_rate)
        reward_parts["pos_ang_rate"] = self.__pos_ang_rate_reward_function(
            pos_ang_rate,
            self.cfg.reward.max_pos_ang_rate_diff) * self.cfg.reward.pos_ang_rate_coef

        # rolling reward
        reward_parts["rolling"] = self.__rolling_reward_function(
            jnp.abs(ang_rate[1] - self.cfg.reward.rolling_speed),
            self.cfg.reward.max_rolling_diff) * self.cfg.reward.rolling_coef
        
        reward_parts["rolling"] -= self.__rolling_reward_function(
            jnp.abs(ang_rate[1] + self.cfg.reward.rolling_speed),
            self.cfg.reward.max_rolling_diff) * self.cfg.reward.rolling_coef
        
        reward_parts["rolling"] += (self.__ang_rate_reward_function(
            jnp.linalg.norm(jnp.array([ang_rate[0], ang_rate[2]])),
            self.cfg.reward.max_ang_rate_diff)) * self.cfg.reward.rolling_coef * 0.1
        
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
            action_reward=reward_parts["thrust"],
            quat_reward=reward_parts["quat"],
            yaw_reward=reward_parts["yaw"],
            ang_rate_reward=reward_parts["ang_rate"],
            torque=reward_parts["torque"],
            action_diff=reward_parts["action_diff"],
        )

        return reward, jnp.astype(done, jnp.float32), metrics

    def init_metrics(self):
        return {
            "pos_reward": 0.0,
            "vel_reward": 0.0,
            "action_reward": 0.0,
            "quat_reward": 0.0,
            "yaw_reward": 0.0,
            "ang_rate_reward": 0.0,
            "torque": 0.0,
            "action_diff": 0.0,
        }
    
    def reset_aux_state(self, data, rng: jnp.ndarray) -> dict:
        # sample target position
        low_pos, hi_pos = jnp.array([self.cfg.system.reset_pos_x_range[0], self.cfg.system.reset_pos_y_range[0], self.cfg.system.reset_pos_z_range[0]]), jnp.array([self.cfg.system.reset_pos_x_range[1], self.cfg.system.reset_pos_y_range[1], self.cfg.system.reset_pos_z_range[1]])
        if self.cfg.reward.train_simple_controller:
            target_state = jnp.array([0.0, 0.0, 1.0])
        else:
            target_state = jax.random.uniform(rng, (3,), minval=low_pos, maxval=hi_pos)

        return {
            "target_state": target_state,
            "i_target_state": target_state,
            "old_action": jnp.zeros(4),
            "i_term": jnp.zeros(3),
        }
    
    def get_aux_observations(self, aux_state: dict) -> jnp.ndarray:
        if self.cfg.reward.use_i_term:
            return jnp.concatenate([aux_state["target_state"], aux_state["i_term"]], axis=0)
        else:
            return aux_state["target_state"]
    
    def step_aux_state(self, data, action, aux_states, current_time, rng: jnp.ndarray) -> dict:
        aux_states["old_action"] = action[0:4]
        if self.cfg.reward.use_i_term:
            pos = self.parse_state(data)["pos"]
            error = aux_states["i_target_state"] - pos
            aux_states["i_term"] = aux_states["i_term"] + error * self.cfg.system.simulation_dt * self.cfg.reward.i_scaling
            aux_states["i_term"] = jnp.clip(aux_states["i_term"], -1.0, 1.0)
        return aux_states
        
    def get_ang_rate_reward(self, ang_rate):
        reward = (self.__ang_rate_reward_function(
            jnp.linalg.norm(ang_rate),
            self.cfg.reward.max_ang_rate_diff)) * self.cfg.reward.ang_rate_coef
        return reward

    def __done(self, pos, quat, vel, ang_rate):
        failed = pos[0] < -self.cfg.reward.max_pos_fail_x
        failed |= pos[0] > self.cfg.reward.max_pos_fail_x
        failed |= pos[1] < -self.cfg.reward.max_pos_fail_y
        failed |= pos[1] > self.cfg.reward.max_pos_fail_y
        failed |= pos[2] > self.cfg.reward.max_pos_fail_z
        failed |= pos[2] < -self.cfg.reward.max_pos_fail_z
        failed |= pos[2] < self.cfg.reward.min_pos_fail_z
        
        # print(f"np.linalg.norm(pos): {np.linalg.norm(pos)}")
        failed |= jnp.linalg.norm(vel) > self.cfg.reward.max_vel_fail
        # print(f" np.linalg.norm(vel): { np.linalg.norm(vel)}")

        failed |= jnp.linalg.norm(ang_rate) > self.cfg.reward.max_ang_rate_fail

        # failed |= pos[2] < -2.0  # 0.1

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
        num_timesteps=400_000_000,  #600_000_000, 
        num_resets_per_eval=cfg.num_resets_per_eval,
        num_evals=10, 
        reward_scaling=0.1, 
        episode_length=2000, 
        normalize_observations=True, 
        action_repeat=1, 
        unroll_length=100, 
        num_minibatches=24 * 2, 
        num_updates_per_batch=8,
        discounting=0.99, 
        learning_rate=3e-4, 
        entropy_cost=1e-4, 
        num_envs=cfg.num_systems,
        batch_size=512, 
        network_factory=functools.partial(ppo_networks.make_ppo_networks),
        seed=0)

    env = QuadcopterTargetControl(cfg)
    eval_env = QuadcopterTargetControl(cfg)

    times = [datetime.now()]

    # Called by brax after every evaluation (num_evals times during training).
    def progress(num_steps, metrics):
        times.append(datetime.now())
        print(f"steps={num_steps}  eval/episode_reward={float(metrics['eval/episode_reward']):.2f}")

        if wandb is not None:  # only set when cfg.wandb_log is true
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