from pathlib import Path
import mujoco
import mujoco.viewer as mujoco_viewer
from mujoco.renderer import Renderer
import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import hydra
import omegaconf
import random
import jax
import jax.numpy as jnp

from examples.target_control.target_control_train import QuadcopterTargetControl
import cv2

from brax.training.agents.ppo import networks as ppo_networks
from brax.io import model
from brax.training.acme import running_statistics

from orbax import checkpoint as ocp

import matplotlib.pyplot as plt

import mediapy as media
import pandas as pd


@hydra.main(config_path="../../parameters", config_name="main", version_base="1.1")
def main(cfg: omegaconf.DictConfig):
    delta_t = 0.01 / 2.5
    slowdown_factor = 1.0
    jax.config.update('jax_platform_name', 'cpu')
    media.set_show_save_dir("/data/RIG")
    
    random.seed(cfg.seed*1000)
    np.random.seed(cfg.seed*1000)
    key = jax.random.PRNGKey(cfg.seed*1000)

    cfg.system.velocity_noise_scale = 0.01 * 0
    cfg.system.angular_rate_noise_scale = 0.01 * 0
    cfg.system.measurement_noise_scale = 0.01 * 0

    cfg.system.reset_pose_error_angle = 0* 0.26
    cfg.system.reset_pose_error_pos = 0.0
    cfg.system.reset_motor_model_scale = 0.0
    cfg.system.reset_motor_diff_scale = 0.0

    # create agent
    normalize = running_statistics.normalize
    ppo_network = ppo_networks.make_ppo_networks(
      (3+4+3+3 + 6) * 6 + 3 + 1, 8, preprocess_observations_fn=normalize, activation=jax.nn.silu
    )
    make_policy = ppo_networks.make_inference_fn(ppo_network)
    
    if False:
        params = model.load_params("/data/RIG/brax_save")
    else:
        files = [f for f in Path("/data/RIG/target_control_dr7/starting/23/checkpoints").iterdir() if f.is_file() and f.stem.isdigit()]
        if not files:
            raise FileNotFoundError("No file with a numeric name found in /data/RIG")
        valid_files = [f for f in files if int(f.stem) <= 450000000_000000]
        if not valid_files:
            raise FileNotFoundError("No file with a numeric name below 100 million found in /data/RIG")
        largest_file = max(valid_files, key=lambda f: int(f.stem))
        print("Loading params from:", largest_file)
        params = model.load_params(largest_file)
    
    jit_inference_fn = jax.jit(make_policy(params))

    # create environment
    config = {'mass_body1': np.float64(0.0455), 'inertia_x': np.float64(3.5e-05), 'inertia_y': np.float64(4.0e-05), 'inertia_z': np.float64(7.19e-5 * 0.82), 'mass_offsetx': np.float64(-0.001), 'mass_offsety': np.float64(-0.00021975373810106426 * 0), 'mass_offsetz': np.float64(-0.0045306701671183535 * 0), 'max_torque_multiplier1': np.float64(1.0), 'max_torque_multiplier2': np.float64(1.0), 'max_torque_multiplier3': np.float64(1.0), 'max_torque_multiplier4': np.float64(1.0), 'thrust_multiplier1': np.float64(1.0), 'thrust_multiplier2': np.float64(1.0), 'thrust_multiplier3': np.float64(1.0), 'thrust_multiplier4': np.float64(1.0), 'quadcopter_dimension': np.float64(0.0353 * 2), 'pwm_gamma': np.float64(0.0), 'pwm_gamma1': np.float64(0.5289172856266848), 'rotor_delay': np.float64(0.020447333042501376), 'position_body1': [-0.0010756468684873773, 0.007610250467330411, 11.013593944701656], 'quat_body1': [0.998843153797347, 0.006028605542538987, 0.015263499542348107, -0.04517776945862793], 'max_thrust1': 100000, 'mass_rotor1': 1e-05, "motor_a": 1 / 0.05, "motor_b": 2.9e3, "J_m": 0.5e-07 * 1e3}

    eval_env = QuadcopterTargetControl(cfg, config=config)
    mj_model = eval_env.mj_model
    mj_data = mujoco.MjData(mj_model)

    print(mj_model.body_inertia)

    positions = np.array([[0.0, 0.0, 1.1],
                          [0.0, 0.0, 1.0],
                          [3.0, 0.0, 1.0],
                        [-3.0, 0.0, 1.0],
                        [3.0, 0.0, 1.0],
                        #   [1.0, 0.0, 1.0],
                        #   [1.0, 0.0, 1.0],
                        # [3.0, 0.0, 1.0],
                        # [-3.0, 0.0, 1.0],
                        # [3.0, 0.0, 1.0],
                        #   [2.5, 1.2, 0.8],
                        #   [-2.5, -1.2, 1.8],
                        #   [-2.5, 1.2, 1.8],
                        #   [2.5, -1.2, 1.8],
                        #   [0.0, 0.0, 1.0]
                        ])
    
    time_per_segment = 4.0
    

    jit_reset = jax.jit(eval_env.reset)
    jit_step = jax.jit(eval_env.step)

    for i in range(100):
        # initialize the state
        rng = jax.random.PRNGKey(100 + i)
        state = jit_reset(rng)
        rollout = [state.pipeline_state]

        qpos = state.pipeline_state.qpos.at[0:3].set(positions[0])
        qpos = qpos.at[3:7].set(jnp.array([1.0, 0.0, 0.0, 0.0]))
        qvel = state.pipeline_state.qvel.at[0:6].set(jnp.zeros(6))

        state = state.replace(pipeline_state=state.pipeline_state.replace(qpos=qpos, qvel=qvel))

        # grab a trajectory
        render_every = 2
        actions = []
        states = []
        rs =  []
        old_segment = 0
        for i in range(int(time_per_segment / 0.004 * len(positions))):
            segment = int(i // (time_per_segment / 0.004))
            current_target = positions[segment, :]
            state.info["pos_error"] = (-current_target + jnp.array([0.0, 0.0, 1.0])) * 1
            state.info["target_state"] = jnp.array([0.0, 0.0, 1.0])
            state.info["i_target_state"] = current_target
            state.info["battery_level"] = state.info["battery_level"].at[0].set(1.0)
            
            act_rng, rng = jax.random.split(rng)
            obs = state.obs
            pos = obs[:3]
            quat = obs[6:10]
            ctrl, crtl_raw = jit_inference_fn(obs, act_rng)
            ctrl = jax.nn.tanh(crtl_raw["logits"][0:8])
            state = jit_step(state, ctrl) 
            rollout.append(state.pipeline_state)
            obs = state.obs.at[0:3].set(state.obs[0:3] - state.info["pos_error"])
            states.append(obs)
            
            pwm, action = eval_env.nn_action_to_physics_action(ctrl, state.info["battery_level"])
            actions.append(action * (2**16 - 1))
            rs.append(eval_env.get_ang_rate_reward(state.obs[10:13]))

        target_pos = state.info["target_state"]
        camera = mujoco.MjvCamera()
        camera.lookat = [0, 0, 1.0]
        camera.distance = 6.0
        camera.azimuth = 0.0
        camera.elevation = -20.0
        
        media.show_video(eval_env.render(rollout[::render_every], height=480, width=600, camera=camera), fps=1.0 / (slowdown_factor*delta_t) / render_every, title="Quad")

        # Convert collected lists to numpy arrays
        states_arr = np.array(states)  # shape: (n_steps, 13)
        actions_arr = np.array(actions)  # shape: (n_steps, 8)
        time_steps = np.arange(states_arr.shape[0])
        time_act = np.arange(actions_arr.shape[0])

        # Assuming each state observation is ordered as:
        # [pos0, pos1, pos2, vel0, vel1, vel2, quat_w, quat_x, quat_y, quat_z, ang_rate0, ang_rate1, ang_rate2]
        pos = states_arr[:, :3]
        vel = states_arr[:, 3:6]
        rot_mat = states_arr[:, 6:12]
        def extract_rot_mat(rot_mat_):
            rot_mat = rot_mat_.reshape((3, 2))
            rot_mat = jnp.concatenate((rot_mat, jnp.zeros((3, 1))), axis=1)
            rot_mat = rot_mat.at[:, 2].set(jnp.cross(rot_mat[:, 0], rot_mat[:, 1]))

            return rot_mat
        
        def rot_mat_to_euler(rot_mat):
            sy = jnp.sqrt(rot_mat[0, 0] * rot_mat[0, 0] + rot_mat[1, 0] * rot_mat[1, 0])

            x = jnp.arctan2(rot_mat[2, 1], rot_mat[2, 2])
            y = jnp.arctan2(-rot_mat[2, 0], sy)
            z = jnp.arctan2(rot_mat[1, 0], rot_mat[0, 0])

            return jnp.array([x, y, z])

        rot_mat = jax.vmap(extract_rot_mat)(jnp.array(rot_mat))
        euler_angles = np.array(jax.vmap(rot_mat_to_euler)(rot_mat))
        
        ang_rates = states_arr[:, 12:15]

        fig, axs = plt.subplots(5, 1, figsize=(10, 15))

        # First plot: Positions
        steps_per_segment = int(time_per_segment / 0.004)
        # Repeat each setpoint for the duration of its segment to form a target trajectory matching time_steps length
        target_traj = np.repeat(positions, steps_per_segment, axis=0)[:len(time_steps)]
        for i in range(pos.shape[1]):
            # Plot the position error (difference between actual and target)
            axs[0].plot(time_steps, pos[:, i], label=f'pos{i}')
            # Plot the setpoint (target trajectory) as dashed lines
            axs[0].plot(time_steps, target_traj[:, i], '--', label=f'setpoint pos{i}')
        axs[0].set_title('Position & Setpoints')
        axs[0].legend()
        axs[0].grid(True)

        # Second plot: Velocities
        for i in range(vel.shape[1]):
            axs[1].plot(time_steps, vel[:, i], label=f'vel{i}')
        axs[1].set_title('Velocity')
        axs[1].legend()
        axs[1].grid(True)

        # Third plot: Euler Angles (converted from quaternion)
        for i, ang in enumerate(['roll', 'pitch', 'yaw']):
            axs[2].plot(time_steps, np.degrees(euler_angles[:, i]), label=f'{ang} (degrees)')
        axs[2].set_title('Euler Angles (Degrees)')
        axs[2].legend()
        axs[2].grid(True)

        # Fourth plot: Angular Rates
        for i in range(ang_rates.shape[1]):
            axs[3].plot(time_steps, ang_rates[:, i], label=f'ang_rate{i}')
        axs[3].set_title('Angular Rates')
        axs[3].legend()
        axs[3].grid(True)

        # Fifth plot: Actions
        for i in range(actions_arr.shape[1]):
            axs[4].plot(time_act, actions_arr[:, i], label=f'action{i}')
        axs[4].set_title('Actions')
        axs[4].legend()
        axs[4].grid(True)
        axs[4].set_ylim(0, 2**16 - 1)

        plt.tight_layout()
        plot_path = Path(__file__).parent / "trajectory_plot.png"
        plt.savefig(plot_path)
        # print(f"Plot saved to {plot_path}")
        plt.show()

        df = pd.DataFrame({
            'time': time_steps[::5] * 0.004,
            'x': pos[::5, 0],
            'y': pos[::5, 1],
            'z': pos[::5, 2],
            'vx': vel[::5, 0],
            'vy': vel[::5, 1],
            'vz': vel[::5, 2],
            'gyro_x': ang_rates[::5, 0],
            'gyro_y': ang_rates[::5, 1],
            'gyro_z': ang_rates[::5, 2],
            'yaw': np.degrees(euler_angles[::5, 2]),
            'roll': np.degrees(euler_angles[::5, 0]),
            'pitch': np.degrees(euler_angles[::5, 1]),
            'pwm_m1': actions_arr[::5, 0],
            'pwm_m2': actions_arr[::5, 1],
            'pwm_m3': actions_arr[::5, 2],
            'pwm_m4': actions_arr[::5, 3],
        })
        # csv_path = "target_control_sim.csv"
        # df.to_csv(csv_path, index=False)
        # print(f"Data saved to {csv_path}")

    exit()
        
if __name__ == '__main__':
    main()
    
