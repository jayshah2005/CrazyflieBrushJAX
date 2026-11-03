"""Fly a trained target-control policy through a sequence of waypoints in simulation.

Usage:
    python -m examples.target_control.target_control_eval
    python -m examples.target_control.target_control_eval eval_params=$HOME/crazyflie_logs/training/checkpoints/420249600

Writes a video (quad_<n>.mp4), plots (trajectory_<n>.png) and a CSV of the
flight (trajectory_<n>.csv) to ${root_dir}/eval/<timestamp>/.
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")  # offscreen rendering, works without a display

from datetime import datetime
from pathlib import Path

import hydra
import imageio_ffmpeg
import jax
import jax.numpy as jnp
import mediapy as media
import mujoco
import numpy as np
import matplotlib
import omegaconf
from matplotlib import pyplot as plt

from brax.io import model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

from examples.target_control.target_control_train import QuadcopterTargetControl

# Run on CPU: the rollout is sequential, and the GPU may be busy training.
jax.config.update("jax_platforms", "cpu")
media.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())

# Nominal (system-identified) Crazyflie Brushless parameters from the authors.
NOMINAL_CONFIG = {'mass_body1': np.float64(0.0455), 'inertia_x': np.float64(3.5e-05), 'inertia_y': np.float64(4.0e-05), 'inertia_z': np.float64(7.19e-5 * 0.82), 'mass_offsetx': np.float64(-0.001), 'mass_offsety': np.float64(-0.00021975373810106426 * 0), 'mass_offsetz': np.float64(-0.0045306701671183535 * 0), 'max_torque_multiplier1': np.float64(1.0), 'max_torque_multiplier2': np.float64(1.0), 'max_torque_multiplier3': np.float64(1.0), 'max_torque_multiplier4': np.float64(1.0), 'thrust_multiplier1': np.float64(1.0), 'thrust_multiplier2': np.float64(1.0), 'thrust_multiplier3': np.float64(1.0), 'thrust_multiplier4': np.float64(1.0), 'quadcopter_dimension': np.float64(0.0353 * 2), 'pwm_gamma': np.float64(0.0), 'pwm_gamma1': np.float64(0.5289172856266848), 'rotor_delay': np.float64(0.020447333042501376), 'position_body1': [-0.0010756468684873773, 0.007610250467330411, 11.013593944701656], 'quat_body1': [0.998843153797347, 0.006028605542538987, 0.015263499542348107, -0.04517776945862793], 'max_thrust1': 100000, 'mass_rotor1': 1e-05, "motor_a": 1 / 0.05, "motor_b": 2.9e3, "J_m": 0.5e-07 * 1e3}

# Waypoints to fly to, one after another (x, y, z in metres).
WAYPOINTS = np.array([[0.0, 0.0, 1.1],
                      [0.0, 0.0, 1.0],
                      [3.0, 0.0, 1.0],
                      [-3.0, 0.0, 1.0],
                      [3.0, 0.0, 1.0]])


def load_policy(params_path: Path, env_obs_size: int, cfg):
    """Load saved policy params and rebuild a network whose shape is read from the params."""
    if not params_path.is_file():
        raise FileNotFoundError(f"No policy found at {params_path}. Set eval_params=<path>.")
    params = model.load_params(params_path)
    policy_params = params[1]  # params = (normalizer, policy, value)

    layers = policy_params["params"]
    kernels = [layers[f"hidden_{i}"]["kernel"] for i in range(len(layers))]
    obs_size = kernels[0].shape[0]
    hidden_sizes = tuple(k.shape[1] for k in kernels[:-1])
    action_size = kernels[-1].shape[1] // 2  # policy outputs a mean and a std per action

    modified = datetime.fromtimestamp(params_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    print(f"Loaded {params_path} (saved {modified}): {obs_size} inputs, hidden {hidden_sizes}, {action_size} actions")

    if obs_size != env_obs_size:
        raise ValueError(
            f"The policy expects {obs_size} inputs but the environment produces {env_obs_size}.\n"
            f"The policy was trained with a different observation config "
            f"(current: system.observe_omega={cfg.system.observe_omega}). Either evaluate a policy trained "
            f"with this config, or override the config to match, e.g. system.observe_omega="
            f"{str(not cfg.system.observe_omega).lower()}.\n"
            f"Note: a retrain writes into the same folder, so old and new checkpoints can be mixed until it finishes."
        )

    ppo_network = ppo_networks.make_ppo_networks(
        obs_size, action_size,
        preprocess_observations_fn=running_statistics.normalize,
        policy_hidden_layer_sizes=hidden_sizes,
        activation=jax.nn.silu,
    )
    make_policy = ppo_networks.make_inference_fn(ppo_network)
    return jax.jit(make_policy(params, deterministic=True)), action_size


def rot_mat_to_euler(rot_mat_cols):
    """Roll/pitch/yaw from the first two columns of a rotation matrix (as observed by the policy)."""
    r = rot_mat_cols.reshape((3, 2))
    r = jnp.concatenate([r, jnp.cross(r[:, 0], r[:, 1])[:, None]], axis=1)
    sy = jnp.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)
    return jnp.array([jnp.arctan2(r[2, 1], r[2, 2]), jnp.arctan2(-r[2, 0], sy), jnp.arctan2(r[1, 0], r[0, 0])])


def save_plots(path, dt, states, actions, waypoints, steps_per_segment, show):
    time = np.arange(states.shape[0]) * dt
    pos, vel, ang_rates = states[:, 0:3], states[:, 3:6], states[:, 12:15]
    euler = np.degrees(np.array(jax.vmap(rot_mat_to_euler)(jnp.array(states[:, 6:12]))))
    target = np.repeat(waypoints, steps_per_segment, axis=0)[:len(time)]

    fig, axs = plt.subplots(5, 1, figsize=(10, 15), sharex=True)
    for i, name in enumerate("xyz"):
        line, = axs[0].plot(time, pos[:, i], label=name)
        axs[0].plot(time, target[:, i], "--", color=line.get_color(), label=f"{name} target")
    axs[0].set_title("Position (m)")
    for i, name in enumerate("xyz"):
        axs[1].plot(time, vel[:, i], label=name)
    axs[1].set_title("Velocity (m/s)")
    for i, name in enumerate(["roll", "pitch", "yaw"]):
        axs[2].plot(time, euler[:, i], label=name)
    axs[2].set_title("Attitude (deg)")
    for i, name in enumerate("xyz"):
        axs[3].plot(time, ang_rates[:, i], label=name)
    axs[3].set_title("Angular rate (rad/s)")
    for i in range(actions.shape[1]):
        axs[4].plot(time, actions[:, i], label=f"a{i}")
    axs[4].set_title("Actions (PWM, 16-bit)")
    axs[4].set_ylim(0, 2**16 - 1)
    axs[4].set_xlabel("time (s)")
    for ax in axs:
        ax.legend(loc="upper right", fontsize="small", ncol=2)
        ax.grid(True)
    fig.tight_layout()
    fig.savefig(path)
    if show:
        plt.show()
    plt.close(fig)
    return euler


@hydra.main(config_path="../../parameters", config_name="eval", version_base="1.1")
def main(cfg: omegaconf.DictConfig):
    out_dir = Path.cwd()  # hydra changes into ${root_dir}/eval/<timestamp>
    if not cfg.eval_show_plots:
        matplotlib.use("Agg")  # draw plots to files only, no window
    dt = cfg.system.simulation_dt
    render_every = 2

    # Evaluate without sensor noise or randomized errors.
    cfg.system.velocity_noise_scale = 0.0
    cfg.system.angular_rate_noise_scale = 0.0
    cfg.system.measurement_noise_scale = 0.0
    cfg.system.reset_pose_error_angle = 0.0
    cfg.system.reset_pose_error_pos = 0.0
    cfg.system.reset_motor_model_scale = 0.0
    cfg.system.reset_motor_diff_scale = 0.0

    eval_env = QuadcopterTargetControl(cfg, config=NOMINAL_CONFIG)
    policy, _ = load_policy(Path(cfg.eval_params).expanduser(), eval_env.observation_size, cfg)

    jit_reset = jax.jit(eval_env.reset)
    jit_step = jax.jit(eval_env.step)

    steps_per_segment = int(cfg.eval_time_per_waypoint / dt)
    n_steps = steps_per_segment * len(WAYPOINTS)

    camera = mujoco.MjvCamera()
    camera.lookat = [0, 0, 1.0]
    camera.distance = 6.0
    camera.azimuth = 0.0
    camera.elevation = -20.0

    for episode in range(cfg.eval_episodes):
        rng = jax.random.PRNGKey(100 + episode)
        state = jit_reset(rng)

        # start hovering at the first waypoint, level and at rest
        ps = state.pipeline_state
        qpos = ps.qpos.at[0:3].set(WAYPOINTS[0]).at[3:7].set(jnp.array([1.0, 0.0, 0.0, 0.0]))
        qvel = ps.qvel.at[0:6].set(jnp.zeros(6))
        state = state.replace(pipeline_state=ps.replace(qpos=qpos, qvel=qvel))

        rollout = [state.pipeline_state]
        states, actions = [], []
        for step in range(n_steps):
            current_target = WAYPOINTS[step // steps_per_segment]
            # The policy was trained to fly to (0, 0, 1). To fly to another waypoint, shift the
            # position it observes so that the waypoint appears to be at (0, 0, 1).
            state.info["pos_error"] = jnp.array([0.0, 0.0, 1.0]) - current_target
            state.info["target_state"] = jnp.array([0.0, 0.0, 1.0])
            state.info["i_target_state"] = current_target
            state.info["battery_level"] = state.info["battery_level"].at[0].set(1.0)

            act_rng, rng = jax.random.split(rng)
            ctrl, _ = policy(state.obs, act_rng)  # deterministic: tanh of the policy mean
            state = jit_step(state, ctrl)
            rollout.append(state.pipeline_state)

            # undo the shift to log the true position
            states.append(np.array(state.obs.at[0:3].set(state.obs[0:3] - state.info["pos_error"])))
            _, action = eval_env.nn_action_to_physics_action(ctrl, state.info["battery_level"])
            actions.append(np.array(action) * (2**16 - 1))

        states, actions = np.array(states), np.array(actions)

        final_err = np.linalg.norm(states[-1, 0:3] - WAYPOINTS[-1])
        print(f"Episode {episode}: final distance to last waypoint {final_err:.3f} m, "
              f"min altitude {states[:, 2].min():.2f} m")

        video_path = out_dir / f"quad_{episode}.mp4"
        frames = eval_env.render(rollout[::render_every], height=480, width=640, camera=camera)
        media.write_video(video_path, frames, fps=1.0 / dt / render_every)

        plot_path = out_dir / f"trajectory_{episode}.png"
        euler = save_plots(plot_path, dt, states, actions, WAYPOINTS, steps_per_segment, cfg.eval_show_plots)

        # every 5th step (50 Hz), same columns as the authors' sim log, for comparison with real flight logs
        csv_path = out_dir / f"trajectory_{episode}.csv"
        k = slice(None, None, 5)
        table = np.column_stack([np.arange(len(states))[k] * dt, states[k, 0:6], states[k, 12:15],
                                 euler[k, 2], euler[k, 0], euler[k, 1], actions[k, 0:4]])
        np.savetxt(csv_path, table, delimiter=",", comments="", fmt="%.6g",
                   header="time,x,y,z,vx,vy,vz,gyro_x,gyro_y,gyro_z,yaw,roll,pitch,pwm_m1,pwm_m2,pwm_m3,pwm_m4")

    print(f"Saved video, plots and CSV to {out_dir}")


if __name__ == "__main__":
    main()
