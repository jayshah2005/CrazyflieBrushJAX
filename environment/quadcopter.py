from functools import partial
import os
from pathlib import Path
import mujoco
import numpy as np
from jinja2 import Environment, FileSystemLoader

import jax
import jax.numpy as jnp

import mujoco
from mujoco import mjx

from brax.envs.base import PipelineEnv, State
from brax.io import mjcf

from abc import ABC, abstractmethod


def get_min_max(amplitude):
    """Returns the min and max values for the randomization."""
    ampli_max = 1 + amplitude
    ampli_min = (1 - amplitude)
    return ampli_min, ampli_max

def pwm_to_thrust(pwm):
    """
    Converts a PWM signal to thrust using a cubic polynomial fit based on empirical data.

    Parameters:
        pwm (jnp.ndarray): A JAX numpy array representing the PWM signal, expected to be in the range [0, 1].

    """
    return -0.23009526 * (pwm)**3 + 0.56176458 * (pwm)**2 -0.0433191*(pwm)

def pwm_to_torques(pwm):
    """
    Converts a PWM signal to torque using a cubic polynomial fit based on empirical data.

    Parameters:
        pwm (jnp.ndarray): A JAX numpy array representing the PWM signal, expected to be in the range [0, 1].
    """
    return  -0.0003396 * pwm**3 +  0.00087032 * pwm**2  +  0.0002896 *pwm

def omega_to_force(omega):
    """
    Converts an angular velocity (omega) to thrust

    Parameters:
        omega (jnp.ndarray): A JAX numpy array representing the angular velocity in rad/s.
    """
    pwm = omega / 2.9e3
    return pwm_to_thrust(pwm)

def omega_to_torque(omega):
    """
    Converts an angular velocity (omega) to torque

    Parameters:
        omega (jnp.ndarray): A JAX numpy array representing the angular velocity in rad/s.

    """
    pwm = omega / 2.9e3
    return pwm_to_torques(pwm)

# domain randomization
def domain_randomize(sys, rng, domain_randomization_amplitude, config):
    """
    Domain randomization of the mjx model (this is needed for BRAX)
    """
    def rand(rng, sys_):
        # mass
        key, key2 = jax.random.split(rng, 2)
        ampli_min, ampli_max = get_min_max(domain_randomization_amplitude["mass_body1"])
        mass_body = jax.random.uniform(key2, (1, ), minval=ampli_min*config["mass_body1"], maxval=ampli_max*config["mass_body1"])
        mass = sys_.body_mass.at[1].set(mass_body[0])

        # point of mass
        key, key2 = jax.random.split(rng, 2)
        mass_offset = jax.random.uniform(key2, (3, ), minval=-domain_randomization_amplitude["mass_offset"], maxval=domain_randomization_amplitude["mass_offset"])

        quadcopter_dimension = config["quadcopter_dimension"]
        site_pos = sys_.site_pos.at[0, :].set(jnp.array([quadcopter_dimension / 2 + mass_offset[0], -quadcopter_dimension / 2 + mass_offset[1], 0.01])) 
        site_pos = site_pos.at[1, :].set(jnp.array([-quadcopter_dimension / 2 + mass_offset[0], -quadcopter_dimension / 2 + mass_offset[1], 0.01]))
        site_pos = site_pos.at[2, :].set(jnp.array([-quadcopter_dimension / 2 + mass_offset[0], quadcopter_dimension / 2 + mass_offset[1], 0.01]))
        site_pos = site_pos.at[3, :].set(jnp.array([quadcopter_dimension / 2 + mass_offset[0], quadcopter_dimension / 2 + mass_offset[1], 0.01]))

        # inertia
        keys = jax.random.split(key, 4)
        key = keys[0]
        keys = keys[1:] 
        ampli_min, ampli_max = get_min_max(domain_randomization_amplitude["inertia"])
        inertia_x = jax.random.uniform(keys[0], (1, ), minval=ampli_min*config["inertia_x"], maxval=ampli_max*config["inertia_x"])
        inertia_y = jax.random.uniform(keys[1], (1, ), minval=ampli_min*config["inertia_y"], maxval=ampli_max*config["inertia_y"])
        inertia_z = jax.random.uniform(keys[2], (1, ), minval=ampli_min*config["inertia_z"], maxval=ampli_max*config["inertia_z"])

        inertia_z = jax.lax.cond(
            inertia_x[0] + inertia_y[0] >= inertia_z[0], lambda x: x[2], lambda x: (x[0] + x[1]) * 0.9999, (inertia_x, inertia_y, inertia_z)
            )

        inertia = sys_.body_inertia.at[3+0].set(inertia_x[0])
        inertia = inertia.at[3+1].set(inertia_y[0])
        inertia = inertia.at[3+2].set(inertia_z[0])
        
        # actuator
        keys = jax.random.split(key, 9)
        key = keys[0]
        keys = keys[1:] 
        ampli_min, ampli_max = get_min_max(domain_randomization_amplitude["thrust"])
        motor0_1_thrust = jax.random.uniform(keys[0], (1,), minval=ampli_min*config["thrust_multiplier1"], maxval=ampli_max*config["thrust_multiplier1"])
        motor0_2_thrust = jax.random.uniform(keys[1], (1,), minval=ampli_min*config["thrust_multiplier2"], maxval=ampli_max*config["thrust_multiplier2"])
        motor0_3_thrust = jax.random.uniform(keys[2], (1,), minval=ampli_min*config["thrust_multiplier3"], maxval=ampli_max*config["thrust_multiplier3"])
        motor0_4_thrust = jax.random.uniform(keys[3], (1,), minval=ampli_min*config["thrust_multiplier4"], maxval=ampli_max*config["thrust_multiplier4"])

        ampli_min, ampli_max = get_min_max(domain_randomization_amplitude["torque"])
        motor01_torque = jax.random.uniform(keys[4], (1,), minval=-ampli_max, maxval=-ampli_min)
        motor02_torque = jax.random.uniform(keys[5], (1,), minval=ampli_min, maxval=ampli_max)
        motor03_torque = jax.random.uniform(keys[6], (1,), minval=-ampli_max, maxval=-ampli_min)
        motor04_torque = jax.random.uniform(keys[7], (1,), minval=ampli_min, maxval=ampli_max)

        gear = sys_.actuator_gear.at[0, 2].set(motor0_1_thrust[0])
        gear = gear.at[1, 2].set(motor0_2_thrust[0])
        gear = gear.at[2, 2].set(motor0_3_thrust[0])
        gear = gear.at[3, 2].set(motor0_4_thrust[0])
        
        gear = gear.at[4, 5].set(motor01_torque[0])
        gear = gear.at[5, 5].set(motor02_torque[0])
        gear = gear.at[6, 5].set(motor03_torque[0])
        gear = gear.at[7, 5].set(motor04_torque[0])

        return mass, site_pos, inertia, gear

    mass_, site_pos_, inertia_, gear_ = jax.vmap(rand, in_axes=(0, None))(rng, sys)

    in_axes = jax.tree_util.tree_map(lambda x: None, sys)
    in_axes = in_axes.tree_replace({
        'body_mass': 0,
        'site_pos': 0,
        'body_inertia': 0,
        'actuator_gear': 0,
    })

    sys = sys.tree_replace({
        'body_mass': mass_,
        'site_pos': site_pos_,
        'body_inertia': inertia_,
        'actuator_gear': gear_,
    })

    return sys, in_axes


class Quadcopter(PipelineEnv, ABC):

    def __init__(
        self,
        cfg,  # hydra cfg
        config=None,  # cfg from sys id
        auxilliary_body = "",
        auxilliary_equality_constraints = "",
        custom_omega_to_thrust=None,
        custom_omega_to_torque=None,
        **kwargs,
    ):
        """
        Initializes a quadcopter simulation environment instance using provided configuration settings.
        Parameters:
            cfg (object): A Hydra configuration object containing environment-specific settings.
            config (dict, optional): A dictionary with system identification parameters including mass, inertia, motor parameters, and more.
                If None, a default configuration with pre-defined values is used.
            auxilliary_body (str, optional): An auxiliary body configuration setting; defaults to an empty string.
            auxilliary_equality_constraints (str, optional): An auxiliary equality constraints configuration setting; defaults to an empty string.
            custom_omega_to_thrust (callable, optional): A custom function to convert angular velocity (omega) to thrust. If None, a default conversion function is used.
            custom_omega_to_torque (callable, optional): A custom function to convert angular velocity (omega) to torque. If None, a default conversion function is used.
        """
        if config is None:
            config =   {'mass_body1': np.float64(0.0445), 'inertia_x': np.float64(3.3e-05), 'inertia_y': np.float64(3.6e-05), 'inertia_z': np.float64(7.19e-5), 'mass_offsetx': np.float64(0), 'mass_offsety': np.float64(0), 'mass_offsetz': np.float64(0), 'max_torque_multiplier1': np.float64(1.0), 'max_torque_multiplier2': np.float64(1.0), 'max_torque_multiplier3': np.float64(1.0), 'max_torque_multiplier4': np.float64(1.0), 'thrust_multiplier1': np.float64(1.0), 'thrust_multiplier2': np.float64(1.0), 'thrust_multiplier3': np.float64(1.0), 'thrust_multiplier4': np.float64(1.0), 'quadcopter_dimension': np.float64(0.0353 * 2), 'pwm_gamma': np.float64(0.0), 'pwm_gamma1': np.float64(0.5289172856266848), 'rotor_delay': np.float64(0.020447333042501376), 'position_body1': [-0.0010756468684873773, 0.007610250467330411, 11.013593944701656], 'quat_body1': [0.998843153797347, 0.006028605542538987, 0.015263499542348107, -0.04517776945862793], 'max_thrust1': 100000, 'mass_rotor1': 1e-05, "motor_a": 1 / 0.05, "motor_b": 2.9e3, "J_m": 0.5e-07 * 1e3}
    
        ############################
        # generate xml file
        ############################
        config["auxilliary_body"] = auxilliary_body
        config["auxilliary_equality_constraints"] = auxilliary_equality_constraints

        jinja_environment = Environment(loader=FileSystemLoader(Path(os.path.dirname(__file__))))
        template_xml_file = jinja_environment.get_template(f'quadcopter.xml.jinja')

        config['scene_path'] = Path(os.path.dirname(__file__))

        output = template_xml_file.render(config)

        print(output)

        mj_model = mujoco.MjModel.from_xml_string(output)
        mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        mj_model.opt.iterations = 6
        mj_model.opt.ls_iterations = 6
        self.mj_model = mj_model

        sys = mjcf.load_model(mj_model)

        # overwrite the control steps, as we do this manually.
        physics_steps_per_control_step = 1
        kwargs['n_frames'] = kwargs.get(
            'n_frames', physics_steps_per_control_step)
        kwargs['backend'] = 'mjx'

        super().__init__(sys, **kwargs)

        self.config = config
        self.cfg = cfg

        # this is needed to extract the auxiliary state from the state info.
        self.__own_state_info_keys = ["rng", "action_delays", "action_delays_pwm", "action_delay", "action_delay_1", "action_delay_pwm", "action_1", "action_filtered", "state_delays", "battery_level", "action_delays_obs", "disturbance_freq", "disturbance_ampl", "current_time", "pose_error", "pos_error" "motor_ac", "motor_bc", "motor_a", "motor_b", "J_m", "Omega", "pose_delays"]

        domain_randomization_amplitude = {"mass_body1": self.cfg.system.dr_mass_body, 
                                          "inertia": self.cfg.system.dr_inertia, 
                                          "thrust": self.cfg.system.dr_thrust, 
                                          "torque": self.cfg.system.dr_torque,
                                          "mass_offset": self.cfg.system.dr_mass_offset}
        domain_randomization_amplitude = jax.tree_util.tree_map(lambda x: x * self.cfg.system.dr_scaling, domain_randomization_amplitude)
        self.domain_randomize = partial(domain_randomize, domain_randomization_amplitude=domain_randomization_amplitude, config=config)

        # calculate hovering thrust. (might come handy for rewards)
        thrust_per_motor = config["mass_body1"] * 9.81 / 4.0
        tol = 1e-6
        low = 0.0
        high = 2.9e3
        for _ in range(50):
            mid = (low + high) / 2.0
            diff = omega_to_force(mid) - thrust_per_motor
            if abs(diff) < tol:
                break
            if diff > 0:
                high = mid
            else:
                low = mid
        hover_omega = mid
        self.hover_command = hover_omega / 2.9e3

        # allow users to adjust these
        if custom_omega_to_thrust is not None:
            self.omega_to_thrust = custom_omega_to_thrust
        else:
            self.omega_to_thrust = omega_to_force
        
        if custom_omega_to_torque is not None:
            self.omega_to_torque = custom_omega_to_torque
        else:
            self.omega_to_torque = omega_to_torque


    @abstractmethod
    def get_reward(self, data: mjx.Data, thrust: jnp.ndarray, aux_state, metrics) -> jnp.ndarray:
        """
        Calculates the reward for a given simulation state and thrust input.

        Parameters:
            data (mjx.Data): The current state of the simulation.
            thrust (jnp.ndarray): The thrust applied to the quadcopter, represented as a JAX numpy array.
            aux_state (dict): Additional states target state and battery level (can be defined by the user).
            metrics: A dictionary field to store metrics during the simulation. update via metrics.update()

        Returns:
            reward (jnp.ndarray): A JAX numpy array representing the computed reward based on the provided state and thrust.
            done (jnp.ndarray): A JAX numpy array indicating whether the episode is done (1 for done, 0 for not done).
            metrics: The updated metrics.

        Example:
            reward = get_reward(current_state, applied_thrust)
        """
        """Calculates the reward for the given state and thrust."""
        pass

    @abstractmethod
    def init_metrics(self) -> dict:
        """
        Initialize simulation metrics.
        This method sets up and returns a dictionary of performance metrics
        that are used to monitor and evaluate the quadcopter's simulation.
        The metrics are initialized with their respective default values.
        Returns:
            dict: A dictionary where the keys are metric names and the values
                  are their initial values.
        """

        pass

    @abstractmethod
    def reset_aux_state(self, data: mjx.Data, rng: jnp.ndarray) -> dict:
        """
        Initialize auxiliary state.
        This method sets up and returns a dictionary of auxiliary states
        that are used to monitor and evaluate the quadcopter's simulation.
        The auxiliary states are initialized with their respective default values.

        Parameters:
            data (mjx.Data): The initial state of the simulation.
            rng (jnp.ndarray): A JAX random number generator state for initializing random values.

        Returns:
            dict: A dictionary where the keys are auxiliary state names and the values
                  are their initial values.
        """
        pass

    @abstractmethod
    def get_aux_observations(self, aux_state: dict) -> jnp.ndarray:
        """
        Extracts auxiliary observations from the auxiliary state dictionary.

        Parameters:
            aux_state (dict): A dictionary containing auxiliary states, including target state and battery level.

        Returns:
            jnp.ndarray: A JAX numpy array containing the observations.
        """
        pass

    @abstractmethod
    def step_aux_state(self, data, action, aux_states, current_time, rng: jnp.ndarray) -> dict:
        """
        Updates the auxiliary state based on the current simulation data and time.

        Parameters:
            data (mjx.Data): The current state of the simulation.
            aux_states (dict): A dictionary containing auxiliary states
            current_time (jnp.ndarray): The current time in the simulation in seconds
            rng (jnp.ndarray): A JAX random number generator state for sampling random values.

        Returns:
            dict: The updated auxiliary states.
        """
        pass

    def reset(self, rng: jnp.ndarray) -> State:
        """
        Resets the environment to an initial state.
        
        Parameters:
            rng (jnp.ndarray): A JAX random number generator state for sampling random values during reset.
        """
        rng, rng1, rng2, rng_quat, rng_ang_rate, rng_battery, rng_aux_state, rng_oszi, rng_pose_error, rng_motor, rng_omega = jax.random.split(rng, 11)

        ############################
        # sample state
        ############################
        # position
        low_pos, hi_pos = jnp.array([self.cfg.system.reset_pos_x_range[0], self.cfg.system.reset_pos_y_range[0], self.cfg.system.reset_pos_z_range[0]]), jnp.array([self.cfg.system.reset_pos_x_range[1], self.cfg.system.reset_pos_y_range[1], self.cfg.system.reset_pos_z_range[1]])
        qpos = self.sys.qpos0
        qpos = qpos.at[0:3].set(jax.random.uniform(rng1, (3,), minval=low_pos, maxval=hi_pos))

        # quaternions
        rng_quat_angle, rng_quat_axis = jax.random.split(rng_quat, 2)
        val = jnp.array([self.cfg.system.reset_quat_x, self.cfg.system.reset_quat_y, self.cfg.system.reset_quat_z])
        quat_axis = jax.random.uniform(rng_quat_axis, (3,), minval=-val, maxval=val)
        quat_axis /= jnp.linalg.norm(quat_axis)
        quat_angle = jax.random.uniform(rng_quat_angle, (), minval=0, maxval=self.cfg.system.reset_quat_angle * 2 * jnp.pi)

        quats = jax.numpy.array([jnp.cos(quat_angle / 2), quat_axis[0] * jnp.sin(quat_angle / 2), quat_axis[1] * jnp.sin(quat_angle / 2), quat_axis[2] * jnp.sin(quat_angle / 2)])

        qpos = qpos.at[3:7].set(quats)

        # velocity
        velocity_min = jnp.array([self.cfg.system.reset_velocity_x_max[0], self.cfg.system.reset_velocity_y_max[0], self.cfg.system.reset_velocity_z_max[0]])
        velocity_max = jnp.array([self.cfg.system.reset_velocity_x_max[1], self.cfg.system.reset_velocity_y_max[1], self.cfg.system.reset_velocity_z_max[1]])
        qvel = jnp.zeros((self.sys.nv,))
        qvel = qvel.at[0:3].set(jax.random.uniform(rng2, (3,), minval=velocity_min, maxval=velocity_max))

        # angular rate
        angular_rate_min = jnp.array([self.cfg.system.reset_angular_velocity_x_max[0], self.cfg.system.reset_angular_velocity_y_max[0], self.cfg.system.reset_angular_velocity_z_max[0]])
        angular_rate_max = jnp.array([self.cfg.system.reset_angular_velocity_x_max[1], self.cfg.system.reset_angular_velocity_y_max[1], self.cfg.system.reset_angular_velocity_z_max[1]])
        ang_rate = jax.random.uniform(rng_ang_rate, (3,), minval=angular_rate_min, maxval=angular_rate_max)
        qvel = qvel.at[3:6].set(ang_rate)

        data = self.pipeline_init(qpos, qvel)

        # sample battery level
        battery_level = jax.random.uniform(rng_battery, (1,), minval=3.0 / 4.2, maxval=1.0)

        # sample auxiliary state and put in state_info
        aux_state = self.reset_aux_state(data, rng_aux_state)

        dr_scaling = self.cfg.system.dr_scaling
        
        # sample pose error (to simulate motion capture errors, for example)
        key1, key2, key3, key4 = jax.random.split(rng_pose_error, 4)
        error_angle = jax.random.uniform(key1, (), minval=-self.cfg.system.reset_pose_error_angle * dr_scaling, maxval=self.cfg.system.reset_pose_error_angle * dr_scaling)
        error_axis = jax.random.uniform(key2, (3,), minval=-1.0, maxval=1.0)
        error_axis /= jnp.linalg.norm(error_axis) + 1e-6
        pose_error = jnp.array([jnp.cos(error_angle / 2), 
                                error_axis[0] * jnp.sin(error_angle / 2), 
                                error_axis[1] * jnp.sin(error_angle / 2), 
                                error_axis[2] * jnp.sin(error_angle / 2)
                               ])

        pos_error = jax.random.uniform(key3, (3,), 
                                       minval=-self.cfg.system.reset_pose_error_pos * dr_scaling, 
                                       maxval=self.cfg.system.reset_pose_error_pos * dr_scaling)

        obs = self.get_obs(data, jnp.concatenate([jnp.zeros((5,15)).flatten(), jnp.zeros((10, 4)).flatten()]), aux_state, battery_level, jnp.zeros((10, 10)), pose_error, pos_error)
        reward, done, zero = jnp.zeros(3)
        metrics = self.init_metrics()

        # sample motor model
        key1, key2, key3, key4, key5 = jax.random.split(rng_motor, 5)
        scales = jax.random.uniform(rng_motor, (3,), 
                                    minval=(1 - self.cfg.system.reset_motor_model_scale* dr_scaling), 
                                    maxval=(1 + self.cfg.system.reset_motor_model_scale* dr_scaling))
        scales2 = jax.random.uniform(key1, (3,4), 
                                     minval=(1 - self.cfg.system.reset_motor_diff_scale * dr_scaling), 
                                     maxval=(1 + self.cfg.system.reset_motor_diff_scale* dr_scaling))
        a = self.config["motor_a"] * scales[0] * scales2[0, :]
        b = self.config["motor_b"] * scales[1] * scales2[1, :]
        J_m = self.config["J_m"] * scales[2] * scales2[2, :]
        ad = jnp.exp(-a * 1e-4)
        bd = b * (1 - ad)

        state_info = {
            "rng": rng,
            "action_delays": jnp.zeros((7, 4)),
            "action_delays_pwm": jnp.zeros((5, 4)),
            "action_delay": jnp.zeros(8),
            "action_delay_1": jnp.zeros(8),
            "action_delay_pwm": jnp.zeros(4),
            "action_1": jnp.zeros(8),
            "action_filtered": jnp.zeros(8),
            "state_delays": jnp.zeros((5, 15)),
            "action_delays_obs": jnp.zeros((10, 4)),
            "battery_level": battery_level,
            "current_time": jnp.zeros(()),
            "pose_delays": jnp.zeros((10, 10)),
            "pose_error": pose_error,
            "pos_error": pos_error,
            "motor_ac": a,
            "motor_bc": b,
            "motor_a": ad,
            "motor_b": bd,
            "J_m": J_m,
            "Omega": jax.random.uniform(rng_omega, (4,), minval=1000.0, maxval=1500.0),  # initial angular velocity in rad/s
        }
        state_info = state_info | aux_state

        return State(data, obs, reward, done, metrics, state_info)

    def step(self, state: State, action: jnp.ndarray) -> State:
        """Runs one timestep of the environment's dynamics."""
        rng, omega_rng, meas_rng, aux_state_rng, vel_rng, angular_rate_rng = jax.random.split(state.info['rng'], 6)

        state_info = state.info
        aux_states = {k: state_info[k] for k in state_info if k not in self.__own_state_info_keys}

        # physics step
        data0 = state.pipeline_state
        action, action_uncomp = self.nn_action_to_physics_action(action, battery_level=state_info["battery_level"])
        rng, key2 = jax.random.split(rng, 2)
        if self.cfg.system.action_delay > 0:
            a = state_info["action_delays"][self.cfg.system.action_delay - 1, 0:4]
        else:
            a = action[0:4]

        def physics_step(i, val):
            data = val[0]
            omega = val[1]
            rng_key = val[2]
            rng_key, key2 = jax.random.split(rng_key, 2)
            
            torques = state_info["J_m"] * (-state_info["motor_ac"] * omega + state_info["motor_ac"] * state_info["motor_bc"] * a[0:4]) * 1e-3 + \
                  self.omega_to_torque(omega)  # compensate for motor delay
            omega = state_info["motor_a"] * omega + state_info["motor_b"] * a[0:4] + self.cfg.system.motor_noise_scale * jax.random.normal(key2,(4,)) 
            thrusts = self.omega_to_thrust(omega)

            real_action = jnp.concatenate([thrusts, torques], axis=0)  # only use first 4 actions for thrust, last 4 for torques
            data = self.pipeline_step(data, real_action)
            return (data, omega, rng_key)
        
        val = jax.lax.fori_loop(0, 40, physics_step, (data0, state_info["Omega"], omega_rng))
        data = val[0]
        omega = val[1]

        # state noise
        qvel = data.qvel.at[0:3].set(data.qvel[0:3] + jax.random.normal(vel_rng, (3, )) * self.cfg.system.velocity_noise_scale)
        qvel = qvel.at[3:6].set(qvel[3:6] + jax.random.normal(key2, (3, )) * self.cfg.system.angular_rate_noise_scale)
        data = data.tree_replace({"qvel": qvel})

        # calculate reward filter out own state info keys so user only gets its own auxiliary state
        reward, done, metrics = self.get_reward(data, action, aux_states, state.metrics)

        # get measurements
        obs = self.get_obs(data, omega, jnp.concatenate([state.info['state_delays'].flatten(), 
                                                  state.info['action_delays_obs'].flatten()]), 
                            aux_states, state.info['battery_level'], 
                            state_info["pose_delays"], 
                            state_info["pose_error"], 
                            state_info["pos_error"])

        # add measurement noise
        obs = obs.at[0:15].set(obs[0:15] + jax.random.normal(meas_rng, (15,)) * self.cfg.system.measurement_noise_scale)

        # update state_info
        state_info["rng"] = rng
        state_info["state_delays"] = jnp.roll(state_info["state_delays"], shift=1, axis=0)
        state_info["state_delays"] = state_info["state_delays"].at[0, :].set(obs[0:15])
        state_info["action_delays_obs"] = jnp.roll(state_info["action_delays_obs"], shift=1, axis=0)
        state_info["action_delays_obs"] = state_info["action_delays_obs"].at[0, :].set(action_uncomp[0:4])
        state_info["action_delays"] = jnp.roll(state_info["action_delays"], shift=1, axis=0)
        state_info["action_delays"] = state_info["action_delays"].at[0, :].set(action)
        state_info["pose_delays"] = jnp.roll(state_info["pose_delays"], shift=1, axis=0)
        state_info["pose_delays"] = state_info["pose_delays"].at[0, :].set(jnp.concatenate([data.qpos[0:3], data.qpos[3:7], data.qvel[0:3]]))
        state_info["current_time"] = state_info["current_time"] + self.cfg.system.simulation_dt
        state_info["Omega"] = val[1]

        aux_states = self.step_aux_state(data, action, aux_states, current_time=state_info["current_time"], rng=aux_state_rng)
        state_info = state_info | aux_states


        return state.replace(
            pipeline_state=data, obs=obs, reward=reward, done=done, info=state_info, metrics=metrics)
  
    def nn_action_to_physics_action(self, action: jnp.ndarray, battery_level: jnp.ndarray) -> jnp.ndarray:
        """
        Convert a neural network action to a physics action by normalizing the input to the PWM signal range and also simulating the battery
        Parameters:
            action (jnp.ndarray): An array containing at least 4 elements representing the
                action inputs, expected to be in the range [-1.0, 1.0]. The first four elements
                are used for normalization and scaling.
            battery_level (jnp.ndarray): An array representing the current battery level,
        Returns:
            tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
                A tuple containing:
                    - A concatenated jnp.ndarray of thrust and torque values derived from the PWM signal.
                    - The normalized action array (values in the range [0, 1]) for the first four inputs.
                    - The PWM signal
        """
        # normalize action to [0, 1]
        action = (jnp.clip(action[0:4], -1.0, 1.0) + 1.0) / 2.0

        # Clip action to battery level. 
        # The compensation itself is done in the firmware.
        # We only have to account that the maximum thrust is reduced by the battery level.
        action_battery =jnp.clip(action, min=0.0, max=battery_level[0])
        pwm = action_battery
        return pwm, action

    def parse_state(self, data: mjx.Data) -> dict:
        """
        Parses the generalized coordinates from mujoco into a state dictionary with position, quaternion, velocity and angular rate.

        Parameters:
            data (mjx.Data): The current state of the simulation.
        """
        return {
            "pos": data.qpos[0:3],
            "quat": data.qpos[3:7],
            "vel": data.qvel[0:3],
            "ang_rate": data.qvel[3:6],
        }

    def get_obs(
        self, data: State, omega: jnp.ndarray, past_states: jnp.ndarray, aux_states: dict, battery_level: jnp.ndarray, pose_delays, pose_error, pos_error
    ) -> jnp.ndarray:
        """
        Returns the observation for the current state. The observation includes position, velocity, rotation matrix, angular rate, past states, auxiliary states and battery level.

        Parameters:
            data (mjx.Data): The current state of the simulation.
            omega (jnp.ndarray): The current angular velocity of the motors.
            past_states (jnp.ndarray): A concatenated array of past states for state delay.
            aux_states (dict): A dictionary containing auxiliary states, including target state and battery level.
            battery_level (jnp.ndarray): An array representing the current battery level.
            pose_delays (jnp.ndarray): A concatenated array of past poses for pose delay
            pose_error (jnp.ndarray): A quaternion representing the error in pose estimation.
            pos_error (jnp.ndarray): A vector representing the error in position estimation.
        """

        parsed_state = self.parse_state(data)

        # Convert quaternion to rotation matrix
        if self.cfg.system.use_pose_delays:
            q = pose_delays[3, 3:7]
            pos = pose_delays[3, 0:3]
            vel = pose_delays[3, 7:10]
        else:
            q = parsed_state["quat"]
            pos = parsed_state["pos"]
            vel = parsed_state["vel"]
        rot_matrix = quat_to_rot_mat(q)

        # apply pose error to rotation matrix and position
        rot_matrix_dist = quat_to_rot_mat(pose_error)
        rot_matrix = rot_matrix @ rot_matrix_dist
        pos = parsed_state["pos"] + pos_error

        omega = omega / 2.9e3  # for numerical stability 
        
        aux_obs = self.get_aux_observations(aux_states)
        if not self.cfg.system.observe_omega:
            if aux_obs is None:
                return jnp.concatenate([pos, vel, rot_matrix[:, 0:2].flatten(), parsed_state["ang_rate"], past_states, battery_level])
                

            return jnp.concatenate([pos, vel, rot_matrix[:, 0:2].flatten(), parsed_state["ang_rate"], past_states, aux_obs, battery_level])
        else:
            if aux_obs is None:
                return jnp.concatenate([pos, vel, rot_matrix[:, 0:2].flatten(), parsed_state["ang_rate"], past_states, omega, battery_level])
            return jnp.concatenate([pos, vel, rot_matrix[:, 0:2].flatten(), parsed_state["ang_rate"], past_states, aux_obs, omega, battery_level])


def quat_to_rot_mat(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    rot_matrix = jnp.array([
        [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w),   2 * (x * z + y * w)],
        [2 * (x * y + z * w),   1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),   2 * (y * z + x * w),   1 - 2 * (x**2 + y**2)]
    ])
    return rot_matrix
    
