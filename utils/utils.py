import os
import equinox as eqx
import numpy as np

def quaternion_to_rotation_matrix(quat):
    q0, q1, q2, q3 = quat
    R = np.array([
        [1 - 2*(q2**2 + q3**2), 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3), 1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 1 - 2*(q1**2 + q2**2)]
    ])
    return R


def log(wandb, data):
    if wandb is not None:
        wandb.log(data=data) 

def create_model_dir():
    if not os.path.exists("models"):
        os.makedirs("models")

def save_model(cfg, model, name=""):
    eqx.tree_serialise_leaves(f"models/{name}{cfg.rl_agent.name}.eqx", model)

def load_model(cfg, model, name=""):
    return eqx.tree_deserialise_leaves(f"models/{cfg.rl_agent.name}.eqx", model)


if __name__ == "__main__":
    f = 10 / 15   #xy: 10 / 15  z: 10 / 15
    D = 0.17#0.075# z0.17
    h = 1.7
    g = 9.81
    m = 0.043
    w = 2 * np.pi * f

    uncertainty = 0.1

    f_upper = (f*(1+uncertainty))
    f_lower = (f*(1-uncertainty))

    w_upper = 2 * np.pi * f_upper
    w_lower = 2 * np.pi * f_lower

    h_upper = h * (1 + uncertainty)
    h_lower = h * (1 - uncertainty)

    D_upper = D * (1 + uncertainty)
    D_lower = D * (1 - uncertainty)

    m_upper = m * (1 + uncertainty)
    m_lower = m * (1 - uncertainty)

    I_upper = (m_upper * g * D_upper**2) / (4 * h_lower * w_lower**2)
    I_lower = (m_lower * g * D_lower**2) / (4 * h_upper * w_upper**2)

    I = (m * g * D**2) / (4 * h * w**2)
    print(f"I_lower: {I_lower}, I: {I}, I_upper: {I_upper}")

