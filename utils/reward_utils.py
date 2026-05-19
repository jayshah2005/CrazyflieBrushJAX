import numpy as np 
import jax
import numpy as np
import jax.numpy as jnp 

def triangle_reward(r, max_r):
    r = jnp.abs(r)
    # if r > max_r:
    #     return 0
    return jax.nn.relu(1 - r / (max_r))

def gaussian_reward(r, std):
    return jnp.exp(-0.5 * ((r) / (std/3)) ** 2)

def laplace_reward(r, b):
    return jnp.exp(-jnp.abs(r) / (b/3))


def get_reward_function(name):
    if name == "triangle":
        return triangle_reward
    if name == "gaussian":
        return gaussian_reward
    if name == "laplace":
        return laplace_reward
    raise ValueError(f"Reward function {name} not implemented.")


def interval_constraint_reward(r, delta, epsilon):
    # shape of a trapezoid with a base of 2*delta andwidth of edges epsilon
    return jax.nn.relu(1 - jax.nn.relu(jnp.abs(r) - delta / 2) / epsilon)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Parameters for the interval constraint reward function
    delta = 1.0
    epsilon = 0.2

    # Create an array of r values
    r_values = jnp.linspace(-2, 2, 500)

    # Compute the reward for each r value
    rewards = interval_constraint_reward(r_values, delta, epsilon)

    # Convert rewards from jax array to numpy array for matplotlib compatibility
    rewards_np = np.array(rewards)

    # Plot the interval constraint reward function
    plt.figure(figsize=(8, 4))
    plt.plot(np.array(r_values), rewards_np, label='Interval Constraint Reward')
    plt.xlabel('r')
    plt.ylabel('Reward')
    plt.title('Interval Constraint Reward Function')
    plt.legend()
    plt.grid(True)
    plt.show()

