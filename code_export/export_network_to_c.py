from pathlib import Path
import numpy as np
import os
import jax
import jax.numpy as jnp


from brax.training.agents.ppo import networks as ppo_networks
from brax.io import model
from brax.training.acme import running_statistics

from jinja2 import Template, Environment, FileSystemLoader

if __name__ == "__main__":
    jax.config.update('jax_platform_name', 'cpu')

    base_path = "../crazyflie-firmware/src/modules"

    # create agent
    normalize = running_statistics.normalize
    input_dim = (3+6+3+3) * 6 + 10 * 4 + 3 + 1 
    output_dim = 8
    hidden_dim = 32
    ppo_network = ppo_networks.make_ppo_networks(
      input_dim, output_dim, preprocess_observations_fn=normalize, activation=jax.nn.silu
    )
    make_policy = ppo_networks.make_inference_fn(ppo_network)
    files = [f for f in Path("/data/RIG/target_control_flip7/starting/6/checkpoints").iterdir() if f.is_file() and f.stem.isdigit()]
    if not files:
        raise FileNotFoundError("No file with a numeric name found in /data/RIG")
    valid_files = [f for f in files if int(f.stem) <= 12000_000_000]
    if not valid_files:
        raise FileNotFoundError("No file with a numeric name below 100 million found in /data/RIG")
    largest_file = max(valid_files, key=lambda f: int(f.stem))
    print("Loading params from:", largest_file)
    params = model.load_params(largest_file)
    jit_inference_fn = jax.jit(make_policy(params))

    normalize_params = params[0]
    params = params[1]

    h_code = ""

    h_code += f"static const float input_mean[] = {{" + ", ".join(f"{val}" for val in normalize_params.mean) + "};\n"
    h_code += f"static const float input_std[] = {{" + ", ".join(f"{val}" for val in normalize_params.std) + "};\n"

    for i, hidden_name in enumerate(params['params']):
        hidden_name = f"hidden_{i}"
        h_code += f"// Param {i}: {params['params'][hidden_name]['kernel'].shape}\n"
        h_code += f"static const float hidden_{i}_kernel[] = {{"
        for row in params['params'][hidden_name]['kernel'].T:
            h_code += "  " + ", ".join(f"{val}" for val in row) + ","
        h_code += "};\n\n"
        h_code += f"static const float hidden_{i}_bias[] = {{"
        for val in params['params'][hidden_name]['bias']:
            h_code += f"  {val},"
        h_code += "};\n\n"


    jinja_environment = Environment(loader=FileSystemLoader("code_export"))
    template_xml_file = jinja_environment.get_template(f'nn.h.jinja')
    config = {"network_params": h_code, "name": ""}
    output = template_xml_file.render(config)

    with open(f"{base_path}/interface/controller/nn.h", "w") as f:
        f.write(output)

    print("File 'nn.h' created.")

    c_code = ""

    c_code += f"    fc_layer(input, buffer1, hidden_0_kernel, hidden_0_bias, {input_dim}, {hidden_dim}, 1);\n"
    for i, hidden_name in enumerate(params['params']):
        if i > 0 and i < len(params['params']) - 1:
            c_code += f"    fc_layer(current_input, current_output, hidden_{i}_kernel, hidden_{i}_bias, {hidden_dim}, {hidden_dim}, 1);\n"
            c_code += "      temp = current_input;\n"
            c_code += "      current_input = current_output;\n"
            c_code += "      current_output = temp;\n"
    c_code += f"    fc_layer(current_input, output, hidden_{len(params['params']) - 1}_kernel, hidden_{len(params['params']) - 1}_bias, {hidden_dim}, {output_dim*2}, 0);\n"
        
    template_xml_file = jinja_environment.get_template(f'nn.c.jinja')
    config = {"inference": c_code, "name": "", "h_file": "nn.h"}
    output = template_xml_file.render(config)

    with open(f"{base_path}/src/controller/nn.c", "w") as f:
        f.write(output)

    print("File 'nn.c' created.")
