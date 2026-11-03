"""Export a trained target-control policy to C for the Crazyflie firmware.

Usage (from the repo root):
    python -m code_export.export_network_to_c
    python -m code_export.export_network_to_c --params ~/crazyflie_logs/training/checkpoints/420249600
    python -m code_export.export_network_to_c --firmware ../CrazyflieBrushJAX_firmware
    python -m code_export.export_network_to_c system.observe_omega=true   # Hydra overrides go last

What it does:
  1. Loads the policy (default: ${root_dir}/${experiment_file_label}/brax_save from parameters/main.yaml)
     and reads the network shape from the saved weights. Nothing about the shape is hardcoded.
  2. Checks that the number of network inputs matches the observation size of the environment
     built from the current config, and the firmware's NUM_NN_INPUTS if --firmware is given.
     The generated C cannot check this itself: a mismatch would compile fine and silently
     produce garbage motor commands.
  3. Generates nn.h / nn.c, compiles them with gcc together with a small test harness, and checks
     that the C network gives the same motor commands as the JAX policy on random inputs.
  4. Only if every check passes, writes nn.h / nn.c to --out (default: code_export/generated/)
     and, if given, into the firmware checkout.

The C network mirrors brax's PPO policy as trained by target_control_train.py:
inputs normalized as (x - mean) / std (done in the firmware's controller_nn.c), dense layers with
SiLU (brax's default "swish") activations, and a linear output layer whose first values are the
action means. The firmware applies tanh to the first 4 outputs and uses them as the motor commands,
which is the deterministic action of the policy (the simulator also only uses actions 0-3).
"""
import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import jax
import numpy as np
from hydra import compose, initialize_config_dir
from jinja2 import Environment, FileSystemLoader

from brax.io import model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

jax.config.update("jax_platforms", "cpu")  # export is tiny; keep the GPU free for training

REPO = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = REPO / "code_export"
NUM_MOTORS = 4            # the firmware turns the first 4 network outputs into motor PWM
VERIFY_SAMPLES = 1000     # random inputs used to compare the C and JAX networks
VERIFY_TOLERANCE = 1e-4   # max allowed difference in motor command (range -1..1)

# Where the generated files go inside a CrazyflieBrushJAX_firmware checkout,
# and the controller whose NUM_NN_INPUTS must match the network.
FIRMWARE_HEADER = Path("src/modules/interface/controller/nn.h")
FIRMWARE_SOURCE = Path("src/modules/src/controller/nn.c")
FIRMWARE_CONTROLLER = Path("src/modules/src/controller/controller_nn.c")


def load_config(overrides):
    """Compose parameters/main.yaml (with overrides) the same way the training script does."""
    # version_base matches the training script so the config resolves identically
    with initialize_config_dir(config_dir=str(REPO / "parameters"), version_base="1.1"):
        return compose(config_name="main", overrides=overrides)


def load_policy(params_path: Path):
    """Return (brax params, normalizer state, [(kernel, bias), ...]) from a saved policy."""
    if not params_path.is_file():
        sys.exit(f"No policy found at {params_path}. Pass --params <path>.")
    params = model.load_params(params_path)  # (normalizer state, policy params, value params)
    normalizer, policy_params = params[0], params[1]["params"]

    # brax names the policy's dense layers hidden_0 ... hidden_N (the last one is the output layer)
    names = sorted(policy_params, key=lambda n: int(n.split("_")[1]))
    if names != [f"hidden_{i}" for i in range(len(names))]:
        sys.exit(f"Unexpected policy layers {list(policy_params)}; this exporter only supports brax's MLP policy.")
    layers = [(np.asarray(policy_params[n]["kernel"]), np.asarray(policy_params[n]["bias"])) for n in names]
    return params, normalizer, layers


def check_shapes(layers):
    """Validate the layer chain and return (input_size, hidden_size, output_size)."""
    for (k1, _), (k2, _) in zip(layers, layers[1:]):
        if k1.shape[1] != k2.shape[0]:
            sys.exit(f"Layer shapes do not chain: {k1.shape} -> {k2.shape}")
    hidden_sizes = {k.shape[1] for k, _ in layers[:-1]}
    if len(hidden_sizes) != 1:
        # nn.c ping-pongs between two buffers of one fixed size
        sys.exit(f"All hidden layers must have the same width for the C code, got {sorted(hidden_sizes)}")
    input_size, output_size = layers[0][0].shape[0], layers[-1][0].shape[1]
    if output_size < 2 * NUM_MOTORS:
        sys.exit(f"Policy has {output_size} outputs, expected at least {2 * NUM_MOTORS} (mean and std per motor)")
    return input_size, hidden_sizes.pop(), output_size


def env_observation_size(cfg):
    """Observation size of the environment the current config describes."""
    # imported here because it pulls in the whole simulation stack
    from examples.target_control.target_control_train import QuadcopterTargetControl
    return QuadcopterTargetControl(cfg).observation_size


def firmware_num_inputs(firmware: Path):
    """Evaluate `#define NUM_NN_INPUTS (...)` from the firmware's NN controller."""
    controller = firmware / FIRMWARE_CONTROLLER
    if not controller.is_file():
        sys.exit(f"{controller} not found. Is {firmware} a CrazyflieBrushJAX_firmware checkout?")
    match = re.search(r"#define\s+NUM_NN_INPUTS\s+(.+)", controller.read_text())
    if not match or not re.fullmatch(r"[\d\s()+*]+", match.group(1).strip()):
        sys.exit(f"Could not read NUM_NN_INPUTS from {controller}")
    return eval(match.group(1))  # safe: checked above that it is only digits, + * and parentheses


def c_array(name, values):
    return f"static const float {name}[] = {{{', '.join(f'{v}' for v in values)}}};\n"


def generate_c(normalizer, layers, input_size, hidden_size, output_size):
    """Render nn.h and nn.c from the jinja templates. Returns (header, source)."""
    for i, (kernel, bias) in enumerate(layers):
        if not (np.all(np.isfinite(kernel)) and np.all(np.isfinite(bias))):
            sys.exit(f"Layer hidden_{i} contains NaN/inf weights; the policy is broken.")
    mean, std = np.asarray(normalizer.mean), np.asarray(normalizer.std)
    if not (np.all(np.isfinite(mean)) and np.all(np.isfinite(std)) and np.all(std > 0)):
        # the firmware divides by input_std
        sys.exit("Observation normalizer has NaN/inf values or a non-positive std; the policy is broken.")

    h = f"#define NN_INPUT_SIZE {input_size}\n#define NN_OUTPUT_SIZE {output_size}\n\n"
    h += c_array("input_mean", mean)
    h += c_array("input_std", std)
    for i, (kernel, bias) in enumerate(layers):
        # fc_layer reads kernel[out * input_size + in], so store the transpose, one output neuron per line
        h += f"\n// Layer {i}: {kernel.shape[0]} -> {kernel.shape[1]}\n"
        h += f"static const float hidden_{i}_kernel[] = {{\n"
        h += "".join("  " + ", ".join(f"{v}" for v in row) + ",\n" for row in kernel.T)
        h += "};\n"
        h += c_array(f"hidden_{i}_bias", bias)

    # input -> buffer1, then swap between the two buffers for each hidden layer, last layer -> output
    last = len(layers) - 1
    # (the template already indents the first line)
    c = f"fc_layer(input, buffer1, hidden_0_kernel, hidden_0_bias, {input_size}, {hidden_size}, 1);\n"
    for i in range(1, last):
        c += f"    fc_layer(current_input, current_output, hidden_{i}_kernel, hidden_{i}_bias, {hidden_size}, {hidden_size}, 1);\n"
        c += "    temp = current_input;\n    current_input = current_output;\n    current_output = temp;\n"
    c += f"    fc_layer(current_input, output, hidden_{last}_kernel, hidden_{last}_bias, {hidden_size}, {output_size}, 0);\n"

    templates = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)))
    header = templates.get_template("nn.h.jinja").render(network_params=h, name="")
    source = templates.get_template("nn.c.jinja").render(inference=c, name="", h_file="nn.h", hidden_dim=hidden_size)
    return header, source


# Mimics the firmware: normalize the inputs, call forward_(), print the motor commands (tanh applied in nn.c).
HARNESS_C = r"""
#include <stdio.h>
#include "nn.h"
int main(void) {
    float raw[NN_INPUT_SIZE], x[NN_INPUT_SIZE], out[NN_OUTPUT_SIZE];
    while (fread(raw, sizeof(float), NN_INPUT_SIZE, stdin) == NN_INPUT_SIZE) {
        for (int i = 0; i < NN_INPUT_SIZE; i++) x[i] = (raw[i] - input_mean[i]) / input_std[i];
        forward_(x, out);
        fwrite(out, sizeof(float), %d, stdout);
    }
    return 0;
}
""" % NUM_MOTORS


def verify_against_jax(params, normalizer, layers, workdir: Path):
    """Compile the generated C and compare its motor commands with the JAX policy. Returns max abs error."""
    input_size, hidden_size, output_size = check_shapes(layers)
    (workdir / "harness.c").write_text(HARNESS_C)
    exe = workdir / "harness"
    build = subprocess.run(["gcc", "-O2", "-std=c99", "-Wall", "-Wno-unused-variable", "-o", str(exe),
                            str(workdir / "harness.c"), str(workdir / "nn.c"), "-lm"],
                           capture_output=True, text=True)
    if build.returncode != 0:
        sys.exit(f"Generated C failed to compile:\n{build.stderr}")

    # Random observations spread like the training data
    rng = np.random.default_rng(0)
    mean, std = np.asarray(normalizer.mean), np.asarray(normalizer.std)
    inputs = (mean + std * rng.standard_normal((VERIFY_SAMPLES, input_size))).astype(np.float32)

    run = subprocess.run([str(exe)], input=inputs.tobytes(), capture_output=True, check=True)
    c_out = np.frombuffer(run.stdout, dtype=np.float32).reshape(VERIFY_SAMPLES, NUM_MOTORS)

    # The same policy in JAX; deterministic=True gives tanh(mean action)
    network = ppo_networks.make_ppo_networks(
        input_size, output_size // 2,
        preprocess_observations_fn=running_statistics.normalize,
        policy_hidden_layer_sizes=(hidden_size,) * (len(layers) - 1),
        activation=jax.nn.silu,
    )
    policy = ppo_networks.make_inference_fn(network)(params, deterministic=True)
    jax_out, _ = jax.jit(jax.vmap(lambda o: policy(o, jax.random.PRNGKey(0))))(inputs)
    return float(np.max(np.abs(c_out - np.asarray(jax_out)[:, :NUM_MOTORS])))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--params", type=Path, help="saved policy (default: final policy of the training run)")
    parser.add_argument("--out", type=Path, default=REPO / "code_export" / "generated",
                        help="where to write nn.h and nn.c (default: code_export/generated/)")
    parser.add_argument("--firmware", type=Path,
                        help="CrazyflieBrushJAX_firmware checkout: check its NUM_NN_INPUTS and copy the files into it")
    parser.add_argument("overrides", nargs="*", help="Hydra config overrides, e.g. system.observe_omega=true")
    args = parser.parse_args()

    cfg = load_config(args.overrides)
    params_path = (args.params or Path(cfg.root_dir) / cfg.experiment_file_label / "brax_save").expanduser()
    params, normalizer, layers = load_policy(params_path)
    input_size, hidden_size, output_size = check_shapes(layers)
    print(f"Policy {params_path}: {input_size} inputs, {len(layers) - 1} hidden layers of {hidden_size}, {output_size} outputs")

    env_size = env_observation_size(cfg)
    if input_size != env_size:
        sys.exit(f"The policy has {input_size} inputs but the configured environment produces {env_size} "
                 f"observations (system.observe_omega={cfg.system.observe_omega}). Export the policy that was "
                 f"trained with this config, or pass matching overrides.")
    if args.firmware:
        fw_size = firmware_num_inputs(args.firmware)
        if input_size != fw_size:
            sys.exit(f"The policy has {input_size} inputs but the firmware's NUM_NN_INPUTS is {fw_size}.")

    header, source = generate_c(normalizer, layers, input_size, hidden_size, output_size)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "nn.h").write_text(header)
        (tmp / "nn.c").write_text(source)
        error = verify_against_jax(params, normalizer, layers, tmp)
    if not error <= VERIFY_TOLERANCE:  # written this way so a NaN error also fails
        sys.exit(f"C and JAX networks disagree (max motor command difference {error:.2e}); not writing any files.")
    print(f"C network matches JAX on {VERIFY_SAMPLES} random inputs (max difference {error:.1e})")

    targets = [(args.out / "nn.h", args.out / "nn.c")]
    if args.firmware:
        targets.append((args.firmware / FIRMWARE_HEADER, args.firmware / FIRMWARE_SOURCE))
    for h_path, c_path in targets:
        h_path.parent.mkdir(parents=True, exist_ok=True)
        c_path.parent.mkdir(parents=True, exist_ok=True)
        h_path.write_text(header)
        c_path.write_text(source)
        print(f"Wrote {h_path} and {c_path}")


if __name__ == "__main__":
    main()
