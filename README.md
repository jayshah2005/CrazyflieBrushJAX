# CrazyflieBrushJAX
This is the accompanying repository to the paper "How to Model Your Crazyflie Brushless".

> **Fork notes:** this fork fixes training on GPU, evaluation and export, and includes a trained
> target-control policy. See [NEXT_STEPS.md](NEXT_STEPS.md) for what changed and how to go from the
> trained policy to a real flight.

# Installation
Clone the git repo and install the dependencies (Tested on Python 3.12)
```bash
pip install -r requirements.txt
```
On Linux this installs the CUDA 13 build of JAX (needs an NVIDIA driver with CUDA 13 support; macOS and
Windows get CPU-only JAX, enough for eval and export but not training). Versions of jax, flax
and brax are pinned because brax 0.14.2 does not work with jax 0.10 or newer.

# How to use it
This repo has two tasks a) allowing to train the model on the Crazyflie Brushless model and b) exporting resulting networks onto the Crazyflie firmware for real world experiments.

## Training
The examples/ folder contains entrance points. One training the target controller and one training a flip controller.
Run
```bash
python -m examples.target_control.target_control_train
```
or 
```bash
python -m examples.flip.flip_train
```

To configure the runs, we use hydra (https://hydra.cc/). The entrance point for our configuration is in parameters/main.yaml. There one can also configure logging directories and weights&biases.

The dynamics simulation itself can be found in environment/quadcopter.py.
To train the Crazyflie Brushless on a new task, users have to derive a new class from the abstract class Quadcopter from environment/quadcopter.py and overwrite abstract functions regarding reward and new auxiliary states (which can be used e.g., to track time or setpoints). For examples please refer to examples/target_control/target_control_train.py.

## Evaluation
```bash
python -m examples.target_control.target_control_eval
```
Flies the trained policy through a sequence of waypoints in simulation and saves a video, plots and a CSV
to `${root_dir}/eval/<timestamp>/`. Settings are in parameters/eval.yaml.

## Export
code_export/export_network_to_c.py converts the trained network to C code for the firmware.
```bash
python -m code_export.export_network_to_c [--params <policy>] [--firmware <firmware checkout>]
```
It reads the network shape from the saved policy, checks it against the environment (and the firmware's
`NUM_NN_INPUTS`), and verifies the generated C against the JAX policy before writing
`code_export/generated/nn.{h,c}` (and copying them into the firmware if `--firmware` is given).
The Crazyflie firmware itself can be found in https://github.com/Data-Science-in-Mechanical-Engineering/CrazyflieBrushJAX_firmware.
