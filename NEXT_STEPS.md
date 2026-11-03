# Next steps: from trained policy to a flying Crazyflie Brushless

This fork gets the target-control pipeline from the paper running end to end on a single GPU:
**train → evaluate in simulation → export to C → flash the firmware**. This file covers what has been
done, where the results are, and what's left before a real flight.

## Where things are

| What | Where |
|---|---|
| Trained policy (final) | `trained_models/target_control/brax_save` |
| Simulated test flight of that policy | `trained_models/target_control/eval/` (video, plots, CSV) |
| Policy exported to C, ready for the firmware | `code_export/generated/nn.h`, `code_export/generated/nn.c` |
| Training outputs on the training machine | `~/crazyflie_logs/training/` (`brax_save`, `checkpoints/<step>`) |
| Eval outputs on the training machine | `~/crazyflie_logs/eval/<timestamp>/` |

## Results

_Filled in when training finishes._

## What was fixed in this fork

- **Training ran on CPU and crashed.** JAX is now the CUDA 13 build, pinned to `jax==0.9.2`.
  `brax==0.14.2` (the latest release) calls `jax.device_put_replicated`, which JAX removed in 0.10, so
  newer JAX breaks training. `flax==0.12.6` is the newest flax that supports JAX 0.9.
- **`Quadcopter.reset()` passed its arguments to `get_obs()` one slot off** (the motor speed was missing).
- **Config key `system.use_pose_delays` was missing.** It's now in `parameters/system/quadcopter.yaml` (default `false`).
- **The policy now trains without motor speed in its inputs** (`observe_omega: false`). The authors'
  firmware (`controller_nn.c`) builds a 134-value input with no motor RPM, so a policy that also reads motor
  speed (138 inputs) can't run on it. See "Using motor RPM" below.
- **Logs go to `~/crazyflie_logs`** (was `/my_log_dir`), and training works with W&B turned off.
- **`target_control_eval.py` was rewritten.** It used to load from the authors' machine and had a hardcoded
  network size that didn't match. It now loads your model, reads the network shape from the weights, and writes
  a video, plots and a CSV.
- **`export_network_to_c.py` was rewritten.** It used to load from the authors' machine and hardcoded the network
  size. A wrong size compiles fine in C and silently produces garbage motor commands. It now reads all sizes
  from the weights, checks them against the environment (and the firmware), compiles the generated C with gcc,
  and checks it gives the same output as the JAX policy before writing anything.

## Setting up on another machine

```bash
git clone https://github.com/jayshah2005/CrazyflieBrushJAX.git
cd CrazyflieBrushJAX
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -c "import jax; print(jax.devices())"   # should list a CudaDevice
```

`jax[cuda13]` needs an NVIDIA driver that supports CUDA 13 (`nvidia-smi` shows the CUDA version). Without a
GPU, JAX falls back to the CPU: evaluation and export still work, but training is far too slow.

## Step 1: Check the policy in simulation

```bash
python -m examples.target_control.target_control_eval eval_params=trained_models/target_control/brax_save
```

The drone starts hovering at (0, 0, 1.1) and flies through waypoints between x = -3 m and x = +3 m, 4 s each.
The outputs go to `~/crazyflie_logs/eval/<timestamp>/`. In `trajectory_0.png`, the solid position lines
should follow the dashed targets without overshooting much, roll/pitch should stay well clear of ±90°, and
the motor commands shouldn't sit at 0 or 65535 for long.

Options: `eval_episodes=5` (more random seeds), `eval_show_plots=true` (open the plot in a window),
`eval_time_per_waypoint=6.0`. To check an intermediate snapshot, point `eval_params` at a file in
`~/crazyflie_logs/training/checkpoints/`.

Simulation uses the nominal (system-identified) drone with sensor noise turned off. A good result here is
necessary for a real flight, but not enough on its own.

## Step 2: Export to C

```bash
python -m code_export.export_network_to_c --params trained_models/target_control/brax_save
```

This writes `code_export/generated/nn.h` and `nn.c`. It refuses to write anything if the network's input
size doesn't match the configured environment, or if the C code gives different motor commands from the
JAX policy (checked on 1000 random inputs; the expected difference is ~1e-6).

## Step 3: Build and flash the firmware

The firmware is the authors' fork of the Bitcraze firmware:
<https://github.com/Data-Science-in-Mechanical-Engineering/CrazyflieBrushJAX_firmware>

```bash
git clone --recursive https://github.com/Data-Science-in-Mechanical-Engineering/CrazyflieBrushJAX_firmware.git
cd CrazyflieBrushJAX
python -m code_export.export_network_to_c --params trained_models/target_control/brax_save \
    --firmware ../CrazyflieBrushJAX_firmware
```

With `--firmware`, the export also reads the firmware's `NUM_NN_INPUTS` (currently 134) and refuses on a
mismatch. It then overwrites `src/modules/interface/controller/nn.h` and `src/modules/src/controller/nn.c`
in the firmware checkout.

Then build and flash with the standard Bitcraze workflow. You need the `arm-none-eabi-gcc` toolchain and,
for `cload`, the Crazyradio and `cfclient`/`cflib`. See the Bitcraze docs for setup.

```bash
cd ../CrazyflieBrushJAX_firmware
make cf21bl_defconfig
make -j$(nproc)
make cload           # put the Crazyflie in bootloader mode first
```

The NN controller is **controller 6** (set the parameter `stabilizer.controller` to 6). It flies to the
position setpoint you send it.

## Before the first real flight

- **Emergency stop.** The authors' firmware turns off the stock tumble shutdown, so the drone can do flips. It
  shuts down if it gets no position update for 0.5 s, which **only works with Crazyswarm2 plus a motion-capture
  system**. Their README says it does *not* work with Lighthouse or Loco. To stop the drone, unplug the
  Crazyradio. Without mocap there's no automatic shutdown at all.
- **Check the battery input.** In simulation, the network's last input is `voltage / 4.2` (0.71 to 1.0,
  `environment/quadcopter.py`, `reset()`). The firmware currently feeds `4.2 / voltage` (1.0 to 1.4,
  `controller_nn.c`, the line marked `// 4.2f / supplyVoltage;`). They agree only on a full battery and move
  in opposite directions as it drains. Confirm which is intended (and fix the firmware or the simulator)
  before long flights. Short flights on a fully charged battery are least affected.
- **Coordinate frame.** The policy uses the position error relative to the setpoint, with the setpoint treated
  as 1 m above the target (`input[2] = z - (setpoint_z - 1)` in the firmware). Send setpoints in the same
  world frame your mocap uses.
- **Start conservatively.** Use prop guards and a net, keep the first setpoint a low hover, and fly short
  flights on a full battery.

## Retraining

```bash
python -m examples.target_control.target_control_train
```

This takes about 40 min on one RTX-class GPU (400M environment steps, 10 evaluations). Outputs go to
`~/crazyflie_logs/training/`. **A new run writes into the same folder.** Old checkpoints stay there until each
one is overwritten, so move the old folder away first if you want to keep it (or set
`experiment_file_label=<name>`). Eval and export both refuse a checkpoint whose input size doesn't match the
config, so a mixed-up file fails loudly instead of being used.

W&B logging is on (`wandb_log`, `wandb_entity` in `parameters/main.yaml`). Set `wandb_log=false` to just print
the reward after each evaluation.

## Using motor RPM (possible follow-up)

The Crazyflie 2.1 Brushless *can* measure motor speed: its BlueJay ESCs report RPM back over bidirectional
DShot, and mainline Bitcraze firmware exposes it as `motor.m1_rpm`…`m4_rpm`. Training with
`system.observe_omega=true` gives the policy that information (138 inputs). Running that on hardware would
require:

1. Checking that the authors' firmware fork includes the bidirectional DShot driver.
2. Changing `controller_nn.c` to add the 4 motor speeds in the same input positions as `get_obs()` uses,
   scaled the same way (rad/s ÷ 2900), and setting `NUM_NN_INPUTS` to 138.
3. Thinking about the gap between simulation and reality: the simulator gives the policy the exact motor speed
   instantly, while the real telemetry is noisy and delayed.

## Known loose ends

- `code_export/export_network_to_c_flip.py` (for the flip task) still has the authors' hardcoded paths and
  sizes. Port it the same way if you train the flip controller.
- The Hydra "version_base 1.1" and "Brax is not actively maintained" warnings are harmless.
