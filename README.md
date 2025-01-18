# CrazyflieBrushJAX
This is the accompanying repository to the paper "How to Model Your Crazyflie Brushless".

# Installation
Clone the git repo and install the dependencies (Tested on Python 3.12)
```bash
pip install -r requirements.txt
```

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

## Export
The resulting network of the training can be found in code_export/export_network_to_c.py.
```bash
python -m code_export.export_network_c
```
In the script, the user needs to modify the paths of the training results and the Crazyflie firmware.
The Crazyflie firmware itself can be found in https://github.com/Data-Science-in-Mechanical-Engineering/CrazyflieBrushJAX_firmware.
