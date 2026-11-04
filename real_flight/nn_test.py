"""First real flights of the NN policy with Lighthouse positioning.

Takes off and hovers on the stock PID controller, optionally hands control to the
NN controller (stabilizer.controller = 6) for a few moves, then switches back to
PID to land. Ctrl-C at any time hands back to PID and lands.

Run with a Python that has cflib installed (not the training .venv):
    python real_flight/nn_test.py              # PID only: check the drone and Lighthouse
    python real_flight/nn_test.py --nn         # NN, 0.5 m steps forward/back
    python real_flight/nn_test.py --nn --step 1.0

Hard kill: set the parameter stabilizer.stop to 1. Unplugging the Crazyradio does
not stop the drone right away (the high-level commander runs onboard); the
firmware cuts the motors after 2 s without radio contact.
"""
import argparse
import time

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

PID, NN = "1", "6"

parser = argparse.ArgumentParser()
parser.add_argument("--uri", default="radio://0/80/2M/E7E7E7E7E7", help="the URI cfclient shows")
parser.add_argument("--nn", action="store_true", help="hand control to the NN while hovering")
parser.add_argument("--step", type=float, default=0.5, help="forward/back distance in metres")
parser.add_argument("--height", type=float, default=1.0, help="hover height above the start point")
args = parser.parse_args()

cflib.crtp.init_drivers()
with SyncCrazyflie(args.uri, cf=Crazyflie()) as scf:
    cf = scf.cf
    hl = cf.high_level_commander

    cf.param.set_value("stabilizer.controller", PID)
    cf.param.set_value("kalman.resetEstimation", "1")
    time.sleep(0.1)
    cf.param.set_value("kalman.resetEstimation", "0")
    time.sleep(2)  # let the estimator converge on the Lighthouse position
    cf.platform.send_arming_request(True)
    time.sleep(1)

    try:
        hl.takeoff(args.height, 2.5)
        time.sleep(4)
        if args.nn:
            cf.param.set_value("stabilizer.controller", NN)
            print("NN in control")
            time.sleep(4)  # NN holds the hover
            # go_to moves the setpoint smoothly over 2 s instead of jumping it
            for dx in (args.step, -args.step, args.step, -args.step):
                hl.go_to(dx, 0, 0, 0, 2.0, relative=True)
                time.sleep(4)
    except KeyboardInterrupt:
        print("Ctrl-C: handing back to PID")
    finally:
        cf.param.set_value("stabilizer.controller", PID)
        print("PID in control, landing")
        time.sleep(0.5)
        hl.land(0.0, 2.5)
        time.sleep(3)
        hl.stop()
        cf.platform.send_arming_request(False)
