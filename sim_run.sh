#!/bin/bash

./build/sim/sim_app
./build/sim/sim_opt_app
# ./build/sim/sim_em_app
# ./build/sim/sim_boem_app

python3 sim_eval.py
