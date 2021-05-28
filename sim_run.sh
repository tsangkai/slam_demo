#!/bin/bash

./build/sim/sim_app
./build/sim/sim_opt_app
./build/sim/sim_em_app
./build/sim/sim_boem_app

python3 sim_eval.py


# REPEAT_VAR="50"
# perf stat -r $REPEAT_VAR ./build/sim/sim_boem_app