#!/bin/bash

REPEAT_VAR="5"  # 50

./build/sim_fixed/sim_fixed_app $REPEAT_VAR
./build/sim_fixed/sim_fixed_opt_app $REPEAT_VAR
./build/sim_fixed/sim_fixed_em_app $REPEAT_VAR
./build/sim_fixed/sim_fixed_boem_app $REPEAT_VAR

python3 sim_fixed_eval.py $REPEAT_VAR