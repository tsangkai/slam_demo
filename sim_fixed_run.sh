#!/bin/bash

REPEAT_VAR="5"  # 50

./bin/sim_fixed_app $REPEAT_VAR
./bin/sim_fixed_opt_app $REPEAT_VAR
./bin/sim_fixed_em_app $REPEAT_VAR
./bin/sim_fixed_boem_app $REPEAT_VAR

python eval/sim_fixed_eval.py $REPEAT_VAR