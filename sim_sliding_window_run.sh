#!/bin/bash

# FIXED TIME WINDOW EXPERIMENTS
./bin/sim_sliding_window_app
REPEAT_VAR_FIXED="1"
./bin/sim_sliding_window_opt_app $REPEAT_VAR_FIXED
./bin/sim_sliding_window_em_app $REPEAT_VAR_FIXED
./bin/sim_sliding_window_boem_app $REPEAT_VAR_FIXED

python3 eval/sim_sliding_window_eval.py $REPEAT_VAR_FIXED







