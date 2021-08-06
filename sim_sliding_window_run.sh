#!/bin/bash

# FIXED TIME WINDOW EXPERIMENTS
REPEAT_VAR_FIXED="10"

./bin/sim_sliding_window_app
./bin/sim_sliding_window_opt_app $REPEAT_VAR_FIXED
./bin/sim_sliding_window_em_app $REPEAT_VAR_FIXED
./bin/sim_sliding_window_boem_app $REPEAT_VAR_FIXED

python eval/sim_sliding_window_eval.py $REPEAT_VAR_FIXED







