#!/bin/bash

# FIXED TIME WINDOW EXPERIMENTS
./build/sim_sliding_window/sim_app_s_win
REPEAT_VAR_FIXED="1"
./build/sim_sliding_window/sim_opt_app_s_win $REPEAT_VAR_FIXED
./build/sim_sliding_window/sim_em_app_s_win $REPEAT_VAR_FIXED
./build/sim_sliding_window/sim_boem_app_s_win $REPEAT_VAR_FIXED

python3 sim_sliding_window_eval.py $REPEAT_VAR_FIXED







