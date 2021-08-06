#!/bin/bash

DURATION="300"
TRIALS="5"   # 50
FIXED_WINDOW_SIZE="60"

./bin/sim_sliding_window_app      -duration=$DURATION
./bin/sim_sliding_window_opt_app  -duration=$DURATION -trials=$TRIALS
./bin/sim_sliding_window_em_app   -duration=$DURATION -trials=$TRIALS -fixed_window_size=$FIXED_WINDOW_SIZE
./bin/sim_sliding_window_boem_app -duration=$DURATION -trials=$TRIALS -fixed_window_size=$FIXED_WINDOW_SIZE

python eval/sim_sliding_window_eval.py $TRIALS
