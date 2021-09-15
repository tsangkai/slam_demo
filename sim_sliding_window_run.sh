#!/bin/bash

DURATION="1000"  #300
TRIALS="10"   #50
FIXED_WINDOW_SIZE="150"

./bin/sim_sliding_window_app      -duration=$DURATION
./bin/sim_sliding_window_opt_app  -duration=$DURATION -trials=$TRIALS -fixed_window_size=$FIXED_WINDOW_SIZE
./bin/sim_sliding_window_em_app   -duration=$DURATION -trials=$TRIALS -fixed_window_size=$FIXED_WINDOW_SIZE
./bin/sim_sliding_window_boem_app -duration=$DURATION -trials=$TRIALS

python eval/sim_sliding_window_eval.py $TRIALS
