#!/bin/bash

DURATION="50"
TRIALS="50"

./bin/sim_fixed_app      -duration=$DURATION -trials=$TRIALS
./bin/sim_fixed_opt_app  -duration=$DURATION -trials=$TRIALS
./bin/sim_fixed_em_app   -duration=$DURATION -trials=$TRIALS
./bin/sim_fixed_boem_app -duration=$DURATION -trials=$TRIALS

python eval/sim_fixed_eval.py $TRIALS