#!/bin/bash

DURATION="50" # sec

./bin/sim_easy_app      -duration=$DURATION
./bin/sim_easy_opt_app  -duration=$DURATION
./bin/sim_easy_em_app   -duration=$DURATION
./bin/sim_easy_boem_app -duration=$DURATION

python eval/sim_easy_eval.py

