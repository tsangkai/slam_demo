#!/bin/bash

./bin/sim_easy_app
./bin/sim_easy_opt_app
./bin/sim_easy_em_app
./bin/sim_easy_boem_app

python eval/sim_easy_eval.py

