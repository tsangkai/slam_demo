#!/bin/bash

./build/sim_easy/sim_easy_app
./build/sim_easy/sim_easy_opt_app
./build/sim_easy/sim_easy_em_app
./build/sim_easy/sim_easy_boem_app

python3 sim_easy_eval.py
