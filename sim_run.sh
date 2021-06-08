#!/bin/bash

REPEAT_VAR="50"

./build/sim/sim_vis/sim_vis $REPEAT_VAR
./build/sim/sim_vis/sim_opt_vis $REPEAT_VAR
./build/sim/sim_vis/sim_em_vis $REPEAT_VAR
./build/sim/sim_vis/sim_boem_vis $REPEAT_VAR

#./build/sim/sim_app
#./build/sim/sim_opt_app
#./build/sim/sim_em_app
#./build/sim/sim_boem_app

#REPEAT_VAR="2"
#perf stat -r $REPEAT_VAR ./build/sim/sim_boem_app


#python sim_eval.py
python eval_sim_fig2.py $REPEAT_VAR