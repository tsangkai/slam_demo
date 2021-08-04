#!/bin/bash

# parameters
TIME_DIFF_VAR="15"
NUM_WIN="3" #"7"
REPEAT_VAR_WIN="5" #"20"

rm -f result/sim_exp_window/*

# ground truth
let TOTAL_TIME_VAR=($NUM_WIN*$TIME_DIFF_VAR)
./build/sim_exp_window/sim_exp_win_app $TOTAL_TIME_VAR


for (( i=0; i<$NUM_WIN; i++ ))
do
  let WINDOW_VAR=($i+1)*$TIME_DIFF_VAR

  perf stat -r $REPEAT_VAR_WIN ./build/sim_exp_window/sim_exp_win_opt_app $WINDOW_VAR |& tee -a result/sim_exp_window/perf_opt.txt
  perf stat -r $REPEAT_VAR_WIN ./build/sim_exp_window/sim_exp_win_em_app $WINDOW_VAR |& tee -a result/sim_exp_window/perf_em.txt
  perf stat -r $REPEAT_VAR_WIN ./build/sim_exp_window/sim_exp_win_boem_app $WINDOW_VAR |& tee -a result/sim_exp_window/perf_boem.txt
done


# visualization
python3 sim_exp_window_eval.py $REPEAT_VAR_WIN $TIME_DIFF_VAR $NUM_WIN

