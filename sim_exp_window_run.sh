#!/bin/bash

# parameters
TIME_DIFF="15"
NUM_WIN="3" #"7"
TRIALS="5" #"20"

let TOTAL_DURATION=($NUM_WIN*$TIME_DIFF)


rm -f result/sim/exp_window/*

# ground truth
./bin/sim_exp_window_app -duration=$TOTAL_DURATION

for (( i=0; i<$NUM_WIN; i++ ))
do
  let DURATION=($i+1)*$TIME_DIFF

  perf stat -r $TRIALS ./bin/sim_exp_window_opt_app  -duration=$DURATION |& tee -a result/sim/exp_window/perf_opt.txt
  perf stat -r $TRIALS ./bin/sim_exp_window_em_app   -duration=$DURATION |& tee -a result/sim/exp_window/perf_em.txt
  perf stat -r $TRIALS ./bin/sim_exp_window_boem_app -duration=$DURATION |& tee -a result/sim/exp_window/perf_boem.txt
done


# visualization
python eval/sim_exp_window_eval.py $TRIALS $TIME_DIFF $NUM_WIN

