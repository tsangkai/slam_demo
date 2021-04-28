#!/bin/bash

dataset_list=( MH_01 MH_02 MH_03 MH_04 MH_05 )


for dataset in ${dataset_list[*]} 
do
  ./build/apps/backend_opt_app ${dataset}
  # read -p "press key"
  ./build/apps/backend_em_app ${dataset}
  # read -p "press key"
  ./build/apps/backend_boem_app ${dataset}
  # read -p "press key"
done


for dataset in ${dataset_list[*]} 
do
  ./build/apps/traj_process_app ${dataset} traj_opt
  ./build/apps/traj_process_app ${dataset} traj_em
  ./build/apps/traj_process_app ${dataset} traj_boem
done


for dataset in ${dataset_list[*]} 
do
  python3 eval_exp.py ${dataset}
done