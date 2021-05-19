#!/bin/bash

dataset_list=( MH_01 MH_02 MH_03 MH_04 MH_05 )


for dataset in ${dataset_list[*]} 
do
  # ./build/exp/frontend_app ${dataset}


  # ./build/exp/backend_opt_app ${dataset}
  # ./build/exp/traj_process_app ${dataset} traj_vo
  # ./build/exp/traj_process_app ${dataset} traj_opt

  ./build/exp/backend_em_app ${dataset}
  ./build/exp/traj_process_app ${dataset} traj_em

  # ./build/exp/backend_boem_app ${dataset}
  # ./build/exp/traj_process_app ${dataset} traj_boem

  python3 exp_eval.py ${dataset}  

  read -p "press key"
done

