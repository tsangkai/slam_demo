# SLAM Demo

This project provides the simulation and experiment results for the ICRA 2022 submission "BLock online EM algorithm for visual-inertial SLAM backend". This project focuses on the backend of SLAM systems. In particular, we investigate both optimization-based and EM-based algorithms. For experiments, we use [EuRoC dataset](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets#the_euroc_mav_dataset).

## Required packages

You will need to install the following dependencies,

* OpenCV
* CMake
* Eigen3        
* ceres

## Build instruction

To build this project, just follow the standard CMake procedure.
```
mkdir build
cd build
cmake ..
cmake --build .
```

## Usage

### Testing

The files in `test` folder ensure that the functionality in the implemented SLAM backend in corrected.

### Simulations

There are 3 main simulations that we present in the paper:
- `sim_easy_run.sh`: Fig. 1. The 3D trajectory plot
- `sim_fixed_run.sh`: Fig. 2. The estimation error plot
- `sim_exp_window_run.sh`: Fig. 3. The estimation accuracy and the processing time plot

The setup parameters for the simulation are in `config/config_sim.yaml`. The parameters for running the simulation, for example the duration and the number of trials, are in each bash script.

### Experiments on EuRoC datasets

The experiment is conducted by `exp_run.sh`, with parameters provided by the datasets in `config/config_fpga_p2_euroc.yaml`. The experiments in this project use the raw data from the images and the IMU sensors, as well as from the estimation trajectory from the frontend. Both kinds of data are preprocessed, and stored in `data`.