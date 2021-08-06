
#include <iostream>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Core>

#include "sim.h"


int main(int argc, char **argv) {

    std::cout << "simulate ground truth ..." << std::endl;

    Eigen::Rand::Vmt19937_64 urng{ (unsigned int) time(0) };

    double time_win = std::stod(argv[1]);

    ExpLandmarkSLAM slam_problem(time_win,"config/config_sim.yaml");

    slam_problem.CreateTrajectory();

    slam_problem.OutputGroundtruth("result/sim/exp_window/gt.csv");

    return 0;
}