// This file generates the ground truth data and the dead reckoning data
// For dead reckoning, this file uses IMU data directly.


#include "sim.h"


int main(int argc, char **argv) {

  std::cout << "generate ground truth and dead reckoning trajectories..." << std::endl;
  
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  Eigen::Rand::Vmt19937_64 urng{ (unsigned int) time(0) };

  ExpLandmarkSLAM slam_problem(FLAGS_duration, "config/config_sim.yaml");

  slam_problem.CreateTrajectory();
  slam_problem.CreateLandmark(urng);

  slam_problem.CreateImuData(urng);
  slam_problem.CreateObservationData(urng);

  slam_problem.OutputGroundtruth("result/sim/easy/gt.csv");

  slam_problem.InitializeSLAMProblem();
  slam_problem.InitializeTrajectory();
  slam_problem.OutputResult("result/sim/easy/dr.csv");

  return 0;
}