// This file generates the ground truth data and the dead reckoning data
// For dead reckoning, this file uses IMU data directly.


#include "sim.h"


int main(int argc, char **argv) {

  std::cout << "generate ground truth and dead reckoning trajectories..." << std::endl;

  google::InitGoogleLogging(argv[0]);

  ExpLandmarkSLAM slam_problem("config/config_sim.yaml");

  slam_problem.CreateTrajectory();
  slam_problem.CreateLandmark();

  slam_problem.CreateImuData();
  slam_problem.CreateObservationData();

  slam_problem.OutputGroundtruth("result/sim/gt.csv");

  slam_problem.InitializeSLAMProblem();
  slam_problem.InitializeTrajectory();
  slam_problem.OutputResult("result/sim/dr.csv");

  return 0;
}