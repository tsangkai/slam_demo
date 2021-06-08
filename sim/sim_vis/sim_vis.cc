// This file generates the ground truth data and the dead reckoning data
// For dead reckoning, this file uses IMU data directly.


#include "sim.h"


int main(int argc, char **argv) {
//  srand((unsigned int) time(NULL)); //eigen uses the random number generator of the standard lib

  std::cout << "generate ground truth and dead reckoning trajectories..." << std::endl;

  google::InitGoogleLogging(argv[0]);
  int num_real = atoi(argv[1]);
  for (size_t i = 0; i < num_real; ++i) {
    ExpLandmarkSLAM slam_problem("config/config_sim.yaml");
    slam_problem.CreateTrajectory();
    slam_problem.CreateLandmark();
    slam_problem.CreateImuData();
    slam_problem.CreateObservationData();
    slam_problem.InitializeSLAMProblem();
    slam_problem.InitializeTrajectory();
    slam_problem.OutputResult("result/sim/dr_"+ std::to_string(i) + ".csv");
    std::cout << "Completed DR trial " << std::to_string(i) << std::endl;
    slam_problem.OutputGroundtruth("result/sim/gt.csv");
  }

  return 0;
}