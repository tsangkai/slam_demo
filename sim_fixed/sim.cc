// This file generates the ground truth data and the dead reckoning data
// For dead reckoning, this file uses IMU data directly.


#include "sim.h"


int main(int argc, char **argv) {
//  srand((unsigned int) time(NULL)); //eigen uses the random number generator of the standard lib

  std::cout << "generate ground truth and dead reckoning trajectories..." << std::endl;


  int num_real = atoi(argv[1]);
  for (size_t i = 0; i < num_real; ++i) {

    Eigen::Rand::Vmt19937_64 urng{ i };


    ExpLandmarkSLAM slam_problem("config/config_sim.yaml");
    slam_problem.CreateTrajectory();
    slam_problem.CreateLandmark(urng);
    slam_problem.CreateImuData(urng);
    slam_problem.CreateObservationData(urng);
    slam_problem.InitializeSLAMProblem();
    slam_problem.InitializeTrajectory();
    slam_problem.OutputResult("result/sim_fixed/dr_"+ std::to_string(i) + ".csv");
    std::cout << "Completed DR trial " << std::to_string(i) << std::endl;
    slam_problem.OutputGroundtruth("result/sim_fixed/gt.csv");
  }

  return 0;
}