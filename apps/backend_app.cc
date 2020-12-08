
#include <cassert>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <opencv2/core/core.hpp>    // for config file reading
#include <Eigen/Core>


#include "so3.h"
#include "vec_3d_parameter_block.h"
#include "quat_parameter_block.h"
#include "imu_data.h"
#include "imu_error.h"
#include "pre_int_imu_error.h"
#include "reprojection_error.h"



// TODO: move this term to somewhere else
Eigen::Vector3d gravity = Eigen::Vector3d(0, 0, -9.81007);      


struct ObservationData {
 EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ObservationData(std::string observation_data_str) {
    std::stringstream str_stream(observation_data_str);          // Create a stringstream of the current line

    if (str_stream.good()) {
        
      std::string data_str;
      std::getline(str_stream, data_str, ',');   // get first string delimited by comma
      timestamp_ = std::stod(data_str)*1e-9;

      std::getline(str_stream, data_str, ','); 
      index_ = std::stoi(data_str);

      for (int i=0; i<2; ++i) {                    
        std::getline(str_stream, data_str, ','); 
        feature_pos_(i) = std::stod(data_str);
      }

      std::getline(str_stream, data_str, ','); 
      size_ = std::stod(data_str);
    }
  }

  Eigen::Matrix2d cov() {
    double sigma_2 = size_ * size_ / 64.0;
    return sigma_2 * Eigen::Matrix2d::Identity();
  }

  double timestamp_;
  size_t index_;
  Eigen::Vector2d feature_pos_; 
  double size_;
};


// This class only handles the memory. The underlying estimate is handled by each parameter blocks.
class State {

 public:
  State(double timestamp) {
    timestamp_ = timestamp;

    rotation_block_ptr_ = new QuatParameterBlock();
    velocity_block_ptr_ = new Vec3dParameterBlock();
    position_block_ptr_ = new Vec3dParameterBlock();
  }

  ~State() {
    delete [] rotation_block_ptr_;
    delete [] velocity_block_ptr_;
    delete [] position_block_ptr_;
  }

  double GetTimestamp() {
    return timestamp_;
  }

  QuatParameterBlock* GetRotationBlock() {
    return rotation_block_ptr_;
  }

  Vec3dParameterBlock* GetVelocityBlock() {
    return velocity_block_ptr_;
  }

  Vec3dParameterBlock* GetPositionBlock() {
    return position_block_ptr_;
  }

  void SetGyroBias(Eigen::Vector3d gyr_bias) {
    gyr_bias_ = gyr_bias;
  }

  void SetAcceBias(Eigen::Vector3d acc_bias) {
    acc_bias_ = acc_bias;
  }

  Eigen::Vector3d GetGyroBias() {
    return gyr_bias_;
  }

  Eigen::Vector3d GetAcceBias() {
    return acc_bias_;
  }

 private:
  double timestamp_;
  QuatParameterBlock* rotation_block_ptr_;
  Vec3dParameterBlock* velocity_block_ptr_;
  Vec3dParameterBlock* position_block_ptr_;

  Eigen::Vector3d gyr_bias_;
  Eigen::Vector3d acc_bias_; 
};

class ExpLandmarkOptSLAM {
 EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 public:

  ExpLandmarkOptSLAM(std::string config_folder_path) {
    ReadConfigurationFiles(config_folder_path);

    quat_parameterization_ptr_ = new ceres::QuaternionParameterization();
    loss_function_ptr_ = new ceres::HuberLoss(1.0);
  }

  bool ReadConfigurationFiles(std::string config_folder_path) {
    std::cout << "read config file from " << config_folder_path + "config_fpga_p2_euroc.yaml" << std::endl;

    // experiment configuration file
    cv::FileStorage experiment_config_file(config_folder_path + "config_fpga_p2_euroc.yaml", cv::FileStorage::READ);

    imu_dt_ = 1.0 / (double) experiment_config_file["imu_params"]["imu_rate"]; 

    // camera extrinsic
    cv::FileNode T_BC_node = experiment_config_file["cameras"][0]["T_SC"];            // from camera frame to body frame
    T_bc_  <<  T_BC_node[0],  T_BC_node[1],  T_BC_node[2],  T_BC_node[3], 
               T_BC_node[4],  T_BC_node[5],  T_BC_node[6],  T_BC_node[7], 
               T_BC_node[8],  T_BC_node[9], T_BC_node[10], T_BC_node[11], 
              T_BC_node[12], T_BC_node[13], T_BC_node[14], T_BC_node[15];

    fu_ = experiment_config_file["cameras"][0]["focal_length"][0];
    fv_ = experiment_config_file["cameras"][0]["focal_length"][1];

    sigma_g_c_ = experiment_config_file["imu_params"]["sigma_g_c"];
    sigma_a_c_ = experiment_config_file["imu_params"]["sigma_a_c"];    

    return true;
  }

  bool ReadInitialCondition() {
    std::cout << "read initial condition" << std::endl;


    // Step 1. red out_kf_time.csv to construct the state
    std::string kf_time_file_path("config/out_kf_time.csv");
    std::cout << "Read keyframe time data at " << kf_time_file_path << std::endl;
    std::ifstream kf_time_file(kf_time_file_path);
    assert(("Could not open ground truth file.", kf_time_file.is_open()));

    std::string gt_timestamp_start = "1403636580838555648";
    std::string gt_timestamp_end =   "1403636762743555584";

    // ignore the header
    std::string line;
    std::getline(kf_time_file, line);

    while (std::getline(kf_time_file, line)) {
      std::stringstream s_stream(line);                // Create a stringstream of the current line

      if (s_stream.good()) {
        std::string time_stamp_str;
        std::getline(s_stream, time_stamp_str, ',');   // get first string delimited by comma
      
        if (gt_timestamp_start <= time_stamp_str && time_stamp_str <= gt_timestamp_end) {

          // state
          state_parameter_.push_back(new State(std::stod(time_stamp_str)*1e-9));
        }
      }
    }

    kf_time_file.close();


    // Step 2. read out_state.csv to initialize state
    std::string state_init_file_path("config/out_state.csv");
    std::cout << "Read initial state estimate data at " << state_init_file_path << std::endl;
    std::ifstream state_init_file(state_init_file_path);
    assert(("Could not open ground truth file.", state_init_file.is_open()));

    // ignore the header
    std::getline(state_init_file, line);

    size_t i=0;
    while (std::getline(state_init_file, line)) {
      std::stringstream s_stream(line);                // Create a stringstream of the current line

      if (s_stream.good()) {
        std::string time_stamp_str;
        std::getline(s_stream, time_stamp_str, ',');   // get first string delimited by comma
          
        if (i<state_parameter_.size() && std::stod(time_stamp_str)*1e-9 == state_parameter_.at(i)->GetTimestamp()) {
          
          std::string temp_str;
          
          // rotation
          double init_q[4];
          for (int j=0; j<4; ++j) {                    
            std::getline(s_stream, temp_str, ',');
            init_q[j] = std::stod(temp_str);
          }

          Eigen::Quaterniond init_rotation(init_q[0], init_q[1], init_q[2], init_q[3]);
          state_parameter_.at(i)->GetRotationBlock()->setEstimate(init_rotation);
          optimization_problem_.AddParameterBlock(state_parameter_.at(i)->GetRotationBlock()->parameters(), 4, quat_parameterization_ptr_);


          // velocity
          double init_v[3];
          for (int j=0; j<3; ++j) {                    
            std::getline(s_stream, temp_str, ',');
            init_v[j] = std::stod(temp_str);
          }

          Eigen::Vector3d init_velocity(init_v);
          state_parameter_.at(i)->GetVelocityBlock()->setEstimate(init_velocity);
          optimization_problem_.AddParameterBlock(state_parameter_.at(i)->GetVelocityBlock()->parameters(), 3);


          // position
          double init_p[3];
          for (int j=0; j<3; ++j) {                    
            std::getline(s_stream, temp_str, ',');
            init_p[j] = std::stod(temp_str);
          }

          Eigen::Vector3d init_position(init_p);
          state_parameter_.at(i)->GetPositionBlock()->setEstimate(init_position);
          optimization_problem_.AddParameterBlock(state_parameter_.at(i)->GetPositionBlock()->parameters(), 3);

          // gyro bias
          double gyro_bias[3];
          for (int j=0; j<3; ++j) {                    
            std::getline(s_stream, temp_str, ',');
            gyro_bias[j] = std::stod(temp_str);
          }

          state_parameter_.at(i)->SetGyroBias(Eigen::Vector3d(gyro_bias));

          // acce bias
          double acce_bias[3];
          for (int j=0; j<3; ++j) {                    
            std::getline(s_stream, temp_str, ',');
            acce_bias[j] = std::stod(temp_str);
          }

          state_parameter_.at(i)->SetAcceBias(Eigen::Vector3d(acce_bias));
        
          i++;
        }
      }
    }

    state_init_file.close();


    return true;
  }


  bool TestStateParameter(std::ostream& output_stream) {

    double gt_timestamp_start = 1403636580.838555648;


    for (size_t i=0; i<state_parameter_.size(); ++i) {
      output_stream << i << " "
                    << state_parameter_.at(i)->GetTimestamp() - gt_timestamp_start << " "
                    << state_parameter_.at(i)->GetRotationBlock()->estimate().w() << " "
                    << state_parameter_.at(i)->GetRotationBlock()->estimate().x() << " "
                    << state_parameter_.at(i)->GetRotationBlock()->estimate().y() << " "
                    << state_parameter_.at(i)->GetRotationBlock()->estimate().z() << " "
                    << state_parameter_.at(i)->GetVelocityBlock()->estimate()[0] << " "
                    << state_parameter_.at(i)->GetVelocityBlock()->estimate()[1] << " "
                    << state_parameter_.at(i)->GetVelocityBlock()->estimate()[2] << " "
                    << state_parameter_.at(i)->GetPositionBlock()->estimate()[0] << " "
                    << state_parameter_.at(i)->GetPositionBlock()->estimate()[1] << " "
                    << state_parameter_.at(i)->GetPositionBlock()->estimate()[2] << " "
                    << state_parameter_.at(i)->GetGyroBias()[0] << " "
                    << state_parameter_.at(i)->GetGyroBias()[1] << " "
                    << state_parameter_.at(i)->GetGyroBias()[2] << " "
                    << state_parameter_.at(i)->GetAcceBias()[0] << " "
                    << state_parameter_.at(i)->GetAcceBias()[1] << " "
                    << state_parameter_.at(i)->GetAcceBias()[2] << std::endl;

    }
    return true;
  }

 private:
  // testing parameters
  double time_begin_;
  double time_end_;

  // experiment parameters
  double imu_dt_;

  Eigen::Matrix4d T_bc_;

  double fu_;
  double fv_;

  // parameter containers
  std::vector<State*>                state_parameter_;
  std::vector<Vec3dParameterBlock*>  landmark_parameter_;

  std::vector<ObservationData>       observation_data_;


  double sigma_g_c_;   // gyro noise density [rad/s/sqrt(Hz)]
  double sigma_a_c_;   // accelerometer noise density [m/s^2/sqrt(Hz)]



  // ceres parameter
  ceres::LocalParameterization* quat_parameterization_ptr_;
  ceres::LossFunction* loss_function_ptr_;

  ceres::Problem optimization_problem_;
  ceres::Solver::Options optimization_options_;
  ceres::Solver::Summary optimization_summary_;
};






int main(int argc, char **argv) {

  google::InitGoogleLogging(argv[0]);

  std::string config_folder_path("config/");
  ExpLandmarkOptSLAM slam_problem(config_folder_path);

  std::string euroc_dataset_path = "../../../dataset/mav0/";

  // initialize the first state
  slam_problem.ReadInitialCondition();
  slam_problem.TestStateParameter(std::cout);

  /***

  // states are constructed here
  std::string observation_file_path = "feature_observation.csv";
  slam_problem.ReadObservationData(observation_file_path);

  // output ground truth data (for comparison)
  slam_problem.ProcessGroundTruth(ground_truth_file_path);

  // setup IMU constraints
  std::string imu_file_path = euroc_dataset_path + "imu0/data.csv";
  slam_problem.ReadIMUData(imu_file_path);

  // initiate landmark estimate
  slam_problem.Triangulate();


  // the result before optimization (for comparison)
  slam_problem.OutputOptimizationResult("trajectory_dr.csv");

  slam_problem.SolveOptimizationProblem();
  slam_problem.OutputOptimizationResult("trajectory.csv");
  ***/

  return 0;
}