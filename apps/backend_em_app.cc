
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
#include <boost/date_time/posix_time/posix_time.hpp>


#include "so3.h"
#include "vec_3d_parameter_block.h"
#include "quat_parameter_block.h"
#include "triangularization.h"
#include "imu_data.h"
#include "imu_error.h"
#include "pre_int_imu_error.h"
#include "reprojection_error.h"   

// TODO: define globally
Eigen::Vector3d gravity = Eigen::Vector3d(0, 0, -9.81007);

struct ObservationData {
 EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ObservationData(double timestamp) {
    timestamp_ = timestamp;
  }

  Eigen::Matrix2d cov() {
    double sigma_2 = size_ * size_ / 64.0;
    return sigma_2 * Eigen::Matrix2d::Identity();
  }

  double timestamp_;
  size_t landmark_id_;
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

  Eigen::Matrix<double, 9, 9> cov_;  
};


// 
struct Estimate {
  Eigen::Quaterniond q_;  
  Eigen::Vector3d v_;
  Eigen::Vector3d p_; 

  Eigen::Matrix<double, 9, 9> cov_;
};


class ExpLandmarkEmSLAM {
 EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 public:

  ExpLandmarkEmSLAM(std::string config_folder_path) {
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

    cu_ = experiment_config_file["cameras"][0]["principal_point"][0];
    cv_ = experiment_config_file["cameras"][0]["principal_point"][1];

    sigma_g_c_ = experiment_config_file["imu_params"]["sigma_g_c"];
    sigma_a_c_ = experiment_config_file["imu_params"]["sigma_a_c"];    

    return true;
  }



  // Due to the implementation of okvis, we first read the time stamp of keyframes,
  // and then obtain state estimate from the visual odometry at keyframes.
  bool ReadInitialTraj(std::string data_folder_path) {
    std::cout << "read initial trajectory" << std::endl;


    // Step 1. red out_kf_time.csv to construct the state
    std::string kf_time_file_path(data_folder_path + "out_kf_time.csv");
    std::cout << "Read keyframe time data at " << kf_time_file_path << std::endl;
    std::ifstream kf_time_file(kf_time_file_path);
    assert(("Could not open keyframe time file.", kf_time_file.is_open()));

    // TODO
    std::string gt_timestamp_start = "1403636624463555584";
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
          state_vec_.push_back(new State(std::stod(time_stamp_str)*1e-9));
        }
      }
    }

    kf_time_file.close();


    // Step 2. read out_state.csv to initialize state
    std::string state_init_file_path(data_folder_path + "out_state.csv");
    std::cout << "Read initial state estimate data at " << state_init_file_path << std::endl;
    std::ifstream state_init_file(state_init_file_path);
    assert(("Could not open initial state estimate file.", state_init_file.is_open()));

    // ignore the header
    std::getline(state_init_file, line);

    size_t i=0;
    while (std::getline(state_init_file, line)) {
      std::stringstream str_stream(line);                // Create a stringstream of the current line

      if (str_stream.good()) {
        std::string time_stamp_str;
        std::getline(str_stream, time_stamp_str, ',');   // get first string delimited by comma
          
        if (i<state_vec_.size() && std::stod(time_stamp_str)*1e-9 == state_vec_.at(i)->GetTimestamp()) {
          
          std::string temp_str;
          
          // rotation
          double init_q[4];
          for (int j=0; j<4; ++j) {                    
            std::getline(str_stream, temp_str, ',');
            init_q[j] = std::stod(temp_str);
          }

          Eigen::Quaterniond init_rotation(init_q[0], init_q[1], init_q[2], init_q[3]);
          state_vec_.at(i)->GetRotationBlock()->setEstimate(init_rotation);

          // velocity
          double init_v[3];
          for (int j=0; j<3; ++j) {                    
            std::getline(str_stream, temp_str, ',');
            init_v[j] = std::stod(temp_str);
          }

          Eigen::Vector3d init_velocity(init_v);
          state_vec_.at(i)->GetVelocityBlock()->setEstimate(init_velocity);

          // position
          double init_p[3];
          for (int j=0; j<3; ++j) {                    
            std::getline(str_stream, temp_str, ',');
            init_p[j] = std::stod(temp_str);
          }

          Eigen::Vector3d init_position(init_p);
          state_vec_.at(i)->GetPositionBlock()->setEstimate(init_position);

          // gyro bias
          double gyro_bias[3];
          for (int j=0; j<3; ++j) {                    
            std::getline(str_stream, temp_str, ',');
            gyro_bias[j] = std::stod(temp_str);
          }

          state_vec_.at(i)->SetGyroBias(Eigen::Vector3d(gyro_bias));

          // acce bias
          double acce_bias[3];
          for (int j=0; j<3; ++j) {                    
            std::getline(str_stream, temp_str, ',');
            acce_bias[j] = std::stod(temp_str);
          }

          state_vec_.at(i)->SetAcceBias(Eigen::Vector3d(acce_bias));
        
          i++;
        }
      }
    }

    state_init_file.close();


    return true;
  }


  // constrcut landmark position vector and observation vector
  bool ReadObservationData(std::string data_folder_path) {
  
    std::string feature_obs_file_path(data_folder_path + "feature_obs.csv");
    std::cout << "Read feature observation data at " << feature_obs_file_path << std::endl;
    std::ifstream feature_obs_file(feature_obs_file_path);
    assert(("Could not open observation data file.", feature_obs_file.is_open()));

    observation_vec_.resize(state_vec_.size()-1);

    // ignore the header
    std::string line;
    std::getline(feature_obs_file, line);

    while (std::getline(feature_obs_file, line)) {

      std::stringstream str_stream(line);

      if (str_stream.good()) {
        
        std::string data_str;
        std::getline(str_stream, data_str, ',');   // get first string delimited by comma
        double timestamp = std::stod(data_str)*1e-9;

        size_t state_idx = state_vec_.size();
        for (size_t j=1; j<state_vec_.size(); ++j) {
          if (state_vec_.at(j)->GetTimestamp() == timestamp) {
            state_idx = j;
            break;
          }
        }

        if (state_idx < state_vec_.size()) {
          ObservationData* feature_obs_ptr = new ObservationData(timestamp);
          observation_vec_.at(state_idx-1).push_back(feature_obs_ptr);

          // landmark id
          std::getline(str_stream, data_str, ','); 
          feature_obs_ptr->landmark_id_ = std::stoi(data_str);

          // u, v
          for (int i=0; i<2; ++i) {                    
            std::getline(str_stream, data_str, ','); 
            feature_obs_ptr->feature_pos_(i) = std::stod(data_str);
          }

          // size
          std::getline(str_stream, data_str, ','); 
          feature_obs_ptr->size_ = std::stod(data_str);          


          // resize landmark vector
          size_t landmark_id = feature_obs_ptr->landmark_id_-1;
          if (landmark_id >= landmark_vec_.size()) {
            landmark_vec_.resize(landmark_id+1);
          }
        }
      }
    }


    for (size_t i=0; i<landmark_vec_.size(); ++i) {
      landmark_vec_.at(i) = new Vec3dParameterBlock();
    }

    feature_obs_file.close();


    // debug
    /***
    size_t num_obs = 0;
    for (size_t j=1; j<state_vec_.size(); ++j) {
      std::cout << state_vec_.at(j)->GetTimestamp() - 1.40363e+09 << " ";

      if (observation_vec_.at(j-1).empty()) {
        std::cout << std::endl;
      }
      else {
        std::cout << observation_vec_.at(j-1).at(0)->timestamp_ - 1.40363e+09 << " " << observation_vec_.at(j-1).size() << std::endl;
        num_obs += observation_vec_.at(j-1).size();
      }
    }
    ***/

    std::cout << "numober of states: " << state_vec_.size() << std::endl;
    std::cout << "numober of observation vec: " << observation_vec_.size() << std::endl;
    std::cout << "numober of landmarks: " << landmark_vec_.size() << std::endl;
    // std::cout << "numober of feature observations: " << num_obs << std::endl;



    // initialize landmark by triangularization
    std::vector<std::vector<TriangularData>> tri_data;        // doesn't have to be a member!?
    tri_data.resize(landmark_vec_.size());

    for (size_t i=0; i<observation_vec_.size(); ++i) {
      for (size_t j=0; j<observation_vec_.at(i).size(); ++j) {

        size_t landmark_idx = observation_vec_.at(i).at(j)->landmark_id_-1;  

        TriangularData tri_data_instance(observation_vec_.at(i).at(j)->feature_pos_,
                                       state_vec_.at(i+1)->GetRotationBlock()->estimate(),
                                       state_vec_.at(i+1)->GetPositionBlock()->estimate());
        tri_data.at(landmark_idx).push_back(tri_data_instance);
      }
    }    

    // set landmark initial estimate
    for (size_t i=0; i<landmark_vec_.size(); ++i) {

      size_t idx_0 = 0;
      size_t idx_1 = std::min(size_t(1), tri_data.at(i).size()-1);

      Eigen::Vector3d init_landmark_pos = EpipolarInitialize(tri_data.at(i).at(idx_0).keypoint_, 
                                                             tri_data.at(i).at(idx_0).rotation_, 
                                                             tri_data.at(i).at(idx_0).position_,
                                                             tri_data.at(i).at(idx_1).keypoint_, 
                                                             tri_data.at(i).at(idx_1).rotation_, 
                                                             tri_data.at(i).at(idx_1).position_,
                                                             T_bc_, fu_, fv_, cu_, cv_);

      landmark_vec_.at(i)->setEstimate(init_landmark_pos);
    }

    // debug
    /***
    for (size_t i=0; i<landmark_vec_.size(); ++i) {

      std::cout << i << " "
                << landmark_vec_.at(i)->estimate()[0] << " "
                << landmark_vec_.at(i)->estimate()[1] << " "
                << landmark_vec_.at(i)->estimate()[2] << std::endl;
    }
    ***/

    return true;
  }





  bool ReadImuData(std::string imu_file_path) {
  
    std::cout << "Read IMU data at " << imu_file_path << std::endl;

    std::ifstream imu_file(imu_file_path);
    assert(("Could not open IMU file.", imu_file.is_open()));

    // Read the column names
    // Extract the first line in the file
    std::string first_line_data_str;
    std::getline(imu_file, first_line_data_str);

    size_t state_idx = 0;                 // the index of the last element

    PreIntIMUData* int_imu_data_ptr = new PreIntIMUData(state_vec_.at(state_idx)->GetGyroBias(),
                                                        state_vec_.at(state_idx)->GetAcceBias(),
                                                        sigma_g_c_,
                                                        sigma_a_c_);

    std::string imu_data_str;
    while (std::getline(imu_file, imu_data_str)) {


      double timestamp;
      Eigen::Vector3d gyr;
      Eigen::Vector3d acc;

      std::stringstream imu_str_stream(imu_data_str); 

      if (imu_str_stream.good()) {
        std::string data_str;
        std::getline(imu_str_stream, data_str, ','); 
        timestamp = std::stod(data_str)*1e-9;

        for (int i=0; i<3; ++i) { 
          std::getline(imu_str_stream, data_str, ','); 
          gyr(i) = std::stod(data_str);
        }

        for (int i=0; i<3; ++i) {                    
          std::getline(imu_str_stream, data_str, ','); 
          acc(i) = std::stod(data_str);
        }
      }

      IMUData imu_data(timestamp, gyr, acc);

      double time_begin = 1403636624.463555584;
      double time_end = 1403636762.743555584;

      
      
      if (time_begin <= imu_data.timestamp_ && imu_data.timestamp_ <= time_end) {


        // starting to put imu data in the previously established state_parameter_
        // case 1: the time stamp of the imu data is after the last state
        if ((state_idx + 1) == state_vec_.size()) {
          int_imu_data_ptr->IntegrateSingleIMU(imu_data, imu_dt_);
        }
        // case 2: the time stamp of the imu data is between two consecutive states
        else if (imu_data.timestamp_ < state_vec_.at(state_idx+1)->GetTimestamp()) {
          int_imu_data_ptr->IntegrateSingleIMU(imu_data, imu_dt_);
        }
        // case 3: the imu data just enter the new interval of integration
        else {

          pre_int_imu_vec_.push_back(int_imu_data_ptr);

          state_idx++;

          // prepare for next iteration
          int_imu_data_ptr = new PreIntIMUData(state_vec_.at(state_idx)->GetGyroBias(),
                                               state_vec_.at(state_idx)->GetAcceBias(),
                                               sigma_g_c_,
                                               sigma_a_c_);

          int_imu_data_ptr->IntegrateSingleIMU(imu_data, imu_dt_);

        }
      }
    }

    imu_file.close();

    std::cout << "Finished reading IMU data." << std::endl;
    return true;
  }



  bool SetupMStep() {

    // add parameter blocks
    for (size_t i=0; i<landmark_vec_.size(); ++i) {
      optimization_problem_.AddParameterBlock(landmark_vec_.at(i)->parameters(), 3);
    }

    for (size_t i=0; i<state_vec_.size(); ++i) {
      optimization_problem_.AddParameterBlock(state_vec_.at(i)->GetRotationBlock()->parameters(), 4);
      optimization_problem_.AddParameterBlock(state_vec_.at(i)->GetVelocityBlock()->parameters(), 3);
      optimization_problem_.AddParameterBlock(state_vec_.at(i)->GetPositionBlock()->parameters(), 3);

      optimization_problem_.SetParameterBlockConstant(state_vec_.at(i)->GetRotationBlock()->parameters());
      optimization_problem_.SetParameterBlockConstant(state_vec_.at(i)->GetVelocityBlock()->parameters());
      optimization_problem_.SetParameterBlockConstant(state_vec_.at(i)->GetPositionBlock()->parameters());
    }
    

    for (size_t i=0; i<observation_vec_.size(); ++i) {
      for (size_t j=0; j<observation_vec_.at(i).size(); ++j) {

        size_t landmark_idx = observation_vec_.at(i).at(j)->landmark_id_-1;  

        ceres::CostFunction* cost_function = new ReprojectionError(observation_vec_.at(i).at(j)->feature_pos_,
                                                                   T_bc_,
                                                                   fu_, fv_,
                                                                   cu_, cv_,
                                                                   observation_vec_.at(i).at(j)->cov());

        optimization_problem_.AddResidualBlock(cost_function,
                                               loss_function_ptr_,
                                               state_vec_.at(i+1)->GetRotationBlock()->parameters(),
                                               state_vec_.at(i+1)->GetPositionBlock()->parameters(),
                                               landmark_vec_.at(landmark_idx)->parameters());
      }
    }


    return true;
  }


  bool SolveEmProblem() {

    std::cout << "Begin solving the EM problem." << std::endl;

    optimization_options_.linear_solver_type = ceres::SPARSE_SCHUR;
    optimization_options_.minimizer_progress_to_stdout = true;
    optimization_options_.num_threads = 6;
    optimization_options_.function_tolerance = 1e-20;
    optimization_options_.parameter_tolerance = 1e-25;
    optimization_options_.max_num_iterations = 100;



    // E step
    std::vector<Estimate*> state_estimate;
    state_estimate.resize(state_vec_.size()-1);

    // forward Kalman filter
    for (size_t i=0; i<state_estimate.size(); ++i) {
      state_estimate.at(i) = new Estimate;

      // time update
      Eigen::Quaterniond q0 = state_vec_.at(i)->GetRotationBlock()->estimate();
      Eigen::Vector3d v0 = state_vec_.at(i)->GetVelocityBlock()->estimate();
      Eigen::Vector3d p0 = state_vec_.at(i)->GetPositionBlock()->estimate();

      double u_dt = pre_int_imu_vec_.at(i)->dt_;
      Eigen::Matrix3d u_dR = pre_int_imu_vec_.at(i)->dR_;  
      Eigen::Vector3d u_dv = pre_int_imu_vec_.at(i)->dv_;  
      Eigen::Vector3d u_dp = pre_int_imu_vec_.at(i)->dp_;
      Eigen::Matrix<double, 9, 9> u_cov = pre_int_imu_vec_.at(i)->cov_;  

      Eigen::Quaterniond imu_q1 = q0 * Eigen::Quaterniond(u_dR);
      Eigen::Vector3d imu_v1 = v0 + q0.toRotationMatrix() * u_dv + gravity * u_dt;
      Eigen::Vector3d imu_p1 = p0 + u_dt*v0 + q0.toRotationMatrix() * u_dp + 0.5 * gravity * u_dt*u_dt;

      Eigen::Vector3d imu_dq = Log_q(state_vec_.at(i+1)->GetRotationBlock()->estimate().conjugate() * imu_q1);
      Eigen::Vector3d imu_dv = imu_v1 - state_vec_.at(i+1)->GetVelocityBlock()->estimate();
      Eigen::Vector3d imu_dp = imu_p1 - state_vec_.at(i+1)->GetPositionBlock()->estimate();

      double kf_constant = 0.2;
      state_estimate.at(i)->q_ = state_vec_.at(i+1)->GetRotationBlock()->estimate() * Exp_q(kf_constant * imu_dq);
      state_estimate.at(i)->v_ = state_vec_.at(i+1)->GetVelocityBlock()->estimate() + kf_constant * imu_dv;
      state_estimate.at(i)->p_ = state_vec_.at(i+1)->GetPositionBlock()->estimate() + kf_constant * imu_dp;

      /***
      state_estimate.at(i)->q_ = state_vec_.at(i+1)->GetRotationBlock()->estimate();
      state_estimate.at(i)->v_ = state_vec_.at(i+1)->GetVelocityBlock()->estimate();
      state_estimate.at(i)->p_ = state_vec_.at(i+1)->GetPositionBlock()->estimate();
      ***/
      

      Eigen::Matrix<double, 9, 9> F = Eigen::Matrix<double, 9, 9>::Zero();
      F.block<3,3>(0,0) = u_dR.transpose();
      F.block<3,3>(3,0) = (-1)*q0.toRotationMatrix()*Skew(u_dv);
      F.block<3,3>(3,0) = Eigen::Matrix3d::Identity();
      F.block<3,3>(6,0) = (-1)*q0.toRotationMatrix()*Skew(u_dp);
      F.block<3,3>(6,3) = u_dt*Eigen::Matrix3d::Identity();
      F.block<3,3>(6,6) = Eigen::Matrix3d::Identity();

      Eigen::Matrix<double, 9, 9> G = Eigen::Matrix<double, 9, 9>::Zero();
      G.block<3,3>(0,0) = Eigen::Matrix3d::Identity();
      G.block<3,3>(3,3) = q0.toRotationMatrix();
      G.block<3,3>(6,6) = q0.toRotationMatrix();

      if (i==0) {
        state_estimate.at(i)->cov_ = G * u_cov * G.transpose();
      }
      else {
        state_estimate.at(i)->cov_ = F * state_estimate.at(i-1)->cov_ * F.transpose() + G * u_cov * G.transpose();
      }

      // observation update
      Eigen::Matrix3d k_R = Eigen::Matrix3d::Identity();
      Eigen::Vector3d k_v = Eigen::Vector3d::Zero();
      Eigen::Vector3d k_p = Eigen::Vector3d::Zero();

      for (size_t j=0; j<observation_vec_.at(i).size(); ++j) {

        Eigen::Vector3d landmark = landmark_vec_.at(observation_vec_.at(i).at(j)->landmark_id_-1)->estimate();
        Eigen::Vector2d measurement = observation_vec_.at(i).at(j)->feature_pos_;
        Eigen::Matrix2d R = observation_vec_.at(i).at(j)->cov();

        Eigen::Matrix3d R_bc = T_bc_.topLeftCorner<3,3>();
        Eigen::Vector3d t_bc = T_bc_.topRightCorner<3,1>();

        Eigen::Matrix3d R_nb = state_estimate.at(i)->q_.toRotationMatrix();
        Eigen::Vector3d t_nb = state_estimate.at(i)->p_;

        Eigen::Vector3d landmark_c = R_bc.transpose() * ((R_nb.transpose()*(landmark - t_nb)) - t_bc);
        Eigen::Vector2d landmark_proj;
        landmark_proj << fu_ * landmark_c[0]/landmark_c[2] + cu_, 
                         fv_ * landmark_c[1]/landmark_c[2] + cv_;

        // exclude outliers
        Eigen::Vector2d innovation = measurement - landmark_proj;
        if (innovation.norm() < 80) {  // if the threshold is too small, no loop closure can occur

          Eigen::Matrix<double, 2, 2> H_cam;
          H_cam << fu_, 0.0,
                   0.0, fv_;

          Eigen::Matrix<double, 2, 3> H_proj;
          H_proj << 1.0/(landmark_c[2]), 0, -(landmark_c[0])/(landmark_c[2]*landmark_c[2]),
                    0, 1.0/(landmark_c[2]), -(landmark_c[1])/(landmark_c[2]*landmark_c[2]);

          Eigen::Matrix<double, 3, 9> H_trans;
          H_trans.setZero();
          H_trans.block<3,3>(0,0) = R_bc.transpose() * Skew(R_nb.transpose()*(landmark - t_nb));
          H_trans.block<3,3>(0,6) = (-1) * R_bc.transpose() * R_nb.transpose();

          Eigen::Matrix<double, 2, 9> H;
          H = H_cam * H_proj * H_trans;


          Eigen::Matrix<double, 9, 2> K;
          K = state_estimate.at(i)->cov_ * H.transpose() * (H * state_estimate.at(i)->cov_ * H.transpose() + R).inverse();
          Eigen::Matrix<double, 9, 1> m;
          m = K * (measurement - landmark_proj);

            k_R = k_R * Exp(m.block<3,1>(0,0));
            k_v = k_v + m.block<3,1>(3,0);
            k_p = k_p + m.block<3,1>(6,0);  

            Eigen::Matrix<double, 9, 9> IKH;
            IKH = Eigen::Matrix<double, 9, 9>::Identity() - K * H;
            state_estimate.at(i)->cov_ = IKH * state_estimate.at(i)->cov_ * IKH.transpose() + K * R * K.transpose();
          
        }
      }

      if (k_p.norm() < 0.65) {

        state_estimate.at(i)->q_ = state_estimate.at(i)->q_ * k_R;
        if (state_estimate.at(i)->q_.w() < 0) {
          state_estimate.at(i)->q_.w() = (-1)*state_estimate.at(i)->q_.w();
          state_estimate.at(i)->q_.x() = (-1)*state_estimate.at(i)->q_.x();
          state_estimate.at(i)->q_.y() = (-1)*state_estimate.at(i)->q_.y();
          state_estimate.at(i)->q_.z() = (-1)*state_estimate.at(i)->q_.z();
        }
        state_estimate.at(i)->v_ = state_estimate.at(i)->v_ + k_v;
        state_estimate.at(i)->p_ = state_estimate.at(i)->p_ + k_p;
      }
    }


    // backward RTS smoother
    for (int i=state_estimate.size()-2; i>-1; --i) {

      // std::cout << "RTS smoother: " << i << std::endl;

      Eigen::Quaterniond q0 = state_estimate.at(i)->q_;
      Eigen::Vector3d v0 = state_estimate.at(i)->v_;
      Eigen::Vector3d p0 = state_estimate.at(i)->p_;

      double u_dt = pre_int_imu_vec_.at(i+1)->dt_;
      Eigen::Matrix3d u_dR = pre_int_imu_vec_.at(i+1)->dR_;  
      Eigen::Vector3d u_dv = pre_int_imu_vec_.at(i+1)->dv_;  
      Eigen::Vector3d u_dp = pre_int_imu_vec_.at(i+1)->dp_;
      Eigen::Matrix<double, 9, 9> u_cov = pre_int_imu_vec_.at(i+1)->cov_;  

      Eigen::Matrix<double, 9, 9> F = Eigen::Matrix<double, 9, 9>::Zero();
      F.block<3,3>(0,0) = u_dR.transpose();
      F.block<3,3>(3,0) = (-1)*q0.toRotationMatrix()*Skew(u_dv);
      F.block<3,3>(3,0) = Eigen::Matrix3d::Identity();
      F.block<3,3>(6,0) = (-1)*q0.toRotationMatrix()*Skew(u_dp);
      F.block<3,3>(6,3) = u_dt*Eigen::Matrix3d::Identity();
      F.block<3,3>(6,6) = Eigen::Matrix3d::Identity();

      Eigen::Matrix<double, 9, 9> G = Eigen::Matrix<double, 9, 9>::Zero();
      G.block<3,3>(0,0) = Eigen::Matrix3d::Identity();
      G.block<3,3>(3,3) = q0.toRotationMatrix();
      G.block<3,3>(6,6) = q0.toRotationMatrix();

      Eigen::Quaterniond q1 = q0 * Eigen::Quaterniond(u_dR);
      Eigen::Vector3d v1 = v0 + q0.toRotationMatrix() * u_dv + gravity * u_dt;
      Eigen::Vector3d p1 = p0 + u_dt*v0 + q0.toRotationMatrix() * u_dp + 0.5 * gravity * u_dt*u_dt;

      Eigen::Matrix<double, 9, 9> C;
      C = state_estimate.at(i)->cov_ * F.transpose() * (F * state_estimate.at(i)->cov_ * F.transpose() + G * u_cov * G.transpose()).inverse();

      Eigen::Matrix<double, 9, 1> residual;
      residual.block<3,1>(0,0) = Log_q(q1.conjugate() * state_estimate.at(i+1)->q_);
      residual.block<3,1>(3,0) = state_estimate.at(i+1)->v_ - v1;
      residual.block<3,1>(6,0) = state_estimate.at(i+1)->p_ - p1;

      Eigen::Matrix<double, 9, 1> m;
      m = (0.3) * C * residual;  // give the IMU results less weight


      // std::cout << m << std::endl;
      // std::cin.get();

      state_estimate.at(i)->q_ = state_estimate.at(i)->q_ * Exp(m.block<3,1>(0,0));
      if (state_estimate.at(i)->q_.w() < 0) {
        state_estimate.at(i)->q_.w() = (-1)*state_estimate.at(i)->q_.w();
        state_estimate.at(i)->q_.x() = (-1)*state_estimate.at(i)->q_.x();
        state_estimate.at(i)->q_.y() = (-1)*state_estimate.at(i)->q_.y();
        state_estimate.at(i)->q_.z() = (-1)*state_estimate.at(i)->q_.z();
      }
      state_estimate.at(i)->v_ = state_estimate.at(i)->v_ + m.block<3,1>(3,0);
      state_estimate.at(i)->p_ = state_estimate.at(i)->p_ + m.block<3,1>(6,0);

      // ignore sigma update

    }

    // update the state estimate
    for (size_t i=0; i<state_estimate.size(); ++i) {
      state_vec_.at(i+1)->GetRotationBlock()->setEstimate(state_estimate.at(i)->q_);
      state_vec_.at(i+1)->GetVelocityBlock()->setEstimate(state_estimate.at(i)->v_);
      state_vec_.at(i+1)->GetPositionBlock()->setEstimate(state_estimate.at(i)->p_);
    }

    // M step
    ceres::Solve(optimization_options_, &optimization_problem_, &optimization_summary_);
    std::cout << optimization_summary_.FullReport() << "\n";
    std::cout << "Final cost " << optimization_summary_.final_cost << std::endl; // -1

    return true;
  }


  bool OutputOptimizationResult(std::string output_file_name) {

    std::ofstream output_file(output_file_name);

    output_file << "timestamp,p_x,p_y,p_z,v_x,v_y,v_z,q_w,q_x,q_y,q_z\n";

    for (size_t i=0; i<state_vec_.size(); ++i) {

      output_file << std::to_string(state_vec_.at(i)->GetTimestamp()) << ",";
      output_file << std::to_string(state_vec_.at(i)->GetPositionBlock()->estimate()(0)) << ",";
      output_file << std::to_string(state_vec_.at(i)->GetPositionBlock()->estimate()(1)) << ",";
      output_file << std::to_string(state_vec_.at(i)->GetPositionBlock()->estimate()(2)) << ",";
      output_file << std::to_string(state_vec_.at(i)->GetVelocityBlock()->estimate()(0)) << ",";
      output_file << std::to_string(state_vec_.at(i)->GetVelocityBlock()->estimate()(1)) << ",";
      output_file << std::to_string(state_vec_.at(i)->GetVelocityBlock()->estimate()(2)) << ",";
      output_file << std::to_string(state_vec_.at(i)->GetRotationBlock()->estimate().w()) << ",";
      output_file << std::to_string(state_vec_.at(i)->GetRotationBlock()->estimate().x()) << ",";
      output_file << std::to_string(state_vec_.at(i)->GetRotationBlock()->estimate().y()) << ",";
      output_file << std::to_string(state_vec_.at(i)->GetRotationBlock()->estimate().z()) << std::endl;
    }

    output_file.close();

    return true;
  }

 private:

  // experiment parameters
  double imu_dt_;
  double sigma_g_c_;   // gyro noise density [rad/s/sqrt(Hz)]
  double sigma_a_c_;   // accelerometer noise density [m/s^2/sqrt(Hz)]

  Eigen::Matrix4d T_bc_;
  double fu_;
  double fv_;
  double cu_;
  double cv_;


  // parameter containers
  std::vector<State*>                         state_vec_;
  std::vector<Vec3dParameterBlock*>           landmark_vec_;

  std::vector<PreIntIMUData*>                 pre_int_imu_vec_;
  std::vector<std::vector<ObservationData*>>  observation_vec_;


  // ceres parameter
  ceres::LocalParameterization*               quat_parameterization_ptr_;
  ceres::LossFunction*                        loss_function_ptr_;

  ceres::Problem                              optimization_problem_;
  ceres::Solver::Options                      optimization_options_;
  ceres::Solver::Summary                      optimization_summary_;

};



int main(int argc, char **argv) {

  google::InitGoogleLogging(argv[0]);

  std::string config_folder_path("config/");
  std::string euroc_dataset_path = "/home/lemur/dataset/EuRoC/MH_01_easy/mav0/";


  ExpLandmarkEmSLAM slam_problem(config_folder_path);

  // initialize the first state
  slam_problem.ReadInitialTraj("data/");
  slam_problem.ReadObservationData("data/");
  slam_problem.ReadImuData(euroc_dataset_path + "imu0/data.csv");

  boost::posix_time::ptime begin_time = boost::posix_time::microsec_clock::local_time();

  slam_problem.SetupMStep();

  slam_problem.SolveEmProblem();
  slam_problem.SolveEmProblem();

  boost::posix_time::ptime end_time = boost::posix_time::microsec_clock::local_time();
  boost::posix_time::time_duration t = end_time - begin_time;
  double dt = ((double)t.total_nanoseconds() * 1e-9);

  std::cout << "The entire time is " << dt << " sec." << std::endl;

  slam_problem.OutputOptimizationResult("trajectory_em.csv");

  return 0;
}