
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


struct ObservationData {
 EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ObservationData(std::string observation_data_str) {
    std::stringstream str_stream(observation_data_str);          // Create a stringstream of the current line

    if (str_stream.good()) {
        
      std::string data_str;
      std::getline(str_stream, data_str, ',');   // get first string delimited by comma
      timestamp_ = std::stod(data_str)*1e-9;

      std::getline(str_stream, data_str, ','); 
      landmark_id_ = std::stoi(data_str);

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
};



struct TriangularData {
  TriangularData(Eigen::Vector2d keypoint, Eigen::Quaterniond rotation, Eigen::Vector3d position) {
    keypoint_ = keypoint;
    rotation_ = rotation;
    position_ = position;
  }

  Eigen::Vector2d keypoint_;
  Eigen::Quaterniond rotation_;
  Eigen::Vector3d position_;
};

// keypoints should be normalized
Eigen::Vector3d EpipolarInitialize(Eigen::Vector2d kp1, Eigen::Quaterniond q1, Eigen::Vector3d p1_n1,
                                   Eigen::Vector2d kp2, Eigen::Quaterniond q2, Eigen::Vector3d p2_n2,
                                   Eigen::Matrix4d T_bc,
                                   double fu, double fv, double cu, double cv) {
  Eigen::Matrix3d K;
  K << fu,  0, cu,
        0, fv, cv,
        0,  0,  1;

  Eigen::Matrix<double, 3, 4> projection;
  projection << 1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0;

  // from body frame to camera frame
  Eigen::Matrix3d R_bc = T_bc.topLeftCorner<3,3>();
  Eigen::Matrix3d R_cb = R_bc.transpose();
  Eigen::Vector3d t_bc = T_bc.topRightCorner<3,1>();

  Eigen::Matrix4d T_cb = Eigen::Matrix4d::Identity();
  T_cb.topLeftCorner<3,3>() = R_cb;
  T_cb.topRightCorner<3,1>() = -R_cb * t_bc;

  // from nagivation frame to body frame
  Eigen::Matrix3d R_nb_1 = q1.toRotationMatrix();
  Eigen::Matrix3d R_bn_1 = R_nb_1.transpose();

  Eigen::Matrix4d T_bn_1 = Eigen::Matrix4d::Identity();
  T_bn_1.topLeftCorner<3,3>() = R_bn_1;
  T_bn_1.topRightCorner<3,1>() = -R_bn_1 * p1_n1;

  Eigen::Matrix3d R_nb_2 = q2.toRotationMatrix();
  Eigen::Matrix3d R_bn_2 = R_nb_2.transpose();

  Eigen::Matrix4d T_bn_2 = Eigen::Matrix4d::Identity();
  T_bn_2.topLeftCorner<3,3>() = R_bn_2;
  T_bn_2.topRightCorner<3,1>() = -R_bn_2 * p2_n2;

  // 
  Eigen::Matrix<double, 3, 4> P1 = K * projection * T_cb * T_bn_1;
  Eigen::Matrix<double, 3, 4> P2 = K * projection * T_cb * T_bn_2;

  // 
  // Eigen::Matrix<double, 3, 4> P1 = projection * T_cb * T_bn_1;
  // Eigen::Matrix<double, 3, 4> P2 = projection * T_cb * T_bn_2;

  //
  Eigen::Matrix4d A;
  A.row(0) = kp1(0) * P1.row(2) - P1.row(0);
  A.row(1) = kp1(1) * P1.row(2) - P1.row(1);
  A.row(2) = kp2(0) * P2.row(2) - P2.row(0);
  A.row(3) = kp2(1) * P2.row(2) - P2.row(1);

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Vector4d vec = svd.matrixV().col(3);

  return vec.head(3) / vec(3);
}






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

    cu_ = experiment_config_file["cameras"][0]["principal_point"][0];
    cv_ = experiment_config_file["cameras"][0]["principal_point"][1];

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



    // set initial condition
    for (size_t i=0; i<1; ++i) {
      optimization_problem_.SetParameterBlockConstant(state_parameter_.at(i)->GetRotationBlock()->parameters());
      optimization_problem_.SetParameterBlockConstant(state_parameter_.at(i)->GetVelocityBlock()->parameters());
      optimization_problem_.SetParameterBlockConstant(state_parameter_.at(i)->GetPositionBlock()->parameters());      
    }


    // loop closure
    state_parameter_.back()->GetPositionBlock()->setEstimate(state_parameter_.front()->GetPositionBlock()->estimate());
    // optimization_problem_.SetParameterBlockConstant(state_parameter_.back()->GetPositionBlock()->parameters());      

    return true;
  }

  bool ReadObservationData() {
  
    std::string feature_obs_file_path("config/feature_obs.csv");
    std::cout << "Read feature observation data at " << feature_obs_file_path << std::endl;
    std::ifstream feature_obs_file(feature_obs_file_path);
    assert(("Could not open ground truth file.", feature_obs_file.is_open()));


    // ignore the header
    std::string line;
    std::getline(feature_obs_file, line);

    while (std::getline(feature_obs_file, line)) {

      observation_data_.push_back(new ObservationData(line));

      size_t landmark_id = observation_data_.back()->landmark_id_-1;

      if (landmark_id >= landmark_parameter_.size()) {
        landmark_parameter_.resize(landmark_id+1);
      }
    }

    for (size_t i=0; i<landmark_parameter_.size(); ++i) {
      landmark_parameter_.at(i) = new Vec3dParameterBlock();
    }

    feature_obs_file.close();

    return true;
  }




  bool Triangulate() {

    tri_data_.resize(landmark_parameter_.size());

    for (size_t i=0; i<observation_data_.size(); ++i) {

      // determine the two nodes of the bipartite graph
      size_t state_idx = 0;
      for (size_t j=0; j<state_parameter_.size(); ++j) {
        if (state_parameter_.at(j)->GetTimestamp() == observation_data_.at(i)->timestamp_) {
          state_idx = j;
          break;
        }
      }

      size_t landmark_idx = observation_data_.at(i)->landmark_id_-1;  

      ceres::CostFunction* cost_function = new ReprojectionError(observation_data_.at(i)->feature_pos_,
                                                                 T_bc_,
                                                                 fu_, fv_,
                                                                 cu_, cv_,
                                                                 observation_data_.at(i)->cov());

      optimization_problem_.AddResidualBlock(cost_function,
                                             NULL, //loss_function_ptr_,
                                             state_parameter_.at(state_idx)->GetRotationBlock()->parameters(),
                                             state_parameter_.at(state_idx)->GetPositionBlock()->parameters(),
                                             landmark_parameter_.at(landmark_idx)->parameters());


      TriangularData tri_data_instance(observation_data_.at(i)->feature_pos_,
                                       state_parameter_.at(state_idx)->GetRotationBlock()->estimate(),
                                       state_parameter_.at(state_idx)->GetPositionBlock()->estimate());
      tri_data_.at(landmark_idx).push_back(tri_data_instance);
    }


    // set landmark initial estimate
    for (size_t i=0; i<landmark_parameter_.size(); ++i) {

      size_t idx_0 = 0;
      size_t idx_1 = std::min(size_t(3), tri_data_.at(i).size()-1);

      Eigen::Vector3d init_landmark_pos = EpipolarInitialize(tri_data_.at(i).at(idx_0).keypoint_, 
                                                             tri_data_.at(i).at(idx_0).rotation_, 
                                                             tri_data_.at(i).at(idx_0).position_,
                                                             tri_data_.at(i).at(idx_1).keypoint_, 
                                                             tri_data_.at(i).at(idx_1).rotation_, 
                                                             tri_data_.at(i).at(idx_1).position_,
                                                             T_bc_, fu_, fv_, cu_, cv_);

      landmark_parameter_.at(i)->setEstimate(init_landmark_pos);
    }

    return true;
  }


  bool ReadIMUData(std::string imu_file_path) {
  
    std::cout << "Read IMU data at " << imu_file_path << std::endl;

    std::ifstream imu_file(imu_file_path);
    assert(("Could not open IMU file.", imu_file.is_open()));

    // Read the column names
    // Extract the first line in the file
    std::string first_line_data_str;
    std::getline(imu_file, first_line_data_str);

    size_t state_idx = 0;                 // the index of the last element

    PreIntIMUData int_imu_data(state_parameter_.at(state_idx)->GetGyroBias(),
                               state_parameter_.at(state_idx)->GetAcceBias(),
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
        if ((state_idx + 1) == state_parameter_.size()) {
          int_imu_data.IntegrateSingleIMU(imu_data, imu_dt_);
        }
        // case 2: the time stamp of the imu data is between two consecutive states
        else if (imu_data.timestamp_ < state_parameter_.at(state_idx+1)->GetTimestamp()) {
          int_imu_data.IntegrateSingleIMU(imu_data, imu_dt_);
        }
        // case 3: the imu data just enter the new interval of integration
        else {

          // add imu constraint
          ceres::CostFunction* cost_function = new PreIntImuError(int_imu_data.dt_,
                                                                  int_imu_data.dR_,
                                                                  int_imu_data.dv_,
                                                                  int_imu_data.dp_,
                                                                  int_imu_data.cov_);

          optimization_problem_.AddResidualBlock(cost_function,
                                                 NULL,
                                                 state_parameter_.at(state_idx+1)->GetRotationBlock()->parameters(),
                                                 state_parameter_.at(state_idx+1)->GetVelocityBlock()->parameters(),
                                                 state_parameter_.at(state_idx+1)->GetPositionBlock()->parameters(),
                                                 state_parameter_.at(state_idx)->GetRotationBlock()->parameters(),
                                                 state_parameter_.at(state_idx)->GetVelocityBlock()->parameters(),
                                                 state_parameter_.at(state_idx)->GetPositionBlock()->parameters());   

          state_idx++;
          
          // prepare for next iteration
          int_imu_data = PreIntIMUData(state_parameter_.at(state_idx)->GetGyroBias(),
                                       state_parameter_.at(state_idx)->GetAcceBias(),
                                       sigma_g_c_,
                                       sigma_a_c_);

          int_imu_data.IntegrateSingleIMU(imu_data, imu_dt_);

        }
      }
    }

    imu_file.close();
    std::cout << "Finished reading IMU data." << std::endl;
    return true;
  }

  bool SolveOptimizationProblem() {

    std::cout << "Begin solving the optimization problem." << std::endl;

    optimization_options_.linear_solver_type = ceres::SPARSE_SCHUR;
    optimization_options_.minimizer_progress_to_stdout = true;
    optimization_options_.num_threads = 6;
    optimization_options_.function_tolerance = 1e-20;
    optimization_options_.parameter_tolerance = 1e-20;
    optimization_options_.max_num_iterations = 100;

    // for (size_t i=0; i<landmark_parameter_.size(); ++i) {
   //    optimization_problem_.SetParameterBlockConstant(landmark_parameter_.at(i)->parameters());
    //}


    for (size_t i=1; i<state_parameter_.size(); ++i) {
      optimization_problem_.SetParameterBlockVariable(state_parameter_.at(i)->GetRotationBlock()->parameters());
      optimization_problem_.SetParameterBlockVariable(state_parameter_.at(i)->GetVelocityBlock()->parameters());
      optimization_problem_.SetParameterBlockVariable(state_parameter_.at(i)->GetPositionBlock()->parameters());
    }

    ceres::Solve(optimization_options_, &optimization_problem_, &optimization_summary_);
    std::cout << optimization_summary_.FullReport() << "\n";

    return true;
  }

  bool OutputOptimizationResult(std::string output_file_name) {

    std::ofstream output_file(output_file_name);

    output_file << "timestamp,p_x,p_y,p_z,v_x,v_y,v_z,q_w,q_x,q_y,q_z\n";

    // hacking
    Eigen::Quaterniond gt_q0(0.589549, -0.095865, -0.801532, -0.028104);
    Eigen::Vector3d gt_p0(4.573070, -1.670144, 0.571735);

    Eigen::Quaterniond q0 = state_parameter_.at(0)->GetRotationBlock()->estimate();
    Eigen::Vector3d p0 = state_parameter_.at(0)->GetPositionBlock()->estimate();


    for (size_t i=0; i<state_parameter_.size(); ++i) {

      Eigen::Quaterniond qt = state_parameter_.at(i)->GetRotationBlock()->estimate() * q0.inverse() * gt_q0;
      Eigen::Vector3d vt = gt_q0.toRotationMatrix() * q0.inverse().toRotationMatrix() * state_parameter_.at(i)->GetVelocityBlock()->estimate();
      Eigen::Vector3d pt = gt_q0.toRotationMatrix() * q0.inverse().toRotationMatrix() * (state_parameter_.at(i)->GetPositionBlock()->estimate()-p0) + gt_p0;

      output_file << std::to_string(state_parameter_.at(i)->GetTimestamp()) << ",";
      output_file << std::to_string(pt(0)) << ",";
      output_file << std::to_string(pt(1)) << ",";
      output_file << std::to_string(pt(2)) << ",";
      output_file << std::to_string(vt(0)) << ",";
      output_file << std::to_string(vt(1)) << ",";
      output_file << std::to_string(vt(2)) << ",";
      output_file << std::to_string(qt.w()) << ",";
      output_file << std::to_string(qt.x()) << ",";
      output_file << std::to_string(qt.y()) << ",";
      output_file << std::to_string(qt.z()) << std::endl;

      /***
      output_file << std::to_string(state_parameter_.at(i)->GetTimestamp()) << ",";
      output_file << std::to_string(state_parameter_.at(i)->GetPositionBlock()->estimate()(0)) << ",";
      output_file << std::to_string(state_parameter_.at(i)->GetPositionBlock()->estimate()(1)) << ",";
      output_file << std::to_string(state_parameter_.at(i)->GetPositionBlock()->estimate()(2)) << ",";
      output_file << std::to_string(state_parameter_.at(i)->GetVelocityBlock()->estimate()(0)) << ",";
      output_file << std::to_string(state_parameter_.at(i)->GetVelocityBlock()->estimate()(1)) << ",";
      output_file << std::to_string(state_parameter_.at(i)->GetVelocityBlock()->estimate()(2)) << ",";
      output_file << std::to_string(state_parameter_.at(i)->GetRotationBlock()->estimate().w()) << ",";
      output_file << std::to_string(state_parameter_.at(i)->GetRotationBlock()->estimate().x()) << ",";
      output_file << std::to_string(state_parameter_.at(i)->GetRotationBlock()->estimate().y()) << ",";
      output_file << std::to_string(state_parameter_.at(i)->GetRotationBlock()->estimate().z()) << std::endl;
      ***/



    }

    output_file.close();

    /***
    std::ofstream output_file_landmark("landmark.csv");

    output_file_landmark << "id,p_x,p_y,p_z\n";

    for (size_t i=0; i<landmark_parameter_.size(); ++i) {
      output_file_landmark << std::to_string(i+1) << ",";
      output_file_landmark << std::to_string(landmark_parameter_.at(i)->estimate()(0)) << ",";
      output_file_landmark << std::to_string(landmark_parameter_.at(i)->estimate()(1)) << ",";
      output_file_landmark << std::to_string(landmark_parameter_.at(i)->estimate()(2)) << std::endl;
    }

    output_file_landmark.close();
    ***/

    return true;
  }


 private:

  // experiment parameters
  double imu_dt_;

  Eigen::Matrix4d T_bc_;

  double fu_;
  double fv_;
  double cu_;
  double cv_;

  // parameter containers
  std::vector<State*>                state_parameter_;
  std::vector<Vec3dParameterBlock*>  landmark_parameter_;

  std::vector<ObservationData*>      observation_data_;


  double sigma_g_c_;   // gyro noise density [rad/s/sqrt(Hz)]
  double sigma_a_c_;   // accelerometer noise density [m/s^2/sqrt(Hz)]

  std::vector<std::vector<TriangularData>> tri_data_;         // tri_data[num_of_landmark][num_of_obs]


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

  std::string euroc_dataset_path = "/home/lemur/dataset/EuRoC/MH_01_easy/mav0/";

  // initialize the first state
  slam_problem.ReadInitialCondition();

  slam_problem.ReadObservationData();

  slam_problem.Triangulate();


  // setup IMU constraints
  std::string imu_file_path = euroc_dataset_path + "imu0/data.csv";
  slam_problem.ReadIMUData(imu_file_path);
  slam_problem.OutputOptimizationResult("trajectory_dr.csv");


  slam_problem.SolveOptimizationProblem();
  slam_problem.OutputOptimizationResult("trajectory.csv");

  return 0;
}