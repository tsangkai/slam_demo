
#include <cassert>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <map>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <opencv2/core/core.hpp>    // for config file reading
#include <Eigen/Core>
#include <boost/date_time/posix_time/posix_time.hpp>


#include "so3.h"
#include "vec_3d_parameter_block.h"
#include "quat_parameter_block.h"
#include "imu_data.h"
#include "imu_error.h"
#include "pre_int_imu_error.h"
#include "reprojection_error.h"   



Eigen::Vector3d gravity = Eigen::Vector3d(0, 0, -9.81007);

struct State {

  double             timestamp_;
  Eigen::Quaterniond q_;
  Eigen::Vector3d    v_;
  Eigen::Vector3d    p_;
};


// This class only handles the memory. The underlying estimate is handled by each parameter blocks.
class StatePara {

 public:
  StatePara(double timestamp) {
    timestamp_ = timestamp;

    rotation_block_ptr_ = new QuatParameterBlock();
    velocity_block_ptr_ = new Vec3dParameterBlock();
    position_block_ptr_ = new Vec3dParameterBlock();
  }

  ~StatePara() {
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

struct ObservationData {
 EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ObservationData(double timestamp) {
    timestamp_ = timestamp;
  }

  Eigen::Matrix2d cov() {
    double sigma_2 = size_ * size_ / 64.0;
    // return sigma_2 * Eigen::Matrix2d::Identity();
    return 1e-7 * Eigen::Matrix2d::Identity();
  }

  double timestamp_;
  size_t landmark_id_;
  Eigen::Vector2d feature_pos_; // u and v
  double size_;
};


Eigen::Quaterniond quat_pos(Eigen::Quaterniond q){
    if (q.w() < 0) {
        q.w() = (-1)*q.w();
        q.x() = (-1)*q.x();
        q.y() = (-1)*q.y();
        q.z() = (-1)*q.z();
    }
    return q;
};


class ExpLandmarkOptSLAM {
 EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 public:

  ExpLandmarkOptSLAM() {

    // TODO: move the private memeber assignment here
    
    T_bc_ << cos(M_PI/2), 0, sin(M_PI/2), 0,
          0,            1, 0,           0,
          -sin(M_PI/2), 0, cos(M_PI/2), 0,
          0, 0, 0, 1;
  }



  bool CreateTrajectory() {

    double T=0;

    for (size_t i=0; i<state_len_; i++) {

      Eigen::Matrix3d rot;
      Eigen::Vector3d vel;      
      Eigen::Vector3d pos;

      rot << cos(w * T), -sin(w * T), 0,
             sin(w * T),  cos(w * T), 0,
             0, 0, 1;

      vel(0) = -r * w * sin(w * T);
      vel(1) =  r * w * cos(w * T);
      vel(2) = r_z * w_z * cos(w_z * T);

      pos(0) = r * cos(w * T);
      pos(1) = r * sin(w * T);
      pos(2) = r_z * sin(w_z * T) + z_h;


      State* state_ptr = new State;

      state_ptr->q_ = quat_pos(Eigen::Quaterniond(rot));
      state_ptr->v_ = vel;
      state_ptr->p_ = pos;

      state_vec_.push_back(state_ptr);

      T = T + dt_;
    }

    return true;
  }


  bool CreateLandmark() {

    for (size_t i=0; i< landmark_len_/4; i++) { //x walls first
      Eigen::Vector3d landmark_pos;

      landmark_pos(0) = (r+box_xy)*Eigen::Vector3d::Random()(0);
      landmark_pos(1) = (r+box_xy);
      landmark_pos(2) = (z_h + box_z)*Eigen::Vector3d::Random()(2) + z_h;

      landmark_vec_.push_back(landmark_pos);
    }

    for (size_t i=landmark_len_/4; i<landmark_len_/2; i++) { //x walls first
      Eigen::Vector3d landmark_pos;
      
      landmark_pos(0) = (r+box_xy)*Eigen::Vector3d::Random()(0);
      landmark_pos(1) = -(r+box_xy);
      landmark_pos(2) = (z_h + box_z)*Eigen::Vector3d::Random()(2) + z_h;

      landmark_vec_.push_back(landmark_pos);
    }

    for (size_t i=landmark_len_/2; i< 3*landmark_len_/4; i++) { //x walls first
      Eigen::Vector3d landmark_pos;
      
      landmark_pos(0) = (r+box_xy);
      landmark_pos(1) = (r+box_xy)*Eigen::Vector3d::Random()(1);
      landmark_pos(2) = (z_h + box_z)*Eigen::Vector3d::Random()(2) + z_h;

      landmark_vec_.push_back(landmark_pos);
    }

    for (size_t i=3*landmark_len_/4; i< landmark_len_; i++) { //x walls first
      Eigen::Vector3d landmark_pos;
      
      landmark_pos(0) = -(r+box_xy);
      landmark_pos(1) = (r+box_xy)*Eigen::Vector3d::Random()(1);
      landmark_pos(2) = (z_h + box_z)*Eigen::Vector3d::Random()(2) + z_h;

      landmark_vec_.push_back(landmark_pos);
    }

    return true;
  }  


  bool CreateImuData() {

    double T = 0.0;
    for (size_t i=0; i<state_len_-1; i++) {

      Eigen::Vector3d a_N;
      a_N(0) = -r*(w*w)*cos(w*T);
      a_N(1) = -r*(w*w)*sin(w*T);
      a_N(2) = -r_z*(w_z*w_z)*sin(w_z*T);

      Eigen::Vector3d omega_B = Eigen::Vector3d(0, 0, w);


      Eigen::Vector3d gyr_noise = sigma_g_c/sqrt(dt_)*Eigen::Vector3d::Random();
      Eigen::Vector3d acc_noise = sigma_a_c/sqrt(dt_)*Eigen::Vector3d::Random();


      IMUData* imu_ptr = new IMUData;
      
      imu_ptr->timestamp_ = T;
      imu_ptr->gyr_ = omega_B + gyr_noise;
      imu_ptr->acc_ = state_vec_.at(i)->q_.toRotationMatrix().transpose() * (a_N - gravity) + acc_noise;

      imu_vec_.push_back(imu_ptr);

      T = T + dt_;
    }


    return true;
  }

  bool CreateObservationData() {


    observation_vec_.resize(state_len_);

    for (size_t i=0; i< state_len_; i++) { //x walls first


      Eigen::Matrix4d T_bn = Eigen::Matrix4d::Identity();
      T_bn.topLeftCorner<3, 3>() = state_vec_.at(i)->q_.toRotationMatrix().transpose();
      T_bn.topRightCorner<3, 1>() = -1 * state_vec_.at(i)->q_.toRotationMatrix().transpose() * state_vec_.at(i)->p_;

      for (size_t m = 0; m < landmark_len_; m++) { //x walls first

        // homogeneous transformation of the landmark to camera frame
        Eigen::Vector4d landmark_n = Eigen::Vector4d(0, 0, 0, 1);
        landmark_n.head(3) = landmark_vec_.at(m);
        Eigen::Vector4d landmark_c = T_bc_.transpose() * T_bn * landmark_n;

        if (landmark_c(2) > 0) {

          Eigen::Vector2d feature_pt;
          
          feature_pt(0) = fu * landmark_c(0) / landmark_c(2) + cu;
          feature_pt(1) = fv * landmark_c(1) / landmark_c(2) + cv;

          // check whether this point is in the frame
          if (abs(feature_pt(0)) <= du/2 && abs(feature_pt(1)) <= dv/2) {

            ObservationData* feature_obs_ptr = new ObservationData(state_vec_.at(i)->timestamp_);
            feature_obs_ptr->landmark_id_ = m;
            feature_obs_ptr->feature_pos_ = feature_pt + 0.03 * Eigen::Vector2d::Random();

            observation_vec_.at(i).push_back(feature_obs_ptr);
          
          }
        }
      }
    }

    return true;
  }



  bool SetupOptProblem() {

    Eigen::Quaterniond q0 = state_vec_.at(0)->q_;
    Eigen::Vector3d v0 = state_vec_.at(0)->v_;
    Eigen::Vector3d p0 = state_vec_.at(0)->p_;

    // the first state
    StatePara* state_para_ptr = new StatePara(state_vec_.at(0)->timestamp_);

    state_para_ptr->GetRotationBlock()->setEstimate(q0);
    state_para_ptr->GetVelocityBlock()->setEstimate(v0);
    state_para_ptr->GetPositionBlock()->setEstimate(p0);
    state_para_vec_.push_back(state_para_ptr);

    // the following states
    for (size_t i=0; i<state_len_-1; ++i) {

      Eigen::Vector3d gyr = imu_vec_.at(i)->gyr_;
      Eigen::Vector3d acc = imu_vec_.at(i)->acc_;

      p0 = p0 + dt_ * v0 + 0.5 * dt_*dt_ * (q0.toRotationMatrix()* acc + gravity);
      v0 = v0 + dt_ * (q0.toRotationMatrix()* acc + gravity);
      q0 = quat_pos(q0 * Exp_q(dt_ * gyr));

      state_para_ptr = new StatePara(state_vec_.at(i+1)->timestamp_);
      state_para_ptr->GetRotationBlock()->setEstimate(q0);
      state_para_ptr->GetVelocityBlock()->setEstimate(v0);
      state_para_ptr->GetPositionBlock()->setEstimate(p0);

      state_para_vec_.push_back(state_para_ptr);
    }


    // the following states
    for (size_t i=0; i<landmark_len_; ++i) {
      Vec3dParameterBlock* landmark_ptr = new Vec3dParameterBlock();
      // landmark_ptr->setEstimate(landmark_vec_.at(i) + 1e-7 * Eigen::Vector3d::Random());
      landmark_ptr->setEstimate(landmark_vec_.at(i));
      landmark_para_vec_.push_back(landmark_ptr);
    }


    // add parameter blocks
    for (size_t i=0; i<state_len_; ++i) {
      optimization_problem_.AddParameterBlock(state_para_vec_.at(i)->GetRotationBlock()->parameters(), 4);
      optimization_problem_.AddParameterBlock(state_para_vec_.at(i)->GetVelocityBlock()->parameters(), 3);
      optimization_problem_.AddParameterBlock(state_para_vec_.at(i)->GetPositionBlock()->parameters(), 3);
    }
    


    optimization_problem_.SetParameterBlockConstant(state_para_vec_.at(0)->GetRotationBlock()->parameters());
    optimization_problem_.SetParameterBlockConstant(state_para_vec_.at(0)->GetVelocityBlock()->parameters());
    optimization_problem_.SetParameterBlockConstant(state_para_vec_.at(0)->GetPositionBlock()->parameters());


    for (size_t i=0; i<landmark_vec_.size(); ++i) {
      optimization_problem_.AddParameterBlock(landmark_para_vec_.at(i)->parameters(), 3);
      optimization_problem_.SetParameterBlockConstant(landmark_para_vec_.at(i)->parameters());
    }

   
    // imu constraints
    for (size_t i=0; i<imu_vec_.size(); ++i) {

      ceres::CostFunction* cost_function = new ImuError(imu_vec_.at(i)->gyr_,
                                                        imu_vec_.at(i)->acc_,
                                                        dt_,
                                                        Eigen::Vector3d(0,0,0),
                                                        Eigen::Vector3d(0,0,0),
                                                        sigma_g_c,
                                                        sigma_a_c);

      optimization_problem_.AddResidualBlock(cost_function,
                                             NULL,
                                             state_para_vec_.at(i+1)->GetRotationBlock()->parameters(),
                                             state_para_vec_.at(i+1)->GetVelocityBlock()->parameters(),
                                             state_para_vec_.at(i+1)->GetPositionBlock()->parameters(),
                                             state_para_vec_.at(i)->GetRotationBlock()->parameters(),
                                             state_para_vec_.at(i)->GetVelocityBlock()->parameters(),
                                             state_para_vec_.at(i)->GetPositionBlock()->parameters());

    }


    // observation constraints
    for (size_t i=0; i<observation_vec_.size(); ++i) {
      for (size_t j=0; j<observation_vec_.at(i).size(); ++j) {

        size_t landmark_idx = observation_vec_.at(i).at(j)->landmark_id_;  

        ceres::CostFunction* cost_function = new ReprojectionError(observation_vec_.at(i).at(j)->feature_pos_,
                                                                   T_bc_,
                                                                   fu, fv,
                                                                   cu, cv,
                                                                   observation_vec_.at(i).at(j)->cov());

        optimization_problem_.AddResidualBlock(cost_function,
                                               NULL,
                                               state_para_vec_.at(i)->GetRotationBlock()->parameters(),
                                               state_para_vec_.at(i)->GetPositionBlock()->parameters(),
                                               landmark_para_vec_.at(landmark_idx)->parameters());
      }
    }


    return true;
  }



  bool SolveOptProblem() {

    std::cout << "Begin solving the optimization problem." << std::endl;

    optimization_options_.linear_solver_type = ceres::SPARSE_SCHUR;
    optimization_options_.minimizer_progress_to_stdout = true;
    optimization_options_.num_threads = 6;
    optimization_options_.function_tolerance = 1e-20;
    optimization_options_.parameter_tolerance = 1e-25;
    optimization_options_.max_num_iterations = 100;


    ceres::Solve(optimization_options_, &optimization_problem_, &optimization_summary_);
    std::cout << optimization_summary_.FullReport() << "\n";

    return true;
  }

  bool OutputGroundtruth(std::string output_folder_name) {
    std::ofstream traj_output_file(output_folder_name + "gt.csv");

    traj_output_file << "timestamp,p_x,p_y,p_z,v_x,v_y,v_z,q_w,q_x,q_y,q_z\n";

    for (size_t i=0; i<state_len_; ++i) {

      traj_output_file << std::to_string(state_vec_.at(i)->timestamp_) << ",";
      traj_output_file << std::to_string(state_vec_.at(i)->p_(0)) << ",";
      traj_output_file << std::to_string(state_vec_.at(i)->p_(1)) << ",";
      traj_output_file << std::to_string(state_vec_.at(i)->p_(2)) << ",";
      traj_output_file << std::to_string(state_vec_.at(i)->v_(0)) << ",";
      traj_output_file << std::to_string(state_vec_.at(i)->v_(1)) << ",";
      traj_output_file << std::to_string(state_vec_.at(i)->v_(2)) << ",";
      traj_output_file << std::to_string(state_vec_.at(i)->q_.w()) << ",";
      traj_output_file << std::to_string(state_vec_.at(i)->q_.x()) << ",";
      traj_output_file << std::to_string(state_vec_.at(i)->q_.y()) << ",";
      traj_output_file << std::to_string(state_vec_.at(i)->q_.z()) << std::endl;
    }

    traj_output_file.close();

    return true;


  }

  bool OutputResult(std::string output_file_name) {

    std::ofstream output_file(output_file_name);

    output_file << "timestamp,p_x,p_y,p_z,v_x,v_y,v_z,q_w,q_x,q_y,q_z\n";

    for (size_t i=0; i<state_len_; ++i) {

      output_file << std::to_string(state_para_vec_.at(i)->GetTimestamp()) << ",";
      output_file << std::to_string(state_para_vec_.at(i)->GetPositionBlock()->estimate()(0)) << ",";
      output_file << std::to_string(state_para_vec_.at(i)->GetPositionBlock()->estimate()(1)) << ",";
      output_file << std::to_string(state_para_vec_.at(i)->GetPositionBlock()->estimate()(2)) << ",";
      output_file << std::to_string(state_para_vec_.at(i)->GetVelocityBlock()->estimate()(0)) << ",";
      output_file << std::to_string(state_para_vec_.at(i)->GetVelocityBlock()->estimate()(1)) << ",";
      output_file << std::to_string(state_para_vec_.at(i)->GetVelocityBlock()->estimate()(2)) << ",";
      output_file << std::to_string(state_para_vec_.at(i)->GetRotationBlock()->estimate().w()) << ",";
      output_file << std::to_string(state_para_vec_.at(i)->GetRotationBlock()->estimate().x()) << ",";
      output_file << std::to_string(state_para_vec_.at(i)->GetRotationBlock()->estimate().y()) << ",";
      output_file << std::to_string(state_para_vec_.at(i)->GetRotationBlock()->estimate().z()) << std::endl;
    }

    output_file.close();

    return true;
  }
  

 private:

  // experiment parameters

  size_t state_len_ = 500;
  size_t landmark_len_ = 4*20;

  double dt_ = 0.02;

  // trajectory parameter
  double r = 5.0; // circle radius x-y plane
  double w = .76; // angular velocity
  double r_z = (1.0/20)*r;
  double w_z = (2.3)*w;
  double z_h = 0.0; // height of the uav

  // IMU parameters
  double sigma_g_c = 6.0e-4;
  double sigma_a_c = 2.0e-3;

  // landmark generation parameter
  double box_xy = 2;  // box offset from the circle
  double box_z = 1;   // box offset from uav height

  double sigma_g_c_;   // gyro noise density [rad/s/sqrt(Hz)]
  double sigma_a_c_;   // accelerometer noise density [m/s^2/sqrt(Hz)]


  // camera parameters

  Eigen::Matrix4d T_bc_;

  double du = 500.0;  // image dimension
  double dv = 1000.0;
  double fu = 500.0;  // focal length
  double fv = 500.0;
  double cu = 0.0;    // principal point
  double cv = 0.0;


  // ground truth containers
  std::vector<State*>                         state_vec_;
  std::vector<Eigen::Vector3d>                landmark_vec_;


  // parameter containers
  std::vector<StatePara*>                     state_para_vec_;
  std::vector<Vec3dParameterBlock*>           landmark_para_vec_;

  std::vector<IMUData*>                       imu_vec_;
  std::vector<std::vector<ObservationData*>>  observation_vec_;
  

  // ceres parameter
  ceres::Problem                              optimization_problem_;
  ceres::Solver::Options                      optimization_options_;
  ceres::Solver::Summary                      optimization_summary_;

};



int main(int argc, char **argv) {

  google::InitGoogleLogging(argv[0]);

  ExpLandmarkOptSLAM slam_problem;

  slam_problem.CreateTrajectory();
  slam_problem.CreateLandmark();

  slam_problem.CreateImuData();
  slam_problem.CreateObservationData();

  slam_problem.SetupOptProblem();

  slam_problem.OutputResult("result/sim/dr.csv");
  slam_problem.SolveOptProblem();
  slam_problem.OutputResult("result/sim/opt.csv");
  slam_problem.OutputGroundtruth("result/sim/");


  // output result

  return 0;
}