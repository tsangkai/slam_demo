
#include <cassert>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <map>

#include <boost/date_time/posix_time/posix_time.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Core>
#include <EigenRand/EigenRand>
#include <opencv2/core.hpp>
#include <opencv2/core/cvstd.hpp>

#include "so3.h"
#include "constant.h"                           // gravity value
#include "imu_data.h"
#include "vec_3d_parameter_block.h"
#include "quat_parameter_block.h"
#include "pre_int_imu_error.h"
#include "reprojection_error.h"   


Eigen::Rand::Vmt19937_64 urng{ (unsigned int) time(0) };


struct State {
  double             t_;
  Eigen::Quaterniond q_;
  Eigen::Vector3d    v_;
  Eigen::Vector3d    p_;
};

struct Estimate {
  double             t_;
  Eigen::Quaterniond q_;  
  Eigen::Vector3d    v_;
  Eigen::Vector3d    p_; 

  Eigen::Matrix<double, 9, 9> cov_;
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

  ObservationData() {
  }

  Eigen::Matrix2d cov() {
    return cov_ * Eigen::Matrix2d::Identity();
  }

  double timestamp_;
  size_t landmark_id_;
  Eigen::Vector2d feature_pos_; // u and v
  double cov_;
};















class ExpLandmarkOptSLAM {
 EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 public:

  ExpLandmarkOptSLAM(std::string config_file_path) {

    cv::FileStorage config_file(config_file_path, cv::FileStorage::READ);

    landmark_len_ = (size_t)(int) config_file["landmark_len"];

    duration_ = (double) config_file["duration"];
    dt_ = (double) config_file["dt"];
    keyframe_rate_ratio_ = (size_t)(int) config_file["keyframe_rate_ratio"];

    // trajectory parameter
    r_ = (double) config_file["trajectory"]["r"];
    w_ = (double) config_file["trajectory"]["w"];
    r_z_ = (double) config_file["trajectory"]["r_z"];
    w_z_ = (double) config_file["trajectory"]["w_z"];
    z_h_ = (double) config_file["trajectory"]["z_h"];

    sigma_g_c_ = (double) config_file["imu_param"]["sigma_g_c"];
    sigma_a_c_ = (double) config_file["imu_param"]["sigma_a_c"];

    box_xy_ = (double) config_file["landmark_generation"]["box_xy"];
    box_z_ = (double) config_file["landmark_generation"]["box_z"];

    landmark_init_noise_ = (double) config_file["landmark_init_noise"];

    cv::FileNode T_bc = config_file["camera"]["T_bc"];
    T_bc_ << (double) T_bc[0], (double) T_bc[1], (double) T_bc[2], (double) T_bc[3],
             (double) T_bc[4], (double) T_bc[5], (double) T_bc[6], (double) T_bc[7],
             (double) T_bc[8], (double) T_bc[9], (double) T_bc[10], (double) T_bc[11],
             (double) T_bc[12], (double) T_bc[13], (double) T_bc[14], (double) T_bc[15];

    du_ = (double) config_file["camera"]["image_dimension"][0];  // image dimension
    dv_ = (double) config_file["camera"]["image_dimension"][1];
    fu_ = (double) config_file["camera"]["focal_length"][0];  // focal length
    fv_ = (double) config_file["camera"]["focal_length"][1];
    cu_ = (double) config_file["camera"]["principal_point"][0];  // principal point
    cv_ = (double) config_file["camera"]["principal_point"][1];

    obs_cov_ = (double) config_file["camera"]["observation_noise"];
  }



  bool CreateTrajectory() {

    double T=0;

    while (T <= duration_) {

      Eigen::Matrix3d rot;
      Eigen::Vector3d vel;      
      Eigen::Vector3d pos;

      rot << cos(w_ * T), -sin(w_ * T), 0,
             sin(w_ * T),  cos(w_ * T), 0,
             0, 0, 1;

      vel(0) = -r_ * w_ * sin(w_ * T);
      vel(1) =  r_ * w_ * cos(w_ * T);
      vel(2) = r_z_ * w_z_ * cos(w_z_ * T);

      pos(0) = r_ * cos(w_ * T);
      pos(1) = r_ * sin(w_ * T);
      pos(2) = r_z_ * sin(w_z_ * T) + z_h_;


      State* state_ptr = new State;

      state_ptr->t_= T;
      state_ptr->q_ = quat_positive(Eigen::Quaterniond(rot));
      state_ptr->v_ = vel;
      state_ptr->p_ = pos;

      state_vec_.push_back(state_ptr);

      T = T + keyframe_rate_ratio_*dt_;
    }

    state_len_ = state_vec_.size();

    std::cout << "state_vec_ " << state_vec_.size() << std::endl;
    return true;
  }


  bool CreateLandmark() {

    for (size_t i=0; i< landmark_len_/4; i++) { //x walls first

      Eigen::Vector3d random_3d_vec = Eigen::Vector3d::Random();
      Eigen::Vector3d* landmark_pos = new Eigen::Vector3d((r_+box_xy_)*random_3d_vec(0),
                                                          (r_+box_xy_),
                                                          box_z_ * random_3d_vec(2) + z_h_);

      landmark_vec_.push_back(landmark_pos);
    }

    for (size_t i=landmark_len_/4; i<landmark_len_/2; i++) {

      Eigen::Vector3d random_3d_vec = Eigen::Vector3d::Random();
      Eigen::Vector3d* landmark_pos = new Eigen::Vector3d((r_+box_xy_)*random_3d_vec(0),
                                                          -(r_+box_xy_),
                                                          box_z_ * random_3d_vec(2) + z_h_);

      landmark_vec_.push_back(landmark_pos);
    }

    for (size_t i=landmark_len_/2; i< 3*landmark_len_/4; i++) {

      Eigen::Vector3d random_3d_vec = Eigen::Vector3d::Random();
      Eigen::Vector3d* landmark_pos = new Eigen::Vector3d((r_+box_xy_),
                                                          (r_+box_xy_)*random_3d_vec(1),
                                                          box_z_ * random_3d_vec(2) + z_h_);

      landmark_vec_.push_back(landmark_pos);
    }

    for (size_t i=3*landmark_len_/4; i< landmark_len_; i++) {

      Eigen::Vector3d random_3d_vec = Eigen::Vector3d::Random();
      Eigen::Vector3d* landmark_pos = new Eigen::Vector3d(-(r_+box_xy_),
                                                          (r_+box_xy_)*random_3d_vec(1),
                                                          box_z_ * random_3d_vec(2) + z_h_);

      landmark_vec_.push_back(landmark_pos);
    }

    return true;
  }  


  bool CreateImuData() {

    double T = 0;
    size_t state_idx = 0;

    imu_vec_.resize(state_len_-1);
    
    while (T <= duration_ && (state_idx+1) < state_vec_.size()) {

      Eigen::Matrix3d rot;
      rot << cos(w_ * T), -sin(w_ * T), 0,
             sin(w_ * T),  cos(w_ * T), 0,
             0, 0, 1;

      Eigen::Vector3d a_N = Eigen::Vector3d(-r_*(w_*w_)*cos(w_*T),
                                            -r_*(w_*w_)*sin(w_*T),
                                            -r_z_*(w_z_*w_z_)*sin(w_z_*T));
      Eigen::Vector3d omega_B = Eigen::Vector3d(0, 0, w_);

      Eigen::Vector3d gyr_noise = sigma_g_c_/sqrt(dt_)*Eigen::Rand::normal<Eigen::Vector3d>(3, 1, urng);
      Eigen::Vector3d acc_noise = sigma_a_c_/sqrt(dt_)*Eigen::Rand::normal<Eigen::Vector3d>(3, 1, urng);


      IMUData* imu_ptr = new IMUData;
      imu_ptr->timestamp_ = T;
      imu_ptr->gyr_ = omega_B + gyr_noise;
      imu_ptr->acc_ = rot.transpose() * (a_N - gravity) + acc_noise;

      imu_vec_.at(state_idx).push_back(imu_ptr);


      if (T + dt_ >= state_vec_.at(state_idx+1)->t_) {
        state_idx++;
      }
      
      T = T + dt_;
    
    }

    std::cout << "imu_vec_ " << imu_vec_.size() << std::endl;

    return true;
  }


  bool CreateObservationData() {

    observation_vec_.resize(state_len_-1);
    double T = keyframe_rate_ratio_*dt_;
    size_t state_idx = 0;

    for (size_t i=1; i<state_vec_.size(); ++i) {

      double t;
      Eigen::Matrix3d rot;
      Eigen::Vector3d pos;

      t = state_vec_.at(i)->t_;
      rot = state_vec_.at(i)->q_.toRotationMatrix();
      pos = state_vec_.at(i)->p_;

      Eigen::Matrix4d T_bn = Eigen::Matrix4d::Identity();
      T_bn.topLeftCorner<3, 3>() = rot.transpose();
      T_bn.topRightCorner<3, 1>() = -1 * rot.transpose() * pos;


      for (size_t m=0; m<landmark_len_; m++) {

        // homogeneous transformation of the landmark to camera frame
        Eigen::Vector4d landmark_n = Eigen::Vector4d(0, 0, 0, 1);
        landmark_n.head(3) = *landmark_vec_.at(m);
        Eigen::Vector4d landmark_c = T_bc_.transpose() * T_bn * landmark_n;

        if (landmark_c(2) > 0) {

          Eigen::Vector2d feature_pt;
          
          feature_pt(0) = fu_ * landmark_c(0) / landmark_c(2) + cu_;
          feature_pt(1) = fv_ * landmark_c(1) / landmark_c(2) + cv_;

          // check whether this point is in the frame
          if (abs(feature_pt(0)) <= du_/2 && abs(feature_pt(1)) <= dv_/2) {

            ObservationData* feature_obs_ptr = new ObservationData();
            feature_obs_ptr->timestamp_ = t;
            feature_obs_ptr->landmark_id_ = m;
            feature_obs_ptr->feature_pos_ = feature_pt + sqrt(obs_cov_) * Eigen::Rand::normal<Eigen::Vector2d>(2, 1, urng);
            feature_obs_ptr->cov_ = obs_cov_;

            observation_vec_.at(i-1).push_back(feature_obs_ptr);
          
          }
        }
      }
    }


    std::cout << "observation_vec_ " << observation_vec_.size() << std::endl;

    return true;
  }




  // initialize state_est_vec_ and landmark_est_vec_
  bool InitializeSLAMProblem() {
    
    // state estimate
    state_est_vec_.resize(state_len_);

    for (size_t i=0; i<state_len_; ++i) {

      Estimate* state_est_ptr = new Estimate;
      state_est_ptr->t_ = state_vec_.at(i)->t_;
      state_est_vec_.at(i) = state_est_ptr;
    }    

    // landmark estimate
    landmark_est_vec_.resize(landmark_len_);

    for (size_t i=0; i<landmark_len_; ++i) {

      Eigen::Vector3d landmark_est = *landmark_vec_.at(i) + landmark_init_noise_ * Eigen::Vector3d::Random();
      landmark_est_vec_.at(i) = new Eigen::Vector3d(landmark_est);
    }


    // forward filtering
    state_est_vec_.at(0)->q_ = state_vec_.at(0)->q_;
    state_est_vec_.at(0)->v_ = state_vec_.at(0)->v_;
    state_est_vec_.at(0)->p_ = state_vec_.at(0)->p_;
    state_est_vec_.at(0)->cov_ = Eigen::Matrix<double, 9, 9>::Zero();


    for (size_t i=0; i<imu_vec_.size(); ++i) {

      Eigen::Quaterniond q = state_est_vec_.at(i)->q_;
      Eigen::Vector3d v = state_est_vec_.at(i)->v_;
      Eigen::Vector3d p = state_est_vec_.at(i)->p_;
      Eigen::Matrix<double, 9, 9> cov = state_est_vec_.at(i)->cov_;

      // forward update for state_est_vec_.at(i+1)

      for (size_t j=0; j<imu_vec_.at(i).size(); ++j) {

        Eigen::Vector3d gyr = imu_vec_.at(i).at(j)->gyr_;  
        Eigen::Vector3d acc = imu_vec_.at(i).at(j)->acc_;

        Eigen::Quaterniond q1 = quat_positive(q * Exp_q(dt_ * gyr));
        Eigen::Vector3d v1 = v + dt_ * (q.toRotationMatrix()* acc + gravity);
        Eigen::Vector3d p1 = p + dt_ * v + 0.5 * dt_*dt_ * (q.toRotationMatrix()* acc + gravity);


        Eigen::Matrix<double, 9, 9> F_t = Eigen::Matrix<double, 9, 9>::Zero();
        F_t.block<3,3>(0,0) = Exp(dt_*gyr).transpose();
        F_t.block<3,3>(3,0) = (-1)*dt_*q.toRotationMatrix()*Skew(acc);
        F_t.block<3,3>(3,3) = Eigen::Matrix3d::Identity();
        F_t.block<3,3>(6,0) = (-0.5)*dt_*dt_*q.toRotationMatrix()*Skew(acc);
        F_t.block<3,3>(6,3) = dt_*Eigen::Matrix3d::Identity();
        F_t.block<3,3>(6,6) = Eigen::Matrix3d::Identity();

        Eigen::Matrix<double, 9, 6> G_t = Eigen::Matrix<double, 9, 6>::Zero();
        G_t.block<3,3>(0,0) = (-1)*dt_*Eigen::Matrix3d::Identity();
        G_t.block<3,3>(3,3) = (-1)*dt_*q.toRotationMatrix();
        G_t.block<3,3>(6,3) = (-0.5)*dt_*dt_*q.toRotationMatrix();

        Eigen::Matrix<double, 6, 6> w_cov = Eigen::Matrix<double, 6, 6>::Zero();
        w_cov.block<3,3>(0,0) = (sigma_g_c_*sigma_g_c_/dt_)*Eigen::Matrix3d::Identity();
        w_cov.block<3,3>(3,3) = (sigma_a_c_*sigma_a_c_/dt_)*Eigen::Matrix3d::Identity();


        q = q1;
        v = v1;
        p = p1;
        cov = F_t * cov * F_t.transpose() + G_t * w_cov * G_t.transpose();
      }

      
      // observation update
      Eigen::Matrix3d k_R = Eigen::Matrix3d::Identity();
      Eigen::Vector3d k_v = Eigen::Vector3d::Zero();
      Eigen::Vector3d k_p = Eigen::Vector3d::Zero();

      for (size_t j=0; j<observation_vec_.at(i).size(); ++j) {

        Eigen::Vector3d landmark = *landmark_est_vec_.at(observation_vec_.at(i).at(j)->landmark_id_);
        Eigen::Vector2d measurement = observation_vec_.at(i).at(j)->feature_pos_;
        Eigen::Matrix2d R = observation_vec_.at(i).at(j)->cov();

        Eigen::Matrix3d R_bc = T_bc_.topLeftCorner<3,3>();
        Eigen::Vector3d t_bc = T_bc_.topRightCorner<3,1>();

        Eigen::Matrix3d R_nb = q.toRotationMatrix();
        Eigen::Vector3d t_nb = p;

        Eigen::Matrix4d T_bn = Eigen::Matrix4d::Identity();
        T_bn.topLeftCorner<3, 3>() = q.toRotationMatrix().transpose();
        T_bn.topRightCorner<3, 1>() = -1 * q.toRotationMatrix().transpose() * p;

        // Eigen::Vector3d landmark_c = R_bc.transpose() * ((R_nb.transpose()*(landmark - t_nb)) - t_bc);
        
        Eigen::Vector4d landmark_n = Eigen::Vector4d(0, 0, 0, 1);
        landmark_n.head(3) = landmark;
        Eigen::Vector4d landmark_c = T_bc_.transpose() * T_bn * landmark_n;


        Eigen::Vector2d landmark_proj;
        landmark_proj << fu_ * landmark_c[0]/landmark_c[2] + cu_, 
                         fv_ * landmark_c[1]/landmark_c[2] + cv_;

        // exclude outliers
        Eigen::Vector2d innovation = measurement - landmark_proj;
        // if (innovation.norm() < 80) {  
        if (1) {  

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
          K = cov * H.transpose() * (H * cov * H.transpose() + R).inverse();
          Eigen::Matrix<double, 9, 1> m;
          m = K * (measurement - landmark_proj);

          k_R = k_R * Exp(m.block<3,1>(0,0));
          k_v = k_v + m.block<3,1>(3,0);
          k_p = k_p + m.block<3,1>(6,0);  

          Eigen::Matrix<double, 9, 9> IKH;
          IKH = Eigen::Matrix<double, 9, 9>::Identity() - K * H;
          cov = IKH * cov * IKH.transpose() + K * R * K.transpose();     // Joseph form
          
        }
      }

      // if (k_p.norm() < 0.65) {
      if (1) {
        state_est_vec_.at(i+1)->q_ = quat_positive(Eigen::Quaterniond(q * k_R));
        state_est_vec_.at(i+1)->v_ = v + k_v;
        state_est_vec_.at(i+1)->p_ = p + k_p;
        state_est_vec_.at(i+1)->cov_ = cov;
      }
    }



    // backward smoothing
    for (size_t i=imu_vec_.size()-1; i>0; --i) {

      // std::cout << "RTS smoother: " << i << std::endl;

      Eigen::Quaterniond q = state_est_vec_.at(i)->q_;
      Eigen::Vector3d v = state_est_vec_.at(i)->v_;
      Eigen::Vector3d p = state_est_vec_.at(i)->p_;
      Eigen::Matrix<double, 9, 9> cov = state_est_vec_.at(i)->cov_;

      Eigen::Matrix<double, 9, 9> F = Eigen::Matrix<double, 9, 9>::Identity();


      for (size_t j=0; j<imu_vec_.at(i).size(); ++j) {

        Eigen::Vector3d gyr = imu_vec_.at(i).at(j)->gyr_;  
        Eigen::Vector3d acc = imu_vec_.at(i).at(j)->acc_;

        Eigen::Quaterniond q1 = quat_positive(q * Exp_q(dt_ * gyr));
        Eigen::Vector3d v1 = v + dt_ * (q.toRotationMatrix()* acc + gravity);
        Eigen::Vector3d p1 = p + dt_ * v + 0.5 * dt_*dt_ * (q.toRotationMatrix()* acc + gravity);


        Eigen::Matrix<double, 9, 9> F_t = Eigen::Matrix<double, 9, 9>::Zero();
        F_t.block<3,3>(0,0) = Exp(dt_*gyr).transpose();
        F_t.block<3,3>(3,0) = (-1)*dt_*q.toRotationMatrix()*Skew(acc);
        F_t.block<3,3>(3,3) = Eigen::Matrix3d::Identity();
        F_t.block<3,3>(6,0) = (-0.5)*dt_*dt_*q.toRotationMatrix()*Skew(acc);
        F_t.block<3,3>(6,3) = dt_*Eigen::Matrix3d::Identity();
        F_t.block<3,3>(6,6) = Eigen::Matrix3d::Identity();

        Eigen::Matrix<double, 9, 6> G_t = Eigen::Matrix<double, 9, 6>::Zero();
        G_t.block<3,3>(0,0) = (-1)*dt_*Eigen::Matrix3d::Identity();
        G_t.block<3,3>(3,3) = (-1)*dt_*q.toRotationMatrix();
        G_t.block<3,3>(6,3) = (-0.5)*dt_*dt_*q.toRotationMatrix();

        Eigen::Matrix<double, 6, 6> w_cov = Eigen::Matrix<double, 6, 6>::Zero();
        w_cov.block<3,3>(0,0) = (sigma_g_c_*sigma_g_c_/dt_)*Eigen::Matrix3d::Identity();
        w_cov.block<3,3>(3,3) = (sigma_a_c_*sigma_a_c_/dt_)*Eigen::Matrix3d::Identity();


        q = q1;
        v = v1;
        p = p1;
        cov = F_t * cov * F_t.transpose() + G_t * w_cov * G_t.transpose();

        F = F_t * F;
      }


      Eigen::Matrix<double, 9, 9> C;
      C = state_est_vec_.at(i)->cov_ * F.transpose() * cov.inverse();

      Eigen::Matrix<double, 9, 1> residual;
      residual.block<3,1>(0,0) = Log_q(q.conjugate() * state_est_vec_.at(i+1)->q_);
      residual.block<3,1>(3,0) = state_est_vec_.at(i+1)->v_ - v;
      residual.block<3,1>(6,0) = state_est_vec_.at(i+1)->p_ - p;

      Eigen::Matrix<double, 9, 1> m;
      m = C * residual;  // give the IMU results less weight


      state_est_vec_.at(i)->q_ = quat_positive(state_est_vec_.at(i)->q_ * Exp_q(m.block<3,1>(0,0)));
      state_est_vec_.at(i)->v_ = state_est_vec_.at(i)->v_ + m.block<3,1>(3,0);
      state_est_vec_.at(i)->p_ = state_est_vec_.at(i)->p_ + m.block<3,1>(6,0);

      // ignore sigma update

    }

    return true;
  }












  bool SolveOptProblem() {

    quat_parameterization_ptr_ = new ceres::QuaternionParameterization();


    // create parameter block
    state_para_vec_.resize(state_len_);

    for (size_t i=0; i<state_len_; ++i) {

      state_para_vec_.at(i) = new StatePara(state_vec_.at(i)->t_);

      state_para_vec_.at(i)->GetRotationBlock()->setEstimate(state_est_vec_.at(i)->q_);
      state_para_vec_.at(i)->GetVelocityBlock()->setEstimate(state_est_vec_.at(i)->v_);
      state_para_vec_.at(i)->GetPositionBlock()->setEstimate(state_est_vec_.at(i)->p_);
    }


    landmark_para_vec_.resize(landmark_len_);

    for (size_t i=0; i<landmark_len_; ++i) {

      landmark_para_vec_.at(i) = new Vec3dParameterBlock();

      landmark_para_vec_.at(i)->setEstimate(*landmark_est_vec_.at(i));
    }


    // add parameter blocks
    for (size_t i=0; i<state_len_; ++i) {

      optimization_problem_.AddParameterBlock(state_para_vec_.at(i)->GetRotationBlock()->parameters(), 4, quat_parameterization_ptr_);
      optimization_problem_.AddParameterBlock(state_para_vec_.at(i)->GetVelocityBlock()->parameters(), 3);
      optimization_problem_.AddParameterBlock(state_para_vec_.at(i)->GetPositionBlock()->parameters(), 3); 
    }

    optimization_problem_.SetParameterBlockConstant(state_para_vec_.at(0)->GetRotationBlock()->parameters());
    optimization_problem_.SetParameterBlockConstant(state_para_vec_.at(0)->GetVelocityBlock()->parameters());
    optimization_problem_.SetParameterBlockConstant(state_para_vec_.at(0)->GetPositionBlock()->parameters());     


    for (size_t i=0; i<landmark_len_; ++i) {
      optimization_problem_.AddParameterBlock(landmark_para_vec_.at(i)->parameters(), 3);
    }


    // add constraints
    for (size_t i=0; i<imu_vec_.size(); ++i) {

      PreIntIMUData* int_imu_data_ptr = new PreIntIMUData(Eigen::Vector3d(0, 0, 0),
                                                          Eigen::Vector3d(0, 0, 0),
                                                          sigma_g_c_,
                                                          sigma_a_c_);

      for (size_t j=0; j<imu_vec_.at(i).size(); ++j) {
        int_imu_data_ptr->IntegrateSingleIMU(*imu_vec_.at(i).at(j), dt_);
      }

      ceres::CostFunction* cost_function = new PreIntImuError(int_imu_data_ptr->dt_,
                                                              int_imu_data_ptr->dR_,
                                                              int_imu_data_ptr->dv_,
                                                              int_imu_data_ptr->dp_,
                                                              int_imu_data_ptr->cov_);

      optimization_problem_.AddResidualBlock(cost_function,
                                             NULL,
                                             state_para_vec_.at(i+1)->GetRotationBlock()->parameters(),
                                             state_para_vec_.at(i+1)->GetVelocityBlock()->parameters(),
                                             state_para_vec_.at(i+1)->GetPositionBlock()->parameters(),
                                             state_para_vec_.at(i)->GetRotationBlock()->parameters(),
                                             state_para_vec_.at(i)->GetVelocityBlock()->parameters(),
                                             state_para_vec_.at(i)->GetPositionBlock()->parameters());
    }

    for (size_t i=0; i<observation_vec_.size(); ++i) {
      for (size_t j=0; j<observation_vec_.at(i).size(); ++j) {

        size_t landmark_idx = observation_vec_.at(i).at(j)->landmark_id_;  

        ceres::CostFunction* cost_function = new ReprojectionError(observation_vec_.at(i).at(j)->feature_pos_,
                                                                   T_bc_,
                                                                   fu_, fv_,
                                                                   cu_, cv_,
                                                                   observation_vec_.at(i).at(j)->cov());

        optimization_problem_.AddResidualBlock(cost_function,
                                               NULL,
                                               state_para_vec_.at(i+1)->GetRotationBlock()->parameters(),
                                               state_para_vec_.at(i+1)->GetPositionBlock()->parameters(),
                                               landmark_para_vec_.at(landmark_idx)->parameters());
      }
    }


    // solve the optimization problem
    optimization_options_.linear_solver_type = ceres::SPARSE_SCHUR;
    optimization_options_.minimizer_progress_to_stdout = true;
    optimization_options_.num_threads = 6;
    optimization_options_.function_tolerance = 1e-20;
    optimization_options_.parameter_tolerance = 1e-25;
    optimization_options_.max_num_iterations = 80; //100;

    ceres::Solve(optimization_options_, &optimization_problem_, &optimization_summary_);
    std::cout << optimization_summary_.FullReport() << "\n";


    // store results
    for (size_t i=1; i<state_len_; ++i) {
      state_est_vec_.at(i)->q_ = state_para_vec_.at(i)->GetRotationBlock()->estimate();
      state_est_vec_.at(i)->v_ = state_para_vec_.at(i)->GetVelocityBlock()->estimate();
      state_est_vec_.at(i)->p_ = state_para_vec_.at(i)->GetPositionBlock()->estimate();
    }

    for (size_t i=0; i<landmark_len_; ++i) {
      *landmark_est_vec_.at(i) = landmark_para_vec_.at(i)->estimate();
    }


    return true;
  }


  bool OutputGroundtruth(std::string output_file_name) {

    std::ofstream output_file(output_file_name);

    output_file << "timestamp,p_x,p_y,p_z,v_x,v_y,v_z,q_w,q_x,q_y,q_z\n";

    for (size_t i=0; i<state_len_; ++i) {

      output_file << std::to_string(state_vec_.at(i)->t_) << ",";
      output_file << std::to_string(state_vec_.at(i)->p_(0)) << ",";
      output_file << std::to_string(state_vec_.at(i)->p_(1)) << ",";
      output_file << std::to_string(state_vec_.at(i)->p_(2)) << ",";
      output_file << std::to_string(state_vec_.at(i)->v_(0)) << ",";
      output_file << std::to_string(state_vec_.at(i)->v_(1)) << ",";
      output_file << std::to_string(state_vec_.at(i)->v_(2)) << ",";
      output_file << std::to_string(state_vec_.at(i)->q_.w()) << ",";
      output_file << std::to_string(state_vec_.at(i)->q_.x()) << ",";
      output_file << std::to_string(state_vec_.at(i)->q_.y()) << ",";
      output_file << std::to_string(state_vec_.at(i)->q_.z()) << std::endl;
    }

    output_file.close();

    return true;


  }

  bool OutputResult(std::string output_file_name) {

    std::ofstream output_file(output_file_name);

    output_file << "timestamp,p_x,p_y,p_z,v_x,v_y,v_z,q_w,q_x,q_y,q_z\n";

    for (size_t i=0; i<state_len_; ++i) {

      output_file << std::to_string(state_est_vec_.at(i)->t_) << ",";
      output_file << std::to_string(state_est_vec_.at(i)->p_(0)) << ",";
      output_file << std::to_string(state_est_vec_.at(i)->p_(1)) << ",";
      output_file << std::to_string(state_est_vec_.at(i)->p_(2)) << ",";
      output_file << std::to_string(state_est_vec_.at(i)->v_(0)) << ",";
      output_file << std::to_string(state_est_vec_.at(i)->v_(1)) << ",";
      output_file << std::to_string(state_est_vec_.at(i)->v_(2)) << ",";
      output_file << std::to_string(state_est_vec_.at(i)->q_.w()) << ",";
      output_file << std::to_string(state_est_vec_.at(i)->q_.x()) << ",";
      output_file << std::to_string(state_est_vec_.at(i)->q_.y()) << ",";
      output_file << std::to_string(state_est_vec_.at(i)->q_.z()) << std::endl;
    }

    output_file.close();

    return true;
  }
  

 private:

  // experiment parameters
  size_t state_len_;
  size_t landmark_len_;

  double duration_;
  double dt_;
  size_t keyframe_rate_ratio_;

  // trajectory parameter
  double r_; // circle radius x-y plane
  double w_; // angular velocity
  double r_z_;
  double w_z_;
  double z_h_; // height of the uav

  // IMU parameters
  double sigma_g_c_;    // gyro noise density [rad/s/sqrt(Hz)]
  double sigma_a_c_;    // accelerometer noise density [m/s^2/sqrt(Hz)]

  // landmark generation parameter
  double box_xy_;  // box offset from the circle
  double box_z_;   // box offset from uav height
  double landmark_init_noise_;

  // camera parameters

  Eigen::Matrix4d T_bc_;

  double obs_cov_; 

  double du_;  // image dimension
  double dv_;
  double fu_;  // focal length
  double fv_;
  double cu_;  // principal point
  double cv_;


  // ground truth containers
  std::vector<State*>                         state_vec_;              // state_len
  std::vector<Eigen::Vector3d*>               landmark_vec_;           // landmark_len

  std::vector<Estimate*>                      state_est_vec_;          // state_len
  std::vector<Eigen::Vector3d*>               landmark_est_vec_;       // landmark_len

  // parameter containers
  std::vector<StatePara*>                     state_para_vec_;         // state_len
  std::vector<Vec3dParameterBlock*>           landmark_para_vec_;      // landmark_len

  std::vector<std::vector<IMUData*>>          imu_vec_;                // state_len-1
  std::vector<std::vector<ObservationData*>>  observation_vec_;        // state_len-1
  

  // ceres parameter
  ceres::Problem                              optimization_problem_;
  ceres::Solver::Options                      optimization_options_;
  ceres::Solver::Summary                      optimization_summary_;
  ceres::LocalParameterization*               quat_parameterization_ptr_;

};



int main(int argc, char **argv) {



  google::InitGoogleLogging(argv[0]);

  ExpLandmarkOptSLAM slam_problem("config/config_sim.yaml");

  slam_problem.CreateTrajectory();
  slam_problem.CreateLandmark();

  slam_problem.CreateImuData();
  slam_problem.CreateObservationData();

  slam_problem.OutputGroundtruth("result/sim/gt.csv");


  boost::posix_time::ptime begin_time = boost::posix_time::microsec_clock::local_time();

  slam_problem.InitializeSLAMProblem();
  slam_problem.OutputResult("result/sim/dr.csv");

  slam_problem.SolveOptProblem();

  boost::posix_time::ptime end_time = boost::posix_time::microsec_clock::local_time();
  boost::posix_time::time_duration t = end_time - begin_time;
  double dt = ((double)t.total_nanoseconds() * 1e-9);

  std::cout << "The entire time is " << dt << " sec." << std::endl;

  slam_problem.OutputResult("result/sim/opt.csv");

  return 0;
}