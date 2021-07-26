
#include <iostream>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Core>

#include "sim.h"



class ExpLandmarkEmSLAM: public ExpLandmarkSLAM {
 EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 public:

  ExpLandmarkEmSLAM(std::string config_file_path): 
    ExpLandmarkSLAM(config_file_path) {

  }



  bool E_step() {

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
        // if (1) {  

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
          
        // }
      }

      // if (k_p.norm() < 0.65) {
      // if (1) {
        state_est_vec_.at(i+1)->q_ = quat_positive(Eigen::Quaterniond(q * k_R));
        state_est_vec_.at(i+1)->v_ = v + k_v;
        state_est_vec_.at(i+1)->p_ = p + k_p;
        state_est_vec_.at(i+1)->cov_ = cov;
      // }
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


  bool M_step() {


    // ceres parameter
    ceres::Problem                  optimization_problem;
    ceres::Solver::Options          optimization_options;
    ceres::Solver::Summary          optimization_summary;
    ceres::LocalParameterization*   quat_parameterization_ptr;

    optimization_options.linear_solver_type = ceres::SPARSE_SCHUR;
    optimization_options.minimizer_progress_to_stdout = true;
    optimization_options.num_threads = 6;
    optimization_options.function_tolerance = 1e-20;
    optimization_options.parameter_tolerance = 1e-25;
    optimization_options.max_num_iterations = 80; //100;

    quat_parameterization_ptr = new ceres::QuaternionParameterization();


    // udpate parameter block

    for (size_t i=0; i<state_len_; ++i) {
      state_para_vec_.at(i)->GetRotationBlock()->setEstimate(state_est_vec_.at(i)->q_);
      state_para_vec_.at(i)->GetVelocityBlock()->setEstimate(state_est_vec_.at(i)->v_);
      state_para_vec_.at(i)->GetPositionBlock()->setEstimate(state_est_vec_.at(i)->p_);
    }

    for (size_t i=0; i<landmark_len_; ++i) {
      landmark_para_vec_.at(i)->setEstimate(*landmark_est_vec_.at(i));
    }


    // add parameter blocks
    for (size_t i=0; i<state_len_; ++i) {

      optimization_problem.AddParameterBlock(state_para_vec_.at(i)->GetRotationBlock()->parameters(), 4, quat_parameterization_ptr);
      optimization_problem.AddParameterBlock(state_para_vec_.at(i)->GetVelocityBlock()->parameters(), 3);
      optimization_problem.AddParameterBlock(state_para_vec_.at(i)->GetPositionBlock()->parameters(), 3); 
    }

    for (size_t i=0; i<state_len_; ++i) {

      optimization_problem.SetParameterBlockConstant(state_para_vec_.at(i)->GetRotationBlock()->parameters());
      optimization_problem.SetParameterBlockConstant(state_para_vec_.at(i)->GetVelocityBlock()->parameters());
      optimization_problem.SetParameterBlockConstant(state_para_vec_.at(i)->GetPositionBlock()->parameters());     
    }

    for (size_t i=0; i<landmark_len_; ++i) {
      optimization_problem.AddParameterBlock(landmark_para_vec_.at(i)->parameters(), 3);
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

      optimization_problem.AddResidualBlock(cost_function,
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

        optimization_problem.AddResidualBlock(cost_function,
                                              NULL,
                                              state_para_vec_.at(i+1)->GetRotationBlock()->parameters(),
                                              state_para_vec_.at(i+1)->GetPositionBlock()->parameters(),
                                              landmark_para_vec_.at(landmark_idx)->parameters());
      }
    }




    ceres::Solve(optimization_options, &optimization_problem, &optimization_summary);
    std::cout << optimization_summary.FullReport() << "\n";


    // store results

    for (size_t i=0; i<landmark_len_; ++i) {
      *landmark_est_vec_.at(i) = landmark_para_vec_.at(i)->estimate();
    }


    return true;
  }


};



int main(int argc, char **argv) {
//  srand((unsigned int) time(NULL)); //eigen uses the random number generator of the standard lib

  std::cout << "simulate EM SLAM..." << std::endl;

  google::InitGoogleLogging(argv[0]);

  int num_real = atoi(argv[1]);

//  Eigen::Rand::Vmt19937_64 urng{ (unsigned int) time(0) };

  for (size_t i = 0; i < num_real; ++i) {


    Eigen::Rand::Vmt19937_64 urng{ i };

    ExpLandmarkEmSLAM slam_problem("config/config_sim.yaml");
    slam_problem.CreateTrajectory();
    slam_problem.CreateLandmark(urng);

    slam_problem.CreateImuData(urng);
    slam_problem.CreateObservationData(urng);


    slam_problem.InitializeSLAMProblem();

    slam_problem.E_step();
    slam_problem.M_step();
    slam_problem.E_step();


    slam_problem.OutputResult("result/sim_fixed/em_" + std::to_string(i) + ".csv");

    std::cout << "Completed EM trial " << std::to_string(i) << std::endl;
  }
  return 0;
}