


#include <string>

#include <boost/date_time/posix_time/posix_time.hpp>
#include <Eigen/Core>
#include <ceres/ceres.h>

#include "backend.h"
#include "constant.h"


class ExpLandmarkEmSLAM: public ExpLandmarkSLAM {
 EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 public:

  ExpLandmarkEmSLAM(std::string config_folder_path): 
    ExpLandmarkSLAM(config_folder_path) {

  }



  bool ExpectationStep(double feature_threshold) {


    double dt_ = imu_dt_;


    // forward Kalman filter
    for (size_t i=0; i<imu_vec_.size(); ++i) {



      Eigen::Quaterniond q0 = state_est_vec_.at(i)->q_;
      Eigen::Vector3d v0 = state_est_vec_.at(i)->v_;
      Eigen::Vector3d p0 = state_est_vec_.at(i)->p_;

      Eigen::Matrix<double, 9, 9> cov = state_est_vec_.at(i)->cov_;



      for (size_t j=0; j<imu_vec_.at(i).size(); ++j) {
        Eigen::Vector3d gyr = imu_vec_.at(i).at(j)->gyr_ - state_est_vec_.at(i)->b_gyr_;  
        Eigen::Vector3d acc = imu_vec_.at(i).at(j)->acc_ - state_est_vec_.at(i)->b_acc_;  

        Eigen::Quaterniond q1 = quat_positive(q0 * Exp_q(dt_ * gyr));
        Eigen::Vector3d v1 = v0 + dt_ * (q0.toRotationMatrix()* acc + gravity);
        Eigen::Vector3d p1 = p0 + dt_ * v0 + 0.5 * dt_*dt_ * (q0.toRotationMatrix()* acc + gravity);

        Eigen::Matrix<double, 9, 9> F = Eigen::Matrix<double, 9, 9>::Zero();
        F.block<3,3>(0,0) = Exp(dt_*gyr).transpose();
        F.block<3,3>(3,0) = (-1)*dt_*q0.toRotationMatrix()*Skew(acc);
        F.block<3,3>(3,3) = Eigen::Matrix3d::Identity();
        F.block<3,3>(6,0) = (-0.5)*dt_*dt_*q0.toRotationMatrix()*Skew(acc);
        F.block<3,3>(6,3) = dt_*Eigen::Matrix3d::Identity();
        F.block<3,3>(6,6) = Eigen::Matrix3d::Identity();

        Eigen::Matrix<double, 9, 6> G = Eigen::Matrix<double, 9, 6>::Zero();
        G.block<3,3>(0,0) = (-1)*dt_*Eigen::Matrix3d::Identity();
        G.block<3,3>(3,3) = (-1)*dt_*q0.toRotationMatrix();
        G.block<3,3>(6,3) = (-0.5)*dt_*dt_*q0.toRotationMatrix();

        Eigen::Matrix<double, 6, 6> w_cov = Eigen::Matrix<double, 6, 6>::Zero();
        w_cov.block<3,3>(0,0) = (sigma_g_c_*sigma_g_c_/dt_)*Eigen::Matrix3d::Identity();
        w_cov.block<3,3>(3,3) = (sigma_a_c_*sigma_a_c_/dt_)*Eigen::Matrix3d::Identity();

        q0 = q1;
        v0 = v1;
        p0 = p1;
        cov = F * cov * F.transpose() + G * w_cov * G.transpose();
      }


      if ((p0 - state_est_vec_.at(i+1)->p_).norm() < 0.02) {

        state_est_vec_.at(i+1)->q_ = quat_positive(q0);
        state_est_vec_.at(i+1)->v_ = v0;
        state_est_vec_.at(i+1)->p_ = p0;
      }

      state_est_vec_.at(i+1)->cov_ = cov;


      // observation update
      Eigen::Matrix3d k_R = Eigen::Matrix3d::Identity();
      Eigen::Vector3d k_v = Eigen::Vector3d::Zero();
      Eigen::Vector3d k_p = Eigen::Vector3d::Zero();
      Eigen::Matrix<double, 9, 9> obs_cov = cov;

      for (size_t j=0; j<observation_vec_.at(i).size(); ++j) {

        Eigen::Vector3d landmark = *landmark_est_vec_.at(observation_vec_.at(i).at(j)->landmark_id_-1);
        Eigen::Vector2d measurement = observation_vec_.at(i).at(j)->feature_pos_;
        Eigen::Matrix2d R = observation_vec_.at(i).at(j)->cov();

        Eigen::Matrix3d R_bc = T_bc_.topLeftCorner<3,3>();
        Eigen::Vector3d t_bc = T_bc_.topRightCorner<3,1>();

        Eigen::Matrix3d R_nb = state_est_vec_.at(i+1)->q_.toRotationMatrix();
        Eigen::Vector3d t_nb = state_est_vec_.at(i+1)->p_;

        Eigen::Vector3d landmark_c = R_bc.transpose() * ((R_nb.transpose()*(landmark - t_nb)) - t_bc);
        Eigen::Vector2d landmark_proj;
        landmark_proj << fu_ * landmark_c[0]/landmark_c[2] + cu_, 
                         fv_ * landmark_c[1]/landmark_c[2] + cv_;

        // exclude outliers
        Eigen::Vector2d innovation = measurement - landmark_proj;


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
        K = obs_cov * H.transpose() * (H * obs_cov * H.transpose() + R).inverse();
        Eigen::Matrix<double, 9, 1> m;
        m = K * (measurement - landmark_proj);

        if (m.block<3,1>(6,0).norm() < feature_threshold) {
          k_R = k_R * Exp(m.block<3,1>(0,0));
          k_v = k_v + m.block<3,1>(3,0);
          k_p = k_p + m.block<3,1>(6,0);  

          Eigen::Matrix<double, 9, 9> IKH;
          IKH = Eigen::Matrix<double, 9, 9>::Identity() - K * H;
          obs_cov = IKH * obs_cov * IKH.transpose() + K * R * K.transpose();     // Joseph form
  
        }
      }


      if (k_p.norm() < 1.0) {

        state_est_vec_.at(i+1)->q_ = quat_positive(Eigen::Quaterniond(state_est_vec_.at(i+1)->q_ * k_R));
        state_est_vec_.at(i+1)->v_ = state_est_vec_.at(i+1)->v_ + k_v;
        state_est_vec_.at(i+1)->p_ = state_est_vec_.at(i+1)->p_ + k_p;

        state_est_vec_.at(i+1)->cov_ = obs_cov;
      }

    }



    // backward RTS smoother
    for (size_t i=imu_vec_.size()-1; i>0; --i) {

      Eigen::Quaterniond q0 = state_est_vec_.at(i)->q_;
      // Eigen::Vector3d v0 = state_est_vec_.at(i)->v_;
      // Eigen::Vector3d p0 = state_est_vec_.at(i)->p_;


      Eigen::Matrix<double, 9, 9> cov = state_est_vec_.at(i)->cov_;

      Eigen::Matrix<double, 9, 9> F_all = Eigen::Matrix<double, 9, 9>::Identity();

      for (size_t j=0; j<imu_vec_.at(i).size(); ++j) {
        Eigen::Vector3d gyr = imu_vec_.at(i).at(j)->gyr_ - state_est_vec_.at(i)->b_gyr_;  
        Eigen::Vector3d acc = imu_vec_.at(i).at(j)->acc_ - state_est_vec_.at(i)->b_acc_;

        Eigen::Quaterniond q1 = quat_positive(q0 * Exp_q(dt_ * gyr));
        // Eigen::Vector3d v1 = v0 + dt_ * (q0.toRotationMatrix()* acc + gravity);
        // Eigen::Vector3d p1 = p0 + dt_ * v0 + 0.5 * dt_*dt_ * (q0.toRotationMatrix()* acc + gravity);

        Eigen::Matrix<double, 9, 9> F = Eigen::Matrix<double, 9, 9>::Zero();
        F.block<3,3>(0,0) = Exp(dt_*gyr).transpose();
        F.block<3,3>(3,0) = (-1)*dt_*q0.toRotationMatrix()*Skew(acc);
        F.block<3,3>(3,3) = Eigen::Matrix3d::Identity();
        F.block<3,3>(6,0) = (-0.5)*dt_*dt_*q0.toRotationMatrix()*Skew(acc);
        F.block<3,3>(6,3) = dt_*Eigen::Matrix3d::Identity();
        F.block<3,3>(6,6) = Eigen::Matrix3d::Identity();

        F_all = F * F_all;

        Eigen::Matrix<double, 9, 6> G = Eigen::Matrix<double, 9, 6>::Zero();
        G.block<3,3>(0,0) = (-1)*dt_*Eigen::Matrix3d::Identity();
        G.block<3,3>(3,3) = (-1)*dt_*q0.toRotationMatrix();
        G.block<3,3>(6,3) = (-0.5)*dt_*dt_*q0.toRotationMatrix();

        Eigen::Matrix<double, 6, 6> w_cov = Eigen::Matrix<double, 6, 6>::Zero();
        w_cov.block<3,3>(0,0) = (sigma_g_c_*sigma_g_c_/dt_)*Eigen::Matrix3d::Identity();
        w_cov.block<3,3>(3,3) = (sigma_a_c_*sigma_a_c_/dt_)*Eigen::Matrix3d::Identity();

        q0 = q1;
        // v0 = v1;
        // p0 = p1;
        cov = F * cov * F.transpose() + G * w_cov * G.transpose();
      }


      Eigen::Quaterniond q_next = quat_positive(state_est_vec_.at(i)->q_ * Exp_q( Log_q(vio_est_vec_.at(i)->q_.conjugate()*vio_est_vec_.at(i+1)->q_)));
      Eigen::Vector3d v_next = state_est_vec_.at(i)->v_ + (vio_est_vec_.at(i+1)->v_ - vio_est_vec_.at(i)->v_);
      Eigen::Vector3d p_next = state_est_vec_.at(i)->p_ + (vio_est_vec_.at(i+1)->p_ - vio_est_vec_.at(i)->p_);



      Eigen::Matrix<double, 9, 9> C;
      C = state_est_vec_.at(i)->cov_ * F_all.transpose() * cov.inverse();

      Eigen::Matrix<double, 9, 1> residual;
      residual.block<3,1>(0,0) = Log_q(q_next.conjugate() * state_est_vec_.at(i+1)->q_);
      residual.block<3,1>(3,0) = state_est_vec_.at(i+1)->v_ - v_next;
      residual.block<3,1>(6,0) = state_est_vec_.at(i+1)->p_ - p_next;

      Eigen::Matrix<double, 9, 1> m;
      m = C * residual;

      state_est_vec_.at(i)->q_ = quat_positive(state_est_vec_.at(i)->q_ * Exp_q(m.block<3,1>(0,0)));
      state_est_vec_.at(i)->v_ = state_est_vec_.at(i)->v_ + m.block<3,1>(3,0);
      state_est_vec_.at(i)->p_ = state_est_vec_.at(i)->p_ + m.block<3,1>(6,0);

      // ignore sigma update
    }





    return true;
  }



  bool MaximizationStep() {


    // ceres parameter
    ceres::Problem                              opt_problem;
    ceres::Solver::Options                      opt_options;
    ceres::Solver::Summary                      opt_summary;

    ceres::LocalParameterization*               quat_parameterization_ptr = new ceres::QuaternionParameterization();
    ceres::LossFunction*                        loss_function_ptr = NULL; //new ceres::HuberLoss(1.0);

    opt_options.linear_solver_type = ceres::SPARSE_SCHUR;
    opt_options.minimizer_progress_to_stdout = true;
    opt_options.num_threads = 6;
    opt_options.function_tolerance = 1e-20;
    opt_options.parameter_tolerance = 1e-25;
    opt_options.max_num_iterations = 40;


    // create parameter blocks for ceres

    state_para_vec_.resize(state_est_vec_.size());
    for (size_t i=0; i<state_est_vec_.size(); ++i) {
      state_para_vec_.at(i) = new StatePara(state_est_vec_.at(i)->t_);

      state_para_vec_.at(i)->GetRotationBlock()->setEstimate(state_est_vec_.at(i)->q_);
      state_para_vec_.at(i)->GetVelocityBlock()->setEstimate(state_est_vec_.at(i)->v_);
      state_para_vec_.at(i)->GetPositionBlock()->setEstimate(state_est_vec_.at(i)->p_);
    }

    landmark_para_vec_.resize(landmark_est_vec_.size());
    for (size_t i=0; i<landmark_est_vec_.size(); ++i) {
      landmark_para_vec_.at(i) = new Vec3dParameterBlock();
      landmark_para_vec_.at(i)->setEstimate(*landmark_est_vec_.at(i));

    }


    // add parameter blocks in the optimization problem
    for (size_t i=0; i<landmark_para_vec_.size(); ++i) {
      opt_problem.AddParameterBlock(landmark_para_vec_.at(i)->parameters(), 3);
    }

    for (size_t i=0; i<state_para_vec_.size(); ++i) {
      opt_problem.AddParameterBlock(state_para_vec_.at(i)->GetRotationBlock()->parameters(), 4, quat_parameterization_ptr);
      opt_problem.AddParameterBlock(state_para_vec_.at(i)->GetVelocityBlock()->parameters(), 3);
      opt_problem.AddParameterBlock(state_para_vec_.at(i)->GetPositionBlock()->parameters(), 3);
    }
    

    // imu constraints
    for (size_t i=0; i<pre_int_imu_vec_.size(); ++i) {
      ceres::CostFunction* cost_function = new PreIntImuError(pre_int_imu_vec_.at(i)->dt_,
                                                              pre_int_imu_vec_.at(i)->dR_,
                                                              pre_int_imu_vec_.at(i)->dv_,
                                                              pre_int_imu_vec_.at(i)->dp_,
                                                              pre_int_imu_vec_.at(i)->cov_);

      opt_problem.AddResidualBlock(cost_function,
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

        size_t landmark_idx = observation_vec_.at(i).at(j)->landmark_id_-1;  

        ceres::CostFunction* cost_function = new ReprojectionError(observation_vec_.at(i).at(j)->feature_pos_,
                                                                   T_bc_,
                                                                   fu_, fv_,
                                                                   cu_, cv_,
                                                                   observation_vec_.at(i).at(j)->cov());

        opt_problem.AddResidualBlock(cost_function,
                                     loss_function_ptr,
                                     state_para_vec_.at(i+1)->GetRotationBlock()->parameters(),
                                     state_para_vec_.at(i+1)->GetPositionBlock()->parameters(),
                                     landmark_para_vec_.at(landmark_idx)->parameters());
      }
    }

    
    for (size_t i=0; i<state_para_vec_.size(); ++i) {
      opt_problem.SetParameterBlockConstant(state_para_vec_.at(i)->GetRotationBlock()->parameters());
      opt_problem.SetParameterBlockConstant(state_para_vec_.at(i)->GetVelocityBlock()->parameters());
      opt_problem.SetParameterBlockConstant(state_para_vec_.at(i)->GetPositionBlock()->parameters());    
    }


    ceres::Solve(opt_options, &opt_problem, &opt_summary);
    std::cout << opt_summary.FullReport() << "\n";


    // put the result back to estimate
    /*
    for (size_t i=1; i<state_est_vec_.size(); ++i) {
      state_est_vec_.at(i)->q_ = state_para_vec_.at(i)->GetRotationBlock()->estimate();
      state_est_vec_.at(i)->v_ = state_para_vec_.at(i)->GetVelocityBlock()->estimate();
      state_est_vec_.at(i)->p_ = state_para_vec_.at(i)->GetPositionBlock()->estimate();
    }
    */

    for (size_t i=0; i<landmark_est_vec_.size(); ++i) {
      *landmark_est_vec_.at(i) = landmark_para_vec_.at(i)->estimate();
    }

    return true;
  }

};










int main(int argc, char **argv) {

  google::InitGoogleLogging(argv[0]);

  std::string dataset = std::string(argv[1]);

  std::string config_folder_path("config/");
  std::string euroc_dataset_path = "/home/lemur/dataset/EuRoC/" + euroc_dataset_name.at(dataset) + "/mav0/";


  ExpLandmarkEmSLAM slam_problem(config_folder_path);

  // initialize the first state
  slam_problem.ReadInitialTraj("data/" + dataset + "/");

  slam_problem.ReadImuData(euroc_dataset_path + "imu0/data.csv");
  slam_problem.ReadObservationData("data/" + dataset + "/");

  boost::posix_time::ptime begin_time = boost::posix_time::microsec_clock::local_time();

  slam_problem.ExpectationStep(0.08);    // 0.06 for MH 03
  slam_problem.MaximizationStep();
  slam_problem.ExpectationStep(0.08);


  boost::posix_time::ptime end_time = boost::posix_time::microsec_clock::local_time();
  boost::posix_time::time_duration t = end_time - begin_time;
  double dt = ((double)t.total_nanoseconds() * 1e-9);

  std::cout << "The entire time is " << dt << " sec." << std::endl;

  slam_problem.OutputResult("result/" + dataset + "/traj_em.csv");

  return 0;
}