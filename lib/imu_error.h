

/**
 * @file imu_error.hpp
 * @brief Header file for the ImuError class.
 * @author Tsang-Kai Chang
 */


#ifndef INCLUDE_IMU_ERROR_H_
#define INCLUDE_IMU_ERROR_H_

#include "so3.h"


/// \brief Implements a nonlinear IMU factor.
class ImuError :
    public ceres::SizedCostFunction<9,     // number of residuals
        4,                         // number of parameters in q_{t+1}
        3,                         // number of parameters in v_{t+1}
        3,                         // number of parameters in p_{t+1}
        4,                         // number of parameters in q_t
        3,                         // number of parameters in v_t
        3> {                       // number of parameters in p_t
        // 3,                         // number of parameters of gyro bias
        // 3> {                        // number of parameters of accel bias
 public:

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /// \brief The base in ceres we derive from
  typedef ceres::SizedCostFunction<9, 4, 3, 3, 4, 3, 3> base_t;

  /// \brief The number of residuals
  static const int kNumResiduals = 9;

  /// \brief The type of the covariance.
  typedef Eigen::Matrix<double, 9, 9> covariance_t;

  /// \brief The type of the information (same matrix dimension as covariance).
  // typedef covariance_t information_t;

  /// \brief The type of hte overall Jacobian.
  // typedef Eigen::Matrix<double, 15, 15> jacobian_t;

  /// \brief The type of the Jacobian w.r.t. poses --
  /// \warning This is w.r.t. minimal tangential space coordinates...
  // typedef Eigen::Matrix<double, 15, 7> jacobian0_t;

  /// \brief The type of Jacobian w.r.t. Speed and biases
  // typedef Eigen::Matrix<double, 15, 9> jacobian1_t;

  /// \brief Default constructor -- assumes information recomputation.
  ImuError() {
  }

  /// \brief Trivial destructor.
  ~ImuError() {
  }

  // TODO: remove the default values
  ImuError(const Eigen::Vector3d gyr_measurement,
           const Eigen::Vector3d acc_measurement,
           const double dt,
           const Eigen::Vector3d gyr_bias = Eigen::Vector3d(0,0,0),
           const Eigen::Vector3d acc_bias = Eigen::Vector3d(0,0,0),
           const double sigma_g_c = 12.0e-4,    // gyro noise density [rad/s/sqrt(Hz)]
           const double sigma_a_c = 8.0e-3) {   // accelerometer noise density [m/s^2/sqrt(Hz)]

    gyr_ = gyr_measurement;
    acc_ = acc_measurement;

    dt_ = dt;

    gyr_bias_ = gyr_bias;
    acc_bias_ = acc_bias;

    sigma_g_c_ = sigma_g_c;
    sigma_a_c_ = sigma_a_c;
  }






  // error term and Jacobian implementation
  /**
   * @brief This evaluates the error term and additionally computes the Jacobians.
   * @param parameters Pointer to the parameters (see ceres)
   * @param residuals Pointer to the residual vector (see ceres)
   * @param jacobians Pointer to the Jacobians (see ceres)
   * @return success of th evaluation.
   */
  bool Evaluate(double const* const * parameters, 
                double* residuals,
                double** jacobians) const {
    
    Eigen::Vector3d gravity = Eigen::Vector3d(0, 0, -9.81007);      

    Eigen::Quaterniond q_t1(parameters[0][0], parameters[0][1], parameters[0][2], parameters[0][3]);
    Eigen::Vector3d v_t1(parameters[1]);
    Eigen::Vector3d p_t1(parameters[2]);
    Eigen::Quaterniond q_t0(parameters[3][0], parameters[3][1], parameters[3][2], parameters[3][3]);
    Eigen::Vector3d v_t0(parameters[4]);
    Eigen::Vector3d p_t0(parameters[5]);

    // residual vectors
    Eigen::Map<Eigen::Vector3d > r_q(residuals+0);      
    Eigen::Map<Eigen::Vector3d > r_v(residuals+3);      
    Eigen::Map<Eigen::Vector3d > r_p(residuals+6);      

    Eigen::Quaterniond q_diff = Exp_q(dt_*(gyr_-gyr_bias_));
    Eigen::Vector3d acc_plus_gravity = q_t0.toRotationMatrix()*(acc_ - acc_bias_) + gravity;
    Eigen::Vector3d v_diff = dt_* acc_plus_gravity;
    Eigen::Vector3d p_diff = dt_*v_t0 + (0.5*dt_*dt_)* acc_plus_gravity;

    r_q = Log_q((q_t0*Exp_q(dt_*(gyr_-gyr_bias_))).conjugate() * q_t1);
    r_v = v_t1 - (v_t0 + v_diff);   // avoid a minus sign
    r_p = p_t1 - (p_t0 + p_diff);


    // covariance
    Eigen::Matrix<double, 6, 6> odometry_cov;
    odometry_cov.setZero();
    odometry_cov.block<3,3>(0,0) = (sigma_g_c_*sigma_g_c_/dt_) * Eigen::Matrix3d::Identity();
    odometry_cov.block<3,3>(3,3) = (sigma_a_c_*sigma_a_c_/dt_) * Eigen::Matrix3d::Identity();

    Eigen::Matrix<double, 9, 6> state_2_odometry;
    state_2_odometry.setZero();
    state_2_odometry.block<3,3>(0,0) = dt_ * RightJacobian(dt_*(gyr_-gyr_bias_));
    state_2_odometry.block<3,3>(3,3) = dt_ * q_t0.toRotationMatrix();
    state_2_odometry.block<3,3>(6,3) = 0.5 * dt_ * dt_ * q_t0.toRotationMatrix();

    /***
    covariance_t covariance;
    covariance = state_2_odometry * odometry_cov * state_2_odometry.transpose();

    Eigen::LLT<covariance_t> lltOfInformation(covariance.inverse());
    covariance_t squareRootInformation_ = lltOfInformation.matrixL().transpose();

    // noise normalization
    r_q = squareRootInformation_.block<3,3>(0,0) * r_q;
    r_v = squareRootInformation_.block<3,3>(3,3) * r_v;
    r_p = squareRootInformation_.block<3,3>(6,6) * r_p;
    ***/


    // covariance adjustment
    double gyr_input_sigma = sigma_g_c_ / sqrt(dt_);
    double acc_input_sigma = sigma_a_c_ / sqrt(dt_);

    double q_noise_sigma = (-1) * dt_ * gyr_input_sigma;
    double v_noise_sigma = (-1) * dt_ * acc_input_sigma;
    double p_noise_sigma = (-0.5) * dt_ * dt_ * acc_input_sigma;


    r_q = (1.0 / q_noise_sigma) * r_q;
    r_v = (1.0 / v_noise_sigma) * r_v;
    r_p = (1.0 / p_noise_sigma) * r_p;

    /*********************************************************************************
                 Jacobian
    *********************************************************************************/

    Eigen::Matrix3d dR_;
    dR_ = Exp(dt_*(gyr_-gyr_bias_));
    Eigen::Vector3d dv_ = dt_ * q_t0.toRotationMatrix() * (acc_ - acc_bias_);
    Eigen::Vector3d dp_ = 0.5 * dt_ * dt_ * q_t0.toRotationMatrix() * (acc_ - acc_bias_);   


    if (jacobians != NULL) {

      // rotation_t1
      if (jacobians[0] != NULL) {
        Eigen::Map<Eigen::Matrix<double, 9, 4, Eigen::RowMajor> > J_q_t1(jacobians[0]);      
        J_q_t1.setZero();

        Eigen::Matrix<double, 3, 3> J_res_q_2_q1 = LeftJacobianInv(r_q) * (q_t0.toRotationMatrix()*dR_).transpose();
      
        J_q_t1.block<3,4>(0,0) = (1.0 / q_noise_sigma)  * J_res_q_2_q1 * QuatLiftJacobian(q_t1);
      }  

      // velocity_t1
      if (jacobians[1] != NULL) {
        Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor> > J_v_t1(jacobians[1]);      
        J_v_t1.setZero();

        Eigen::Matrix<double, 3, 3> J_res_v_2_v1 = Eigen::Matrix3d::Identity();

        J_v_t1.block<3,3>(3,0) = (1.0 / v_noise_sigma)  * J_res_v_2_v1;
      }  


      // position_t1
      if (jacobians[2] != NULL) {
        Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor> > J_p_t1(jacobians[2]);      
        J_p_t1.setZero();

        Eigen::Matrix<double, 3, 3> J_res_v_2_p1 = Eigen::Matrix3d::Identity();

        J_p_t1.block<3,3>(6,0) = (1.0 / p_noise_sigma)  * J_res_v_2_p1;
      }


      // rotation_t
      if (jacobians[3] != NULL) {
        Eigen::Map<Eigen::Matrix<double, 9, 4, Eigen::RowMajor> > J_q_t0(jacobians[3]);      
        J_q_t0.setZero();

        Eigen::Matrix<double, 3, 3> J_res_q_2_q0 = LeftJacobianInv(Log((q_t0.toRotationMatrix()*dR_).transpose() * q_t1)) * (-1) * (q_t0.toRotationMatrix()*dR_).transpose();
        Eigen::Matrix<double, 3, 3> J_res_v_2_q0 = Skew(q_t0.toRotationMatrix() * dv_);
        Eigen::Matrix<double, 3, 3> J_res_p_2_q0 = Skew(q_t0.toRotationMatrix() * dp_);

        J_q_t0.block<3,4>(0,0) = (1.0 / q_noise_sigma) * J_res_q_2_q0 * QuatLiftJacobian(q_t0);
        J_q_t0.block<3,4>(3,0) = (1.0 / v_noise_sigma) * J_res_v_2_q0 * QuatLiftJacobian(q_t0);
        J_q_t0.block<3,4>(6,0) = (1.0 / p_noise_sigma) * J_res_p_2_q0 * QuatLiftJacobian(q_t0);
      }

      // velocity_t
      if (jacobians[4] != NULL) {
        Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor> > J_v_t0(jacobians[4]);
        J_v_t0.setZero();

        Eigen::Matrix<double, 3, 3> J_res_v_2_v0 = (-1) * Eigen::Matrix3d::Identity();
        Eigen::Matrix<double, 3, 3> J_res_p_2_v0 = (-dt_) * Eigen::Matrix3d::Identity();

        J_v_t0.block<3,3>(3,0) = (1.0 / v_noise_sigma) * J_res_v_2_v0;
        J_v_t0.block<3,3>(6,0) = (1.0 / p_noise_sigma) * J_res_p_2_v0;
      }  

      // position_t
      if (jacobians[5] != NULL) {
        Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor> > J_p_t0(jacobians[5]);      
        J_p_t0.setZero();

        Eigen::Matrix<double, 3, 3> J_res_p_2_p0 = (-1) * Eigen::Matrix3d::Identity();

        J_p_t0.block<3,3>(6,0) = (1.0 / p_noise_sigma) * J_res_p_2_p0;
      }  
    }

    return true;
  }


  // sizes
  /// \brief Residual dimension.
  size_t residualDim() const {
    return kNumResiduals;
  }

  /// \brief Number of parameter blocks.
  size_t parameterBlocks() const {
    return parameter_block_sizes().size();
  }

  /// \brief Dimension of an individual parameter block.
  /// @param[in] parameterBlockId ID of the parameter block of interest.
  /// \return The dimension.
  size_t parameterBlockDim(size_t parameterBlockId) const {
    return base_t::parameter_block_sizes().at(parameterBlockId);
  }

  /// @brief Return parameter block type as string
  std::string typeInfo() const {
    return "ImuError";
  }

 protected:

  // measurements
  Eigen::Vector3d gyr_;
  Eigen::Vector3d acc_; 

  Eigen::Vector3d gyr_bias_;
  Eigen::Vector3d acc_bias_;

  // times
  double dt_;

  double sigma_g_c_;    // gyro noise density [rad/s/sqrt(Hz)]
  double sigma_a_c_;    // accelerometer noise density [m/s^2/sqrt(Hz)]

};

#endif /* INCLUDE_IMU_ERROR_H_ */