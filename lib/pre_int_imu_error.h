
/**
 * @file pre_int_imu_error.hpp
 * @brief Header file for the ImuError class.
 * @author Tsang-Kai Chang
 */

// TODO(tsangkai): Implement bias parameters (interface, residual, and jacobian)


#ifndef INCLUDE_PRE_INT_IMU_ERROR_H_
#define INCLUDE_PRE_INT_IMU_ERROR_H_


#include <ceres/ceres.h>

#include "so3.h"


/// \brief Implements a nonlinear IMU factor.
class PreIntImuError :
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
  PreIntImuError() {
  }

  /// \brief Trivial destructor.
  ~PreIntImuError() {
  }

  /// 
  PreIntImuError(const double dt,
                 const Eigen::Matrix3d d_rotation,
                 const Eigen::Vector3d d_velocity,
                 const Eigen::Vector3d d_position,
                 const covariance_t cov = covariance_t::Identity()) {
    dt_ = dt;

    dR_ = d_rotation;
    dv_ = d_velocity;
    dp_ = d_position;

    cov_ = cov;
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

    Eigen::Quaterniond q_t1(parameters[0][0], parameters[0][1], parameters[0][2], parameters[0][3]);
    Eigen::Vector3d v_t1(parameters[1]);
    Eigen::Vector3d p_t1(parameters[2]);
    Eigen::Quaterniond q_t0(parameters[3][0], parameters[3][1], parameters[3][2], parameters[3][3]);
    Eigen::Vector3d v_t0(parameters[4]);
    Eigen::Vector3d p_t0(parameters[5]);

    // TODO: define globally
    Eigen::Vector3d gravity = Eigen::Vector3d(0, 0, -9.81007);

    // residual vectors
    Eigen::Map<Eigen::Vector3d > r_q(residuals+0);      
    Eigen::Map<Eigen::Vector3d > r_v(residuals+3);      
    Eigen::Map<Eigen::Vector3d > r_p(residuals+6);

    r_q = Log((q_t0.toRotationMatrix()*dR_).transpose() * q_t1);
    r_v = v_t1 - (q_t0.toRotationMatrix()*dv_ + v_t0 + gravity*dt_);
    r_p = p_t1 - (q_t0.toRotationMatrix()*dp_ + p_t0 + v_t0*dt_ + 0.5*gravity*dt_*dt_);

    // covariance
    covariance_t delta_2_residual;
    delta_2_residual.setZero();
    delta_2_residual.block<3,3>(0,0) = (-1) * dR_;
    delta_2_residual.block<3,3>(3,3) = (-1) * q_t0.toRotationMatrix();
    delta_2_residual.block<3,3>(6,6) = (-1) * q_t0.toRotationMatrix();

    Eigen::LLT<covariance_t> lltOfInformation(cov_.inverse());
    covariance_t squareRootInformation_ = delta_2_residual * lltOfInformation.matrixL().transpose();

    // weight
    r_q = squareRootInformation_.block<3,3>(0,0) * r_q;
    r_v = squareRootInformation_.block<3,3>(3,3) * r_v;
    r_p = squareRootInformation_.block<3,3>(6,6) * r_p;

    /*********************************************************************************
                 Jacobian
    *********************************************************************************/

    if (jacobians != NULL) {

      // rotation_t1
      if (jacobians[0] != NULL) {
        Eigen::Map<Eigen::Matrix<double, 9, 4, Eigen::RowMajor> > J_q_t1(jacobians[0]);      
        J_q_t1.setZero();

        Eigen::Matrix<double, 3, 3> J_res_q_2_q1 = LeftJacobianInv(r_q) * (q_t0.toRotationMatrix()*dR_).transpose();
      
        J_q_t1.block<3,4>(0,0) = squareRootInformation_.block<3,3>(0,0) * J_res_q_2_q1 * QuatLiftJacobian(q_t1);
      }  

      // velocity_t1
      if (jacobians[1] != NULL) {

        Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor> > J_v_t1(jacobians[1]);
        J_v_t1.setZero();

        Eigen::Matrix<double, 3, 3> J_res_v_2_v1 = Eigen::Matrix3d::Identity();

        J_v_t1.block<3,3>(3,0) = squareRootInformation_.block<3,3>(3,3) * J_res_v_2_v1;
      }  

      // position_t1
      if (jacobians[2] != NULL) {
        Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor> > J_p_t1(jacobians[2]);      
        J_p_t1.setZero();

        Eigen::Matrix<double, 3, 3> J_res_v_2_p1 = Eigen::Matrix3d::Identity();

        J_p_t1.block<3,3>(6,0) = squareRootInformation_.block<3,3>(6,6) * J_res_v_2_p1;
      }

      // rotation_t
      if (jacobians[3] != NULL) {
        Eigen::Map<Eigen::Matrix<double, 9, 4, Eigen::RowMajor> > J_q_t0(jacobians[3]);      
        J_q_t0.setZero();

        Eigen::Matrix<double, 3, 3> J_res_q_2_q0 = LeftJacobianInv(Log((q_t0.toRotationMatrix()*dR_).transpose() * q_t1)) * (-1) * (q_t0.toRotationMatrix()*dR_).transpose();
        Eigen::Matrix<double, 3, 3> J_res_v_2_q0 = Skew(q_t0.toRotationMatrix() * dv_);
        Eigen::Matrix<double, 3, 3> J_res_p_2_q0 = Skew(q_t0.toRotationMatrix() * dp_);

        J_q_t0.block<3,4>(0,0) = squareRootInformation_.block<3,3>(0,0) * J_res_q_2_q0 * QuatLiftJacobian(q_t0);
        J_q_t0.block<3,4>(3,0) = squareRootInformation_.block<3,3>(3,3) * J_res_v_2_q0 * QuatLiftJacobian(q_t0);
        J_q_t0.block<3,4>(6,0) = squareRootInformation_.block<3,3>(6,6) * J_res_p_2_q0 * QuatLiftJacobian(q_t0);
      }  

      // velocity_t
      if (jacobians[4] != NULL) {
        Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor> > J_v_t0(jacobians[4]);
        J_v_t0.setZero();

        Eigen::Matrix<double, 3, 3> J_res_v_2_v0 = (-1) * Eigen::Matrix3d::Identity();
        Eigen::Matrix<double, 3, 3> J_res_p_2_v0 = (-dt_) * Eigen::Matrix3d::Identity();

        J_v_t0.block<3,3>(3,0) = squareRootInformation_.block<3,3>(3,3) * J_res_v_2_v0;
        J_v_t0.block<3,3>(6,0) = squareRootInformation_.block<3,3>(6,6) * J_res_p_2_v0;
      }  

      // position_t
      if (jacobians[5] != NULL) {
        Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor> > J_p_t0(jacobians[5]);      
        J_p_t0.setZero();

        Eigen::Matrix<double, 3, 3> J_res_p_2_p0 = (-1) * Eigen::Matrix3d::Identity();

        J_p_t0.block<3,3>(6,0) = squareRootInformation_.block<3,3>(6,6) * J_res_p_2_p0;
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
    return "PreIntImuError";
  }

 protected:

  // times
  double dt_;
  
  // measurements
  Eigen::Matrix3d dR_;
  Eigen::Vector3d dv_;
  Eigen::Vector3d dp_;

  covariance_t cov_;

};

#endif /* INCLUDE_PRE_INT_IMU_ERROR_H_ */