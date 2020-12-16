/**
 * @file reprojection_error.h
 * @brief Header file for the ReprojectionError class.
 * @author Tsang-Kai Chang
 */

#ifndef INCLUDE_REPROJECTION_ERROR_H_
#define INCLUDE_REPROJECTION_ERROR_H_

#include <ceres/ceres.h>

#include "so3.h"



/// \brief Reprojection error base class.
class ReprojectionError:
    public ceres::SizedCostFunction<2,     // number of residuals
        4,                                 // number of parameters in q_t
        3,                                 // number of parameters in p_t
        3> {                               // number of landmark
 public:

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /// \brief The base class type.
  typedef ceres::SizedCostFunction<2, 4, 3, 3> base_t;

  /// \brief Number of residuals (2)
  static const int kNumResiduals = 2;

  /// \brief The keypoint type (measurement type).
  typedef Eigen::Vector2d measurement_t;

  /// \brief Default constructor.
  ReprojectionError();

  ReprojectionError(const measurement_t & measurement, 
                    Eigen::Matrix4d T_bc,
                    const Eigen::Matrix2d & covariance = Eigen::Matrix2d::Identity()) {
    setMeasurement(measurement);

    T_bc_ = T_bc;
    fu_ = 1.0;
    fv_ = 1.0;
    cu_ = 0.0;
    cv_ = 0.0;

    covariance_ = covariance;
  }

  ReprojectionError(const measurement_t & measurement, 
                    Eigen::Matrix4d T_bc,
                    double fu, double fv, 
                    double cu, double cv,
                    const Eigen::Matrix2d & covariance = Eigen::Matrix2d::Identity()) {
    setMeasurement(measurement);

    T_bc_ = T_bc;
    fu_ = fu;
    fv_ = fv;
    cu_ = cu;
    cv_ = cv;

    covariance_ = covariance;
  }

  /// \brief Trivial destructor.
  ~ReprojectionError() {
  }

  // setters
  /// \brief Set the measurement.
  /// @param[in] measurement The measurement.
  void setMeasurement(const measurement_t& measurement) {
    measurement_ = measurement;
  }


  /// \brief Get the measurement.
  /// \return The measurement vector.
  const measurement_t& measurement() const {
    return measurement_;
  }


  // error term and Jacobian implementation
  // (n)avigation = (w)orld
  // (b)ody frame = (s)ensor
  // (c)amera frame
  bool Evaluate(double const* const * parameters, 
                double* residuals,
                double** jacobians) const {

    // the input order of Eigen::Quaternion() is different from the underlying data structure
    Eigen::Quaterniond q_nb(parameters[0][0], parameters[0][1], parameters[0][2], parameters[0][3]);
    Eigen::Vector3d t_nb(parameters[1]);
    Eigen::Vector4d h_landmark_n(parameters[2][0], parameters[2][1], parameters[2][2], 1);
    // Eigen::Vector3d landmark(parameters[2]);

    // from body frame to camera frame
    Eigen::Matrix3d R_bc = T_bc_.topLeftCorner<3,3>();
    Eigen::Matrix3d R_cb = R_bc.transpose();
    Eigen::Vector3d t_bc = T_bc_.topRightCorner<3,1>();

    Eigen::Matrix4d T_cb = Eigen::Matrix4d::Identity();
    T_cb.topLeftCorner<3,3>() = R_cb;
    T_cb.topRightCorner<3,1>() = -R_cb * t_bc;

    // from nagivation frame to body frame
    Eigen::Matrix3d R_nb = q_nb.toRotationMatrix();
    Eigen::Matrix3d R_bn = R_nb.transpose();

    Eigen::Matrix4d T_bn = Eigen::Matrix4d::Identity();
    T_bn.topLeftCorner<3,3>() = R_bn;
    T_bn.topRightCorner<3,1>() = -R_bn * t_nb;

    // homogeneous transformation of the landmark to camera frame
    Eigen::Vector4d h_landmark_b = T_bn * h_landmark_n;
    Eigen::Vector4d h_landmark_c = T_cb * h_landmark_b;

    measurement_t keypoint;
    keypoint[0] = h_landmark_c[0] / h_landmark_c[2];
    keypoint[1] = h_landmark_c[1] / h_landmark_c[2];

    // PinholeCamera.hpp: 209
    measurement_t error;
    error[0] = fu_ * keypoint[0] + cu_ - measurement_[0];
    error[1] = fv_ * keypoint[1] + cv_ - measurement_[1];

    // covariance
    Eigen::LLT<Eigen::Matrix2d> lltOfInformation(covariance_.inverse());
    Eigen::Matrix2d squareRootInformation_ = lltOfInformation.matrixL().transpose();

    measurement_t weighted_error = squareRootInformation_ * error;

    residuals[0] = weighted_error[0];
    residuals[1] = weighted_error[1];

    /*********************************************************************************
                 Jacobian
    *********************************************************************************/


    if (jacobians != NULL) {

      Eigen::Matrix2d J_residual_to_kp;
      J_residual_to_kp << fu_, 0,
                          0, fv_;

      Eigen::MatrixXd J_kp_to_lm_c(2,3);
      double r_lm_c_2 = 1 / h_landmark_c[2];
      J_kp_to_lm_c << r_lm_c_2, 0, h_landmark_c[0]*(r_lm_c_2*r_lm_c_2),
                      0, r_lm_c_2, h_landmark_c[1]*(r_lm_c_2*r_lm_c_2);

      Eigen::Matrix3d J_lc_to_lb;
      J_lc_to_lb = R_cb;

      // chain rule
      Eigen::MatrixXd J_residual_to_lb(2,3);
      J_residual_to_lb = J_residual_to_kp * J_kp_to_lm_c * J_lc_to_lb;


      // rotation
      if (jacobians[0] != NULL) {
        Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor> > J0(jacobians[0]);

        Eigen::Vector3d landmark_minus_p = h_landmark_n.head<3>() - t_nb;
        Eigen::Matrix3d J_lb_to_dq = (-1)* Skew(R_bn *landmark_minus_p);    // [Bloesch, et. al, 2016] (27)
        
        J0 = squareRootInformation_ * J_residual_to_lb * J_lb_to_dq * QuatLiftJacobian(q_nb);
      }  


      // position
      if (jacobians[1] != NULL) {
        Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor> > J1(jacobians[1]);       

        J1 = squareRootInformation_ * J_residual_to_lb * (-1) * R_bn; 
      }  


      // landmark
      if (jacobians[2] != NULL) {
        Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor> > J2(jacobians[2]);     

        J2 = squareRootInformation_ * J_residual_to_lb * R_bn;
      }  
    }

    return true;
  }


  // sizes
  /// \brief Residual dimension.
  size_t residualDim() const {
    return kNumResiduals;
  }

  /// @brief Residual block type as string
  std::string typeInfo() const {
    return "ReprojectionError";
  }

 protected:

  // the measurement
  measurement_t measurement_; ///< The (2D) measurement.


  Eigen::Matrix4d T_bc_;
  double fu_;
  double fv_;
  double cu_;
  double cv_;
  Eigen::Matrix2d covariance_;
};

#endif /* INCLUDE_REPROJECTION_ERROR_H_ */