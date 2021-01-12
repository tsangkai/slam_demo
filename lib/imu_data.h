
#ifndef INCLUDE_IMU_DATA_H_
#define INCLUDE_IMU_DATA_H_

#include <Eigen/Core>

struct IMUData {
 EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  IMUData(double timestamp, Eigen::Vector3d gyr, Eigen::Vector3d acc) {
    timestamp_ = timestamp;
    gyr_ = gyr;
    acc_ = acc;
  }

  double timestamp_;
  Eigen::Vector3d gyr_;
  Eigen::Vector3d acc_; 
};


struct PreIntIMUData {
 EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  PreIntIMUData(Eigen::Vector3d bias_gyr,
                Eigen::Vector3d bias_acc,
                double sigma_g_c, 
                double sigma_a_c) {

    bias_gyr_ = bias_gyr;
    bias_acc_ = bias_acc;

    sigma_g_c_ = sigma_g_c;
    sigma_a_c_ = sigma_a_c;

    dt_ = 0;

    dR_ = Eigen::Matrix3d::Identity();
    dv_ = Eigen::Vector3d(0, 0, 0);
    dp_ = Eigen::Vector3d(0, 0, 0);

    cov_.setZero();
  }
  
  // the imu_data is measured at imu_data.dt_
  // and this imu_data lasts for imu_dt
  // assume this imu_data is constant over the interval
  bool IntegrateSingleIMU(IMUData imu_data, double imu_dt) {

    Eigen::Vector3d gyr = imu_data.gyr_ - bias_gyr_;
    Eigen::Vector3d acc = imu_data.acc_ - bias_acc_;



    // covariance update
    Eigen::Matrix<double, 9, 9> F = Eigen::Matrix<double, 9, 9>::Identity();
    F.block<3,3>(3,0) = (-1) * dR_ * Hat(acc) * imu_dt;
    F.block<3,3>(6,0) = (-0.5) * dR_ * Hat(acc)*imu_dt*imu_dt;
    F.block<3,3>(6,3) = imu_dt * Eigen::Matrix3d::Identity();

    Eigen::Matrix<double, 9, 6> G;
    G.setZero();
    G.block<3,3>(0,0) = dR_.transpose()*LeftJacobian(gyr*imu_dt)*imu_dt;
    G.block<3,3>(3,3) = dR_.transpose()*imu_dt;
    G.block<3,3>(6,3) = 0.5*dR_.transpose()*imu_dt*imu_dt;

    Eigen::Matrix<double, 6, 6> Q = Eigen::Matrix<double, 6, 6>::Identity();
    Q.block<3,3>(0,0) = (sigma_g_c_ * sigma_g_c_ / imu_dt) * Eigen::Matrix3d::Identity();
    Q.block<3,3>(3,3) = (sigma_a_c_ * sigma_a_c_ / imu_dt) * Eigen::Matrix3d::Identity();

    cov_ = F * cov_ * F.transpose() + G * Q * G.transpose();

    // deviation update
    dp_ = dp_ + imu_dt * dv_ + 0.5 * (imu_dt * imu_dt) * dR_ * acc;
    dv_ = dv_ + imu_dt * dR_ * acc;
    dR_ = dR_ * Exp(imu_dt * gyr);

    dt_ = dt_ + imu_dt;

    return true;
  }

  Eigen::Vector3d bias_gyr_;
  Eigen::Vector3d bias_acc_;

  double sigma_g_c_;
  double sigma_a_c_;

  double dt_;
  Eigen::Matrix3d dR_;  
  Eigen::Vector3d dv_;
  Eigen::Vector3d dp_; 

  Eigen::Matrix<double, 9, 9> cov_;
};

#endif /* INCLUDE_IMU_DATA_H_ */