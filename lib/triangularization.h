
#ifndef INCLUDE_TRIANGULARIZATION_H_
#define INCLUDE_TRIANGULARIZATION_H_

#include <Eigen/Core>

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
  Eigen::Matrix4d A;
  A.row(0) = kp1(0) * P1.row(2) - P1.row(0);
  A.row(1) = kp1(1) * P1.row(2) - P1.row(1);
  A.row(2) = kp2(0) * P2.row(2) - P2.row(0);
  A.row(3) = kp2(1) * P2.row(2) - P2.row(1);

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Vector4d vec = svd.matrixV().col(3);

  return vec.head(3) / vec(3);
}



#endif /* INCLUDE_TRIANGULARIZATION_H_ */