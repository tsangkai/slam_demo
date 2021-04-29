
#ifndef INCLUDE_SO3_H_
#define INCLUDE_SO3_H_


#include <cmath>


const double eps = 1e-8;

double sinc(double x){
  if(fabs(x) > eps) {
    return sin(x)/x;
  }
  else {
    static const double c_2=1.0/6.0;
    static const double c_4=1.0/120.0;
    static const double c_6=1.0/5040.0;
    const double x_2 = x*x;
    const double x_4 = x_2*x_2;
    const double x_6 = x_2*x_2*x_2;
    
    return 1.0 - c_2*x_2 + c_4*x_4 - c_6*x_6;
  }
}


Eigen::Matrix3d Skew(Eigen::Vector3d v) {
  Eigen::Matrix3d m;
  m <<     0, -v(2),  v(1),
        v(2),     0, -v(0),
       -v(1),  v(0),     0;

  return m;
}


Eigen::Matrix3d Hat(Eigen::Vector3d v) {
  Eigen::Matrix3d m;
  m <<     0, -v(2),  v(1),
        v(2),     0, -v(0),
       -v(1),  v(0),     0;

  return m;
}


Eigen::Quaterniond quat_postive(const Eigen::Quaterniond & q_input){
  Eigen::Quaterniond q = q_input;
  if (q.w() < 0) {
      q.w() = (-1)*q.w();
      q.x() = (-1)*q.x();
      q.y() = (-1)*q.y();
      q.z() = (-1)*q.z();
  }
  return q;
}


Eigen::Matrix3d Exp(Eigen::Vector3d omega) {

  if (omega.norm() < eps) {
    return Eigen::Matrix3d::Identity() + Hat(omega);
  }
  else {
    double omega_norm = omega.norm();
    Eigen::Matrix3d hatted_omega = Hat((1/omega_norm)*omega);

    return Eigen::Matrix3d::Identity() + sin(omega_norm) * hatted_omega + (1 - cos(omega_norm)) * hatted_omega * hatted_omega;
  }
}


// from ORB SLAM 3
Eigen::Vector3d Log(Eigen::Matrix3d R) {

    double tr = R(0,0)+R(1,1)+R(2,2);
    Eigen::Vector3d w;
    w  << 0.5 * (R(2,1)-R(1,2)),
          0.5 * (R(0,2)-R(2,0)),
          0.5 * (R(1,0)-R(0,1));

    double costheta = (tr-1.0f)*0.5f;

    if(costheta>1 || costheta<-1)
        return w;

    const double theta = acos(costheta);
    const double s = sin(theta);
    if(fabs(s)<eps)
        return w;
    else
        return theta*w/s;
}


// [Sola] (101)
// [Bloesch et al] (38)
Eigen::Quaterniond Exp_q(const Eigen::Vector3d v) {
  Eigen::Quaterniond q;

  const double v_half_norm = 0.5 * v.norm();
  const double sinc_v_half_norm = sinc(v_half_norm);
  const double cos_v_half_norm = cos(v_half_norm);

  q.w() = cos_v_half_norm;  
  q.vec() = 0.5 * sinc_v_half_norm * v;

  return q;
}

// [Sola] (105)
// [Bloesch et al] (40)
// [Kok et al] (3.39a)
Eigen::Vector3d Log_q(const Eigen::Quaterniond q) {

  Eigen::Quaterniond quat = quat_postive(q);

  double atan = atan2(quat.vec().norm(), quat.w());
  if (abs(atan) < eps) {
    return 2 * quat.vec();
  }

  return 2 * (atan / quat.vec().norm()) * quat.vec();
}


// plus() in okvis
Eigen::Matrix4d QuatLeftMul(const Eigen::Quaterniond & q) {       
  Eigen::Matrix4d Q;
  Q(0,0) = q.w(); Q(0,1) = -q.x(); Q(0,2) = -q.y(); Q(0,3) = -q.z();
  Q(1,0) = q.x(); Q(1,1) =  q.w(); Q(1,2) = -q.z(); Q(1,3) =  q.y();
  Q(2,0) = q.y(); Q(2,1) =  q.z(); Q(2,2) =  q.w(); Q(2,3) = -q.x();
  Q(3,0) = q.z(); Q(3,1) = -q.y(); Q(3,2) =  q.x(); Q(3,3) =  q.w();
  return Q;
}


// oplus() in okvis
Eigen::Matrix4d QuatRightMul(const Eigen::Quaterniond & q) {       
  Eigen::Matrix4d Q;
  Q(0,0) = q.w(); Q(0,1) = -q.x(); Q(0,2) = -q.y(); Q(0,3) = -q.z();
  Q(1,0) = q.x(); Q(1,1) =  q.w(); Q(1,2) =  q.z(); Q(1,3) = -q.y();
  Q(2,0) = q.y(); Q(2,1) = -q.z(); Q(2,2) =  q.w(); Q(2,3) =  q.x();
  Q(3,0) = q.z(); Q(3,1) =  q.y(); Q(3,2) = -q.x(); Q(3,3) =  q.w();
  return Q;
}


// directly from okvis
// Computes the Jacobian from minimal space to naively overparameterised space as used by ceres.
// Follow [Bloesch] definition
Eigen::Matrix<double, 3, 4> QuatLiftJacobian(const Eigen::Quaterniond & q) {

  Eigen::Matrix<double, 3, 4> J_lift;

  const Eigen::Quaterniond q_inv = q.conjugate();
  Eigen::Matrix4d q_inv_right_mul = QuatRightMul(q_inv);

  Eigen::Matrix<double, 3, 4> Jq_pinv;
  Jq_pinv.topLeftCorner<3, 1>().setZero();
  Jq_pinv.bottomRightCorner<3, 3>() = Eigen::Matrix3d::Identity() * 2.0;

  J_lift = Jq_pinv * q_inv_right_mul;

  return J_lift;
}

// [Chirikjian] p.40 (10.86)
Eigen::Matrix3d LeftJacobian(const Eigen::Vector3d & v) {

  Eigen::Matrix3d left_jacobian;

  const double v_norm = v.norm();
  const double v_norm_2 = v_norm*v_norm;
  const double v_norm_3 = v_norm_2*v_norm;
  const Eigen::Matrix3d skewed_v = Skew(v);

  if (v_norm > eps) {
    left_jacobian = Eigen::Matrix3d::Identity() + ((1-cos(v_norm))/v_norm_2) * skewed_v + ((v_norm-sin(v_norm))/v_norm_3) * skewed_v * skewed_v;
  }
  else {
    left_jacobian = Eigen::Matrix3d::Identity() + 0.5 * skewed_v;
  }

  return left_jacobian;
}

// [Chirikjian] p.40 (10.86)
Eigen::Matrix3d LeftJacobianInv(const Eigen::Vector3d & v) {

  Eigen::Matrix3d left_jacobian_inv;

  const double v_norm = v.norm();
  const double v_norm_2 = v_norm*v_norm;
  const Eigen::Matrix3d skewed_v = Skew(v);

  if (v_norm > eps) {
    left_jacobian_inv = Eigen::Matrix3d::Identity() - 0.5 * skewed_v + (1/v_norm_2  - (1+cos(v_norm))/(2*v_norm*sin(v_norm))) * skewed_v * skewed_v;
  }
  else {
    left_jacobian_inv = Eigen::Matrix3d::Identity() - 0.5 * skewed_v;
  }

  return left_jacobian_inv;
}


Eigen::Matrix3d RightJacobian(const Eigen::Vector3d & v) {

  Eigen::Matrix3d right_jacobian;

  const double v_norm = v.norm();
  const double v_norm_2 = v_norm*v_norm;
  const double v_norm_3 = v_norm_2*v_norm;
  const Eigen::Matrix3d skewed_v = Skew(v);

  if (v_norm > eps) {
    right_jacobian = Eigen::Matrix3d::Identity() - ((1-cos(v_norm))/v_norm_2) * skewed_v + ((v_norm-sin(v_norm))/v_norm_3) * skewed_v * skewed_v;
  }
  else {
    right_jacobian = Eigen::Matrix3d::Identity() - 0.5 * skewed_v;
  }

  return right_jacobian;
}

// [Chirikjian] p.40 (10.86)
Eigen::Matrix3d RightJacobianInv(const Eigen::Vector3d & v) {

  Eigen::Matrix3d right_jacobian_inv;

  const double v_norm = v.norm();
  const double v_norm_2 = v_norm*v_norm;
  const Eigen::Matrix3d skewed_v = Skew(v);

  if (v_norm > eps) {
    right_jacobian_inv = Eigen::Matrix3d::Identity() + 0.5 * skewed_v + (1/v_norm_2  - (1+cos(v_norm))/(2*v_norm*sin(v_norm))) * skewed_v * skewed_v;
  }
  else {
    right_jacobian_inv = Eigen::Matrix3d::Identity() + 0.5 * skewed_v;
  }

  return right_jacobian_inv;
}


#endif /* INCLUDE_SO3_H_ */