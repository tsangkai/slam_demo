
#include <Eigen/Core>



class Transformation {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Transformation() {
    t_ = Eigen::Vector3d(0.0, 0.0, 0.0);
    q_ = Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0);
  }

  Transformation(const Eigen::Quaterniond& q_AB, const Eigen::Vector3d& r_AB) {
    t_ = r_AB;
    q_ = q_AB.normalized();
  }

  void SetRandom(double translationMaxMeters,
                 double rotationMaxRadians) {
    // Create a random unit-length axis.
    Eigen::Vector3d axis = rotationMaxRadians * Eigen::Vector3d::Random();

    // Create a random rotation angle in radians.
    Eigen::Vector3d t = translationMaxMeters * Eigen::Vector3d::Random();
    t_ = t;
    q_ = Eigen::AngleAxisd(axis.norm(), axis.normalized());
  }

  Eigen::Matrix4d T() const {
    Eigen::Matrix4d T_ret;
    T_ret.topLeftCorner<3, 3>() = q_.toRotationMatrix();
    T_ret.topRightCorner<3, 1>() = t_;
    T_ret.bottomLeftCorner<1, 3>().setZero();
    T_ret(3, 3) = 1.0;

    return T_ret;
  }

  Eigen::Vector3d t() {
    return t_;
  }

  Eigen::Quaterniond q() {
    return q_;
  }

  Eigen::Matrix3d C() {
    return q_.toRotationMatrix();
  }

  Transformation inverse() {

    Eigen::Quaterniond q_inv = q_.conjugate();
    Eigen::Vector3d t_inv = (-1) * q_inv.toRotationMatrix() * t_;

    return Transformation(q_inv, t_inv);
  }

    // operator*
  Transformation operator*(const Transformation & rhs) const {
    return Transformation(q_ * rhs.q_, q_.toRotationMatrix() * rhs.t_ + t_);
  }

  Transformation& operator=(const Transformation & rhs) {
    t_ = rhs.t_;
    q_ = rhs.q_;
    return *this;
  }

 protected:
  Eigen::Vector3d t_;             ///< Translation {_A}r_{B}.
  Eigen::Quaterniond q_;          ///< Quaternion q_{AB}.
};