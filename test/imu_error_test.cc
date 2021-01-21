#include <vector>
#include <cassert>
#include <cmath>
#include <iostream>

#include <ceres/ceres.h>
#include <Eigen/Core>

#include "transformation.h"
#include "so3.h"
#include "vec_3d_parameter_block.h"
#include "quat_parameter_block.h"
#include "imu_error.h"

#define _USE_MATH_DEFINES


struct ImuParameters{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Transformation T_BS; ///< Transformation from Body frame to IMU (sensor frame S).
  double a_max;        ///< Accelerometer saturation. [m/s^2]
  double g_max;        ///< Gyroscope saturation. [rad/s]
  double sigma_g_c;    ///< Gyroscope noise density.
  double sigma_bg;     ///< Initial gyroscope bias.
  double sigma_a_c;    ///< Accelerometer noise density.
  double sigma_ba;     ///< Initial accelerometer bias
  double sigma_gw_c;   ///< Gyroscope drift noise density.
  double sigma_aw_c;   ///< Accelerometer drift noise density.
  double tau;          ///< Reversion time constant of accerometer bias. [s]
  double g;            ///< Earth acceleration.
  Eigen::Vector3d a0;  ///< Mean of the prior accelerometer bias.
  int rate;            ///< IMU rate in Hz.
};

struct IMUData {
 EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  double timestamp_;
  Eigen::Vector3d gyro_;
  Eigen::Vector3d accel_; 
};


// This class only handles the memory. The underlying estimate is handled by each parameter blocks.
class State {

 public:
  State(double timestamp) {
    timestamp_ = timestamp;

    rotation_block_ptr_ = new QuatParameterBlock();
    velocity_block_ptr_ = new Vec3dParameterBlock();
    position_block_ptr_ = new Vec3dParameterBlock();
  }

  ~State() {
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

 private:
  double timestamp_;
  QuatParameterBlock* rotation_block_ptr_;
  Vec3dParameterBlock* velocity_block_ptr_;
  Vec3dParameterBlock* position_block_ptr_;
};



int main(int argc, char **argv) {
  srand((unsigned int) time(0));

  google::InitGoogleLogging(argv[0]);

  // parameters
  const double dt = 0.01;                                     // time discretization  
  const size_t step = 10;
  Eigen::Vector3d gravity = Eigen::Vector3d(0, 0, -9.81007);      
  double sigma_g_c = 6.0e-4;
  double sigma_a_c = 2.0e-3;

  // generate random motion
  const double m_omega_S_x = Eigen::internal::random(0.1,1.0); // magnitude
  const double m_omega_S_y = Eigen::internal::random(0.1,1.0); // magnitude
  const double m_omega_S_z = Eigen::internal::random(0.1,1.0); // magnitude  
  const double m_a_W_x = Eigen::internal::random(0.1,10.0);
  const double m_a_W_y = Eigen::internal::random(0.1,10.0);
  const double m_a_W_z = Eigen::internal::random(0.1,10.0);


  Eigen::Vector3d omega_S(m_omega_S_x,
                          m_omega_S_y,
                          m_omega_S_z);

  Eigen::Vector3d a_W(m_a_W_x,
                      m_a_W_y,
                      m_a_W_z);


  // starting states
  Transformation T_WS;
  T_WS.SetRandom(10.0, M_PI);

  double t0 = 0;
  Eigen::Quaterniond q0 = T_WS.q();
  Eigen::Vector3d    v0(0.1, 0.1, 0.1);  
  Eigen::Vector3d    p0 = T_WS.t();



  Eigen::Quaterniond dq = Exp_q(omega_S*dt);


  // generate measurements
  Eigen::Vector3d gyr_noise = sigma_g_c/sqrt(dt) * Eigen::Vector3d::Random();
  Eigen::Vector3d acc_noise = sigma_a_c/sqrt(dt) * Eigen::Vector3d::Random();

  Eigen::Vector3d gyr = omega_S + gyr_noise;
  Eigen::Vector3d acc = q0.conjugate().toRotationMatrix()*(a_W - gravity) + acc_noise;
  
  IMUData imu_data;

  imu_data.timestamp_ = t0;
  imu_data.gyro_ = gyr;
  imu_data.accel_ = acc;


  // ending state
  double t1 = t0 + dt;  

  Eigen::Quaterniond q1 = q0 * dq;
  Eigen::Vector3d v1 = v0 + dt*a_W;
  Eigen::Vector3d p1 = p0 + dt*v0 + 0.5*dt*dt*a_W;

  Transformation T_WS_1(q1, p1);



  //=========================================================================================================


  // Build the problem.
  ceres::Problem optimization_problem;
  ceres::LocalParameterization* quat_parameterization_ptr_ = new ceres::QuaternionParameterization();

  // create the pose parameter blocks
  Transformation T_disturb;
  T_disturb.SetRandom(10, 0.2);
  Transformation T_WS_1_disturbed = Transformation(q1, p1) * T_disturb;
  Eigen::Vector3d v1_disturbed = v1 + 5*Eigen::Vector3d::Random();



  State* state_0 = new State(t0);
  State* state_1 = new State(t1);


  optimization_problem.AddParameterBlock(state_0->GetRotationBlock()->parameters(), 4, quat_parameterization_ptr_);
  optimization_problem.AddParameterBlock(state_0->GetVelocityBlock()->parameters(), 3);
  optimization_problem.AddParameterBlock(state_0->GetPositionBlock()->parameters(), 3);

  optimization_problem.AddParameterBlock(state_1->GetRotationBlock()->parameters(), 4, quat_parameterization_ptr_);
  optimization_problem.AddParameterBlock(state_1->GetVelocityBlock()->parameters(), 3);
  optimization_problem.AddParameterBlock(state_1->GetPositionBlock()->parameters(), 3);


  state_0->GetRotationBlock()->setEstimate(q0);
  state_0->GetVelocityBlock()->setEstimate(v0);
  state_0->GetPositionBlock()->setEstimate(p0);

  state_1->GetRotationBlock()->setEstimate(T_WS_1_disturbed.q());
  state_1->GetVelocityBlock()->setEstimate(v1_disturbed);
  state_1->GetPositionBlock()->setEstimate(T_WS_1_disturbed.t());

  optimization_problem.SetParameterBlockConstant(state_0->GetRotationBlock()->parameters());
  optimization_problem.SetParameterBlockConstant(state_0->GetVelocityBlock()->parameters());
  optimization_problem.SetParameterBlockConstant(state_0->GetPositionBlock()->parameters());


  // add constraints
  ceres::CostFunction* cost_function = new ImuError(gyr,
                                                    acc,
                                                    dt,
                                                    Eigen::Vector3d(0,0,0),
                                                    Eigen::Vector3d(0,0,0),                   
                                                    sigma_g_c,
                                                    sigma_a_c);

  optimization_problem.AddResidualBlock(cost_function,
                                        NULL,
                                        state_1->GetRotationBlock()->parameters(),
                                        state_1->GetVelocityBlock()->parameters(),
                                        state_1->GetPositionBlock()->parameters(),
                                        state_0->GetRotationBlock()->parameters(),
                                        state_0->GetVelocityBlock()->parameters(),
                                        state_0->GetPositionBlock()->parameters());   

  // Run the solver!
  std::cout << "run the solver... " << std::endl;
  ceres::Solver::Options optimization_options;
  ceres::Solver::Summary optimization_summary;

  ceres::Solve(optimization_options, &optimization_problem, &optimization_summary);


  // print some infos about the optimization
  std::cout << optimization_summary.FullReport() << "\n";


  Transformation T_WS_1_opt = Transformation(state_1->GetRotationBlock()->estimate(), state_1->GetPositionBlock()->estimate());
  Eigen::Vector3d v1_opt = state_1->GetVelocityBlock()->estimate();

  std::cout << "initial T_WS_1 : " << T_WS_1_disturbed.T() << "\n"
            << "optimized T_WS_1 : " << T_WS_1_opt.T() << "\n"
            << "correct T_WS_1 : " << T_WS_1.T() << "\n";

  std::cout << "rotation difference of the initial T_nb : " << 2*(T_WS_1.q() * T_WS_1_disturbed.q().inverse()).vec().norm() << "\n";
  std::cout << "rotation difference of the optimized T_nb : " << 2*(T_WS_1.q() * T_WS_1_opt.q().inverse()).vec().norm() << "\n";

  std::cout << "velocity difference of the initial T_nb : " << (v1 - v1_disturbed).norm() << "\n";
  std::cout << "velocity difference of the optimized T_nb : " << (v1 - v1_opt).norm() << "\n";

  std::cout << "translation difference of the initial T_nb : " << (T_WS_1.t() - T_WS_1_disturbed.t()).norm() << "\n";
  std::cout << "translation difference of the optimized T_nb : " << (T_WS_1.t() - T_WS_1_opt.t()).norm() << "\n";

  return 0;
}
