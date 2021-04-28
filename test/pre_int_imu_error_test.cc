#include <vector>
#include <cassert>
#include <cmath>
#include <iostream>

#include <ceres/ceres.h>
#include <Eigen/Core>

#include "transformation.h"
#include "so3.h"
#include "imu_data.h"
#include "vec_3d_parameter_block.h"
#include "quat_parameter_block.h"
#include "pre_int_imu_error.h"

#define _USE_MATH_DEFINES


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
  // initialize random number generator
  srand((unsigned int) time(0)); // disabled: make unit tests deterministic...


  Eigen::Vector3d gravity = Eigen::Vector3d(0, 0, -9.81007);      


  // set the imu parameters
  const double a_max = 1000.0;
  const double g_max = 1000.0;
  const size_t rate = 1000; // 1 kHz
  const double sigma_g_c = 6.0e-4;
  const double sigma_a_c = 2.0e-3;
  const double sigma_gw_c = 3.0e-6;
  const double sigma_aw_c = 2.0e-5;
  const double tau = 3600.0;

  // generate random motion
  const double w_omega_S_x = Eigen::internal::random(0.1,10.0); // circular frequency
  const double w_omega_S_y = Eigen::internal::random(0.1,10.0); // circular frequency
  const double w_omega_S_z = Eigen::internal::random(0.1,10.0); // circular frequency
  const double p_omega_S_x = Eigen::internal::random(0.0,M_PI); // phase
  const double p_omega_S_y = Eigen::internal::random(0.0,M_PI); // phase
  const double p_omega_S_z = Eigen::internal::random(0.0,M_PI); // phase
  const double m_omega_S_x = Eigen::internal::random(0.1,1.0);  // magnitude
  const double m_omega_S_y = Eigen::internal::random(0.1,1.0);  // magnitude
  const double m_omega_S_z = Eigen::internal::random(0.1,1.0);  // magnitude
  const double w_a_W_x = Eigen::internal::random(0.1,10.0);
  const double w_a_W_y = Eigen::internal::random(0.1,10.0);
  const double w_a_W_z = Eigen::internal::random(0.1,10.0);
  const double p_a_W_x = Eigen::internal::random(0.1,M_PI);
  const double p_a_W_y = Eigen::internal::random(0.1,M_PI);
  const double p_a_W_z = Eigen::internal::random(0.1,M_PI);
  const double m_a_W_x = Eigen::internal::random(0.1,10.0);
  const double m_a_W_y = Eigen::internal::random(0.1,10.0);
  const double m_a_W_z = Eigen::internal::random(0.1,10.0);

  // generate randomized measurements
  // the interval can not be too large
  // try duration = 0.3, and preintegration fails
  const double duration = 0.1;    
  Transformation T_WS;
  T_WS.SetRandom(10.0, M_PI);

  // time increment
  const double dt = 1.0/double(rate); // time discretization  0.001

  // states
  Eigen::Quaterniond q = T_WS.q();
  Eigen::Vector3d v = Eigen::Vector3d(0,0,0);
  Eigen::Vector3d p = T_WS.t();

  // start
  Eigen::Quaterniond q0;
  Eigen::Vector3d v0;
  Eigen::Vector3d p0;
  double t0;

  // end
  Eigen::Quaterniond q1;
  Eigen::Vector3d v1;
  Eigen::Vector3d p1;
  double t1;

  PreIntIMUData pre_int_imu_data(Eigen::Vector3d(0, 0, 0),
                                 Eigen::Vector3d(0, 0, 0),
                                 sigma_g_c, 
                                 sigma_a_c);

  for(size_t i=0; i < size_t(duration*rate); ++i) {
    double time = double(i)/rate;

    if (i==10){ // set this as starting pose
      q0 = q;
      v0 = v;
      p0 = p;
      t0 = time;
    }
    if (i==size_t(duration*rate)-10){ // set this as end pose
      q1 = q;
      v1 = v;
      p1 = p;
      t1 = time;
    }

    Eigen::Vector3d omega_S(m_omega_S_x*sin(w_omega_S_x*time+p_omega_S_x),
                            m_omega_S_y*sin(w_omega_S_y*time+p_omega_S_y),
                            m_omega_S_z*sin(w_omega_S_z*time+p_omega_S_z));
    Eigen::Vector3d a_W(m_a_W_x*sin(w_a_W_x*time+p_a_W_x),
                        m_a_W_y*sin(w_a_W_y*time+p_a_W_y),
                        m_a_W_z*sin(w_a_W_z*time+p_a_W_z));

    // generate imu measurements
    Eigen::Vector3d gyr_noise = sigma_g_c/sqrt(dt)*Eigen::Vector3d::Random();
    Eigen::Vector3d acc_noise = sigma_a_c/sqrt(dt)*Eigen::Vector3d::Random();

    Eigen::Vector3d gyr = omega_S + gyr_noise;
    Eigen::Vector3d acc = q.toRotationMatrix().transpose() * (a_W - gravity) + acc_noise;

    IMUData imu_data(time, gyr, acc);
    if (i >= 10 && i < size_t(duration*rate)-10) {
      pre_int_imu_data.IntegrateSingleIMU(imu_data, dt);
    }

    // state propagation
    q = q * Exp_q(dt*omega_S);
    p += dt * v + 0.5*(dt*dt) *a_W;
    v += dt * a_W;

  }

  // TEST 1
  // set gyr_noise and acc_noise to (0,0,0), and test whether preintergation is correct
  // the interval can not be too large
  /***
  std::cout << "t1 from group truth = \n" << t_1 << std::endl;
  std::cout << "q1 from group truth = \n" << q_1.coeffs() << std::endl;
  std::cout << "v1 from group truth = \n" << v_1 << std::endl;
  std::cout << "p1 from group truth = \n" << p_1 << std::endl;
  std::cout << "t1 from preintegration = \n" << t_0 + pre_int_imu_data.dt_ << std::endl;
  std::cout << "q1 from preintegration = \n" << (q_0 * Eigen::Quaterniond(pre_int_imu_data.dR_)).coeffs() << std::endl;
  std::cout << "v1 from preintegration = \n" << v_0 + gravity * pre_int_imu_data.dt_ + pre_int_imu_data.dR_*pre_int_imu_data.dv_ << std::endl;
  std::cout << "p1 from preintegration = \n" << p_0 + v_0 * pre_int_imu_data.dt_ + 0.5 * gravity * (pre_int_imu_data.dt_*pre_int_imu_data.dt_) + pre_int_imu_data.dR_*pre_int_imu_data.dp_ << std::endl;
  ***/


  //=========================================================================================================

  // Build the problem.
  ceres::Problem optimization_problem;
  ceres::LocalParameterization* quat_parameterization_ptr_ = new ceres::QuaternionParameterization();

  // create the pose parameter blocks
  Transformation T_disturb;
  T_disturb.SetRandom(1, 0.2);
  Transformation T_WS_0_disturbed = Transformation(q0, p0) * T_disturb;
  Eigen::Vector3d v0_disturbed = v0 + 5*Eigen::Vector3d::Random();
  Eigen::Vector3d p0_disturbed = p0 + 5*Eigen::Vector3d::Random();


  State* state0 = new State(t0);
  State* state1 = new State(t1);


  optimization_problem.AddParameterBlock(state0->GetRotationBlock()->parameters(), 4, quat_parameterization_ptr_);
  optimization_problem.AddParameterBlock(state0->GetVelocityBlock()->parameters(), 3);
  optimization_problem.AddParameterBlock(state0->GetPositionBlock()->parameters(), 3);

  optimization_problem.AddParameterBlock(state1->GetRotationBlock()->parameters(), 4, quat_parameterization_ptr_);
  optimization_problem.AddParameterBlock(state1->GetVelocityBlock()->parameters(), 3);
  optimization_problem.AddParameterBlock(state1->GetPositionBlock()->parameters(), 3);


  state0->GetRotationBlock()->setEstimate(T_WS_0_disturbed.q());
  state0->GetVelocityBlock()->setEstimate(v0_disturbed);
  state0->GetPositionBlock()->setEstimate(T_WS_0_disturbed.t());

  state1->GetRotationBlock()->setEstimate(q1);
  state1->GetVelocityBlock()->setEstimate(v1);
  state1->GetPositionBlock()->setEstimate(p1);

  optimization_problem.SetParameterBlockConstant(state1->GetRotationBlock()->parameters());
  optimization_problem.SetParameterBlockConstant(state1->GetVelocityBlock()->parameters());
  optimization_problem.SetParameterBlockConstant(state1->GetPositionBlock()->parameters());

  // add constraints
  ceres::CostFunction* cost_function = new PreIntImuError(pre_int_imu_data.dt_,
                                                          pre_int_imu_data.dR_,
                                                          pre_int_imu_data.dv_,
                                                          pre_int_imu_data.dp_,
                                                          pre_int_imu_data.cov_);

  optimization_problem.AddResidualBlock(cost_function,
                                        NULL,
                                        state1->GetRotationBlock()->parameters(),
                                        state1->GetVelocityBlock()->parameters(),
                                        state1->GetPositionBlock()->parameters(),
                                        state0->GetRotationBlock()->parameters(),
                                        state0->GetVelocityBlock()->parameters(),
                                        state0->GetPositionBlock()->parameters());   

  // Run the solver!
  std::cout << "run the solver... " << std::endl;
  ceres::Solver::Options optimization_options;
  ceres::Solver::Summary optimization_summary;

  ceres::Solve(optimization_options, &optimization_problem, &optimization_summary);


  // print some infos about the optimization
  std::cout << optimization_summary.FullReport() << "\n";

  
  // Transformation T_WS_1_opt = Transformation(state1->GetRotationBlock()->estimate(), state1->GetPositionBlock()->estimate());
  // Eigen::Vector3d v1_opt = state_1->GetVelocityBlock()->estimate();

  Eigen::Quaterniond q0_opt = state0->GetRotationBlock()->estimate();
  Eigen::Vector3d v0_opt = state0->GetVelocityBlock()->estimate();
  Eigen::Vector3d p0_opt = state0->GetPositionBlock()->estimate();

  /***
  std::cout << "initial T_WS_1 : " << T_WS_1_disturbed.T() << "\n"
            << "optimized T_WS_1 : " << T_WS_1_opt.T() << "\n"
            << "correct T_WS_1 : " << Transformation().T() << "\n";
  ***/

  std::cout << "rotation difference of the initial T_nb : " << 2*(q0 * T_WS_0_disturbed.q().inverse()).vec().norm() << "\n";
  std::cout << "rotation difference of the optimized T_nb : " << 2*(q0 * q0_opt.inverse()).vec().norm() << "\n";

  std::cout << "velocity difference of the initial T_nb : " << (v0 - v0_disturbed).norm() << "\n";
  std::cout << "velocity difference of the optimized T_nb : " << (v0 - v0_opt).norm() << "\n";

  std::cout << "translation difference of the initial T_nb : " << (p0 - T_WS_0_disturbed.t()).norm() << "\n";
  std::cout << "translation difference of the optimized T_nb : " << (p0 - p0_opt).norm() << "\n";

  return 0;
}