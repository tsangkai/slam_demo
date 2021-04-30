#include <vector>
#include <cassert>
#include <cmath>
#include <iostream>

#include <ceres/ceres.h>
#include <Eigen/Core>
#include <EigenRand/EigenRand>

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
  Eigen::Rand::Vmt19937_64 urng{ (unsigned int) time(0) };


  Eigen::Vector3d gravity = Eigen::Vector3d(0, 0, -9.81007);      


  // set the imu parameters
  const size_t rate = 1000; // 1 kHz
  const double sigma_g_c = 6.0e-4;
  const double sigma_a_c = 2.0e-3;

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
  double t0;
  Eigen::Quaterniond q0;
  Eigen::Vector3d v0;
  Eigen::Vector3d p0;

  // end
  double t1;
  Eigen::Quaterniond q1;
  Eigen::Vector3d v1;
  Eigen::Vector3d p1;

  PreIntIMUData pre_int_imu_data(Eigen::Vector3d(0, 0, 0),
                                 Eigen::Vector3d(0, 0, 0),
                                 sigma_g_c, 
                                 sigma_a_c);

  // generate the trajectory
  for(size_t i=0; i < size_t(duration*rate); ++i) {
    double time = double(i)/rate;

    if (i==10){ // set this as starting pose
      t0 = time;
      q0 = q;
      v0 = v;
      p0 = p;
    }
    if (i==size_t(duration*rate)-10){ // set this as end pose
      t1 = time;
      q1 = q;
      v1 = v;
      p1 = p;
    }

    Eigen::Vector3d omega_S(m_omega_S_x*sin(w_omega_S_x*time+p_omega_S_x),
                            m_omega_S_y*sin(w_omega_S_y*time+p_omega_S_y),
                            m_omega_S_z*sin(w_omega_S_z*time+p_omega_S_z));
    Eigen::Vector3d a_W(m_a_W_x*sin(w_a_W_x*time+p_a_W_x),
                        m_a_W_y*sin(w_a_W_y*time+p_a_W_y),
                        m_a_W_z*sin(w_a_W_z*time+p_a_W_z));

    // generate imu measurements
    Eigen::Vector3d gyr_noise = sigma_g_c/sqrt(dt)*Eigen::Rand::normal<Eigen::Vector3d>(3, 1, urng);
    Eigen::Vector3d acc_noise = sigma_a_c/sqrt(dt)*Eigen::Rand::normal<Eigen::Vector3d>(3, 1, urng);

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


  //=========================================================================================================
  ceres::Problem optimization_problem;
  ceres::LocalParameterization* quat_parameterization_ptr_ = new ceres::QuaternionParameterization();

  State* state0 = new State(t0);
  State* state1 = new State(t1);


  optimization_problem.AddParameterBlock(state0->GetRotationBlock()->parameters(), 4, quat_parameterization_ptr_);
  optimization_problem.AddParameterBlock(state0->GetVelocityBlock()->parameters(), 3);
  optimization_problem.AddParameterBlock(state0->GetPositionBlock()->parameters(), 3);

  optimization_problem.AddParameterBlock(state1->GetRotationBlock()->parameters(), 4, quat_parameterization_ptr_);
  optimization_problem.AddParameterBlock(state1->GetVelocityBlock()->parameters(), 3);
  optimization_problem.AddParameterBlock(state1->GetPositionBlock()->parameters(), 3);

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

  ceres::Solver::Options optimization_options;
  ceres::Solver::Summary optimization_summary;

  std::cout << "\n\n";
  std::cout << "====================================================" << std::endl;
  std::cout << "Set state 1 constant and add disturbance to state 0." << std::endl;

  Transformation T_dis;
  T_dis.SetRandom(1, 0.2);
  Transformation T_nb_0_dis = Transformation(q0, p0) * T_dis;
  Eigen::Vector3d v0_dis = v0 + 5*Eigen::Vector3d::Random();
  Eigen::Vector3d p0_dis = p0 + 5*Eigen::Vector3d::Random();

  state0->GetRotationBlock()->setEstimate(T_nb_0_dis.q());
  state0->GetVelocityBlock()->setEstimate(v0_dis);
  state0->GetPositionBlock()->setEstimate(T_nb_0_dis.t());

  state1->GetRotationBlock()->setEstimate(q1);
  state1->GetVelocityBlock()->setEstimate(v1);
  state1->GetPositionBlock()->setEstimate(p1);

  optimization_problem.SetParameterBlockConstant(state1->GetRotationBlock()->parameters());
  optimization_problem.SetParameterBlockConstant(state1->GetVelocityBlock()->parameters());
  optimization_problem.SetParameterBlockConstant(state1->GetPositionBlock()->parameters());


  // solve the optimization problem
  ceres::Solve(optimization_options, &optimization_problem, &optimization_summary);
  std::cout << optimization_summary.FullReport() << "\n";

  Eigen::Quaterniond q0_opt = state0->GetRotationBlock()->estimate();
  Eigen::Vector3d v0_opt = state0->GetVelocityBlock()->estimate();
  Eigen::Vector3d p0_opt = state0->GetPositionBlock()->estimate();


  // output the optimization result
  std::cout << "rotation difference before opt.: \t" << 2*(q0 * T_nb_0_dis.q().inverse()).vec().norm() << "\n";
  std::cout << "rotation difference after opt.: \t" << 2*(q0 * q0_opt.inverse()).vec().norm() << "\n";

  std::cout << "velocity difference before opt.: \t" << (v0 - v0_dis).norm() << "\n";
  std::cout << "velocity difference after opt.: \t" << (v0 - v0_opt).norm() << "\n";

  std::cout << "position difference before opt.: \t" << (p0 - T_nb_0_dis.t()).norm() << "\n";
  std::cout << "position difference after opt.: \t" << (p0 - p0_opt).norm() << "\n";


  std::cout << "\n\n";
  std::cout << "====================================================" << std::endl;
  std::cout << "Set state 0 constant and add disturbance to state 1." << std::endl;

  T_dis.SetRandom(1, 0.2);
  Transformation T_nb_1_dis = Transformation(q1, p1) * T_dis;
  Eigen::Vector3d v1_dis = v1 + 5*Eigen::Vector3d::Random();
  Eigen::Vector3d p1_dis = p1 + 5*Eigen::Vector3d::Random();

  state0->GetRotationBlock()->setEstimate(q0);
  state0->GetVelocityBlock()->setEstimate(v0);
  state0->GetPositionBlock()->setEstimate(p0);

  state1->GetRotationBlock()->setEstimate(T_nb_1_dis.q());
  state1->GetVelocityBlock()->setEstimate(v1_dis);
  state1->GetPositionBlock()->setEstimate(T_nb_1_dis.t());

  optimization_problem.SetParameterBlockConstant(state0->GetRotationBlock()->parameters());
  optimization_problem.SetParameterBlockConstant(state0->GetVelocityBlock()->parameters());
  optimization_problem.SetParameterBlockConstant(state0->GetPositionBlock()->parameters());

  optimization_problem.SetParameterBlockVariable(state1->GetRotationBlock()->parameters());
  optimization_problem.SetParameterBlockVariable(state1->GetVelocityBlock()->parameters());
  optimization_problem.SetParameterBlockVariable(state1->GetPositionBlock()->parameters());


  // solve the optimization problem
  ceres::Solve(optimization_options, &optimization_problem, &optimization_summary);
  std::cout << optimization_summary.FullReport() << "\n";

  Eigen::Quaterniond q1_opt = state1->GetRotationBlock()->estimate();
  Eigen::Vector3d v1_opt = state1->GetVelocityBlock()->estimate();
  Eigen::Vector3d p1_opt = state1->GetPositionBlock()->estimate();

  
  // output the optimization result
  std::cout << "rotation difference before opt.: \t" << 2*(q1 * T_nb_1_dis.q().inverse()).vec().norm() << "\n";
  std::cout << "rotation difference after opt.: \t"  << 2*(q1 * q1_opt.inverse()).vec().norm() << "\n";

  std::cout << "velocity difference before opt.: \t" << (v1 - v1_dis).norm() << "\n";
  std::cout << "velocity difference after opt.: \t"  << (v1 - v1_opt).norm() << "\n";

  std::cout << "position difference before opt.: \t" << (p1 - T_nb_1_dis.t()).norm() << "\n";
  std::cout << "position difference after opt.: \t"  << (p1 - p1_opt).norm() << "\n";


  return 0;
}