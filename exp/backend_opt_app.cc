

#include <string>

#include <Eigen/Core>
#include <ceres/ceres.h>

#include "backend.h"



class ExpLandmarkOptSLAM: public ExpLandmarkSLAM {
 EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 public:

  ExpLandmarkOptSLAM(std::string config_folder_path): 
    ExpLandmarkSLAM(config_folder_path) {

  }




  bool SolveOptProblem() {


    // ceres parameter
    ceres::Problem                              opt_problem;
    ceres::Solver::Options                      opt_options;
    ceres::Solver::Summary                      opt_summary;

    ceres::LocalParameterization*               quat_parameterization_ptr = new ceres::QuaternionParameterization();;
    ceres::LossFunction*                        loss_function_ptr = new ceres::HuberLoss(1.0);

    opt_options.linear_solver_type = ceres::SPARSE_SCHUR;
    opt_options.minimizer_progress_to_stdout = false;
    opt_options.num_threads = 6;
    opt_options.function_tolerance = 1e-20;
    opt_options.parameter_tolerance = 1e-25;
    opt_options.max_num_iterations = 80; //100;


    // create parameter blocks for ceres

    state_para_vec_.resize(state_est_vec_.size());
    for (size_t i=0; i<state_est_vec_.size(); ++i) {
      state_para_vec_.at(i) = new StatePara(state_est_vec_.at(i)->t_);

      state_para_vec_.at(i)->GetRotationBlock()->setEstimate(state_est_vec_.at(i)->q_);
      state_para_vec_.at(i)->GetVelocityBlock()->setEstimate(state_est_vec_.at(i)->v_);
      state_para_vec_.at(i)->GetPositionBlock()->setEstimate(state_est_vec_.at(i)->p_);
    }

    landmark_para_vec_.resize(landmark_est_vec_.size());
    for (size_t i=0; i<landmark_est_vec_.size(); ++i) {
      landmark_para_vec_.at(i) = new Vec3dParameterBlock();
      landmark_para_vec_.at(i)->setEstimate(*landmark_est_vec_.at(i));

    }


    // add parameter blocks in the optimization problem
    for (size_t i=0; i<landmark_para_vec_.size(); ++i) {
      opt_problem.AddParameterBlock(landmark_para_vec_.at(i)->parameters(), 3);
    }

    for (size_t i=0; i<state_para_vec_.size(); ++i) {
      opt_problem.AddParameterBlock(state_para_vec_.at(i)->GetRotationBlock()->parameters(), 4, quat_parameterization_ptr);
      opt_problem.AddParameterBlock(state_para_vec_.at(i)->GetVelocityBlock()->parameters(), 3);
      opt_problem.AddParameterBlock(state_para_vec_.at(i)->GetPositionBlock()->parameters(), 3);
    }
    

    // imu constraints
    for (size_t i=0; i<pre_int_imu_vec_.size(); ++i) {
      ceres::CostFunction* cost_function = new PreIntImuError(pre_int_imu_vec_.at(i)->dt_,
                                                              pre_int_imu_vec_.at(i)->dR_,
                                                              pre_int_imu_vec_.at(i)->dv_,
                                                              pre_int_imu_vec_.at(i)->dp_,
                                                              pre_int_imu_vec_.at(i)->cov_);

      opt_problem.AddResidualBlock(cost_function,
                                   NULL,
                                   state_para_vec_.at(i+1)->GetRotationBlock()->parameters(),
                                   state_para_vec_.at(i+1)->GetVelocityBlock()->parameters(),
                                   state_para_vec_.at(i+1)->GetPositionBlock()->parameters(),
                                   state_para_vec_.at(i)->GetRotationBlock()->parameters(),
                                   state_para_vec_.at(i)->GetVelocityBlock()->parameters(),
                                   state_para_vec_.at(i)->GetPositionBlock()->parameters());   
    }


    // observation constraints
    for (size_t i=0; i<observation_vec_.size(); ++i) {
      for (size_t j=0; j<observation_vec_.at(i).size(); ++j) {

        size_t landmark_idx = observation_vec_.at(i).at(j)->landmark_id_-1;  

        ceres::CostFunction* cost_function = new ReprojectionError(observation_vec_.at(i).at(j)->feature_pos_,
                                                                   T_bc_,
                                                                   fu_, fv_,
                                                                   cu_, cv_,
                                                                   observation_vec_.at(i).at(j)->cov());

        opt_problem.AddResidualBlock(cost_function,
                                     loss_function_ptr,
                                     state_para_vec_.at(i+1)->GetRotationBlock()->parameters(),
                                     state_para_vec_.at(i+1)->GetPositionBlock()->parameters(),
                                     landmark_para_vec_.at(landmark_idx)->parameters());
      }
    }


    opt_problem.SetParameterBlockConstant(state_para_vec_.at(0)->GetRotationBlock()->parameters());
    opt_problem.SetParameterBlockConstant(state_para_vec_.at(0)->GetVelocityBlock()->parameters());
    opt_problem.SetParameterBlockConstant(state_para_vec_.at(0)->GetPositionBlock()->parameters());    

    ceres::Solve(opt_options, &opt_problem, &opt_summary);
    std::cout << opt_summary.FullReport() << "\n";


    // put the result back to estimate
    for (size_t i=1; i<state_est_vec_.size(); ++i) {
      state_est_vec_.at(i)->q_ = state_para_vec_.at(i)->GetRotationBlock()->estimate();
      state_est_vec_.at(i)->v_ = state_para_vec_.at(i)->GetVelocityBlock()->estimate();
      state_est_vec_.at(i)->p_ = state_para_vec_.at(i)->GetPositionBlock()->estimate();
    }

    for (size_t i=0; i<landmark_est_vec_.size(); ++i) {
      *landmark_est_vec_.at(i) = landmark_para_vec_.at(i)->estimate();
    }

    return true;
  }

};



int main(int argc, char **argv) {

  std::string dataset = std::string(argv[1]);

  std::string config_folder_path("config/");
  std::string euroc_dataset_path = "/home/lemur/dataset/EuRoC/" + euroc_dataset_name.at(dataset) + "/mav0/";

  ExpLandmarkOptSLAM slam_problem(config_folder_path);

  // initialize the first state
  slam_problem.ReadInitialTraj("data/" + dataset + "/");
  // slam_problem.OutputResult("result/" + dataset + "/traj_vo.csv");

  slam_problem.ReadImuData(euroc_dataset_path + "imu0/data.csv");
  slam_problem.ReadObservationData("data/" + dataset + "/");
  

  slam_problem.SolveOptProblem();


  slam_problem.OutputResult("result/" + dataset + "/traj_opt.csv");
  
  return 0;
}