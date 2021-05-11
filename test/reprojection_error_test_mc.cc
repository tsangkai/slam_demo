// This test file verifies reprojection_error.h
// modified from TestReprojectionError.h from okvis
// 
// Summary
// In this test file, the orientation and the positin of the agent are disturbed by noise.
// Given camera projected points from N=100 3d landmark, this test file uses reprojection_error
// to recover the real orientation and position.

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include <ceres/ceres.h>
#include <Eigen/Core>
#include <EigenRand/EigenRand>

#include "transformation.h"
#include "vec_3d_parameter_block.h"
#include "quat_parameter_block.h"
#include "reprojection_error.h"

#define _USE_MATH_DEFINES


Eigen::Vector2d Project(Eigen::Vector3d v, double fu, double fv, double cu, double cv) {
  assert(("the point is behind the camera", v[2] > 0));

  Eigen::Vector2d v_temp;  
  Eigen::Vector2d v_ret;
  
  v_temp[0] = v[0] / v[2];
  v_temp[1] = v[1] / v[2];

  v_ret[0] = fu * v_temp[0] + cu;
  v_ret[1] = fv * v_temp[1] + cv;

  return v_ret;
}


Eigen::Vector3d BackProject(Eigen::Vector2d v, double fu, double fv, double cu, double cv) {
  // unscale and center
  Eigen::Vector2d imagePoint2;
  imagePoint2[0] = (v[0] - cu) / fu;
  imagePoint2[1] = (v[1] - cv) / fv;

  Eigen::Vector3d v_ret;

  v_ret[0] = imagePoint2[0];
  v_ret[1] = imagePoint2[1];
  v_ret[2] = 1.0;

  return v_ret;
}


// Creates a random visible point in Euclidean coordinates.
Eigen::Vector3d CreateRandomVisiblePoint(double du, double dv,
                                         double fu, double fv, 
                                         double cu, double cv,
                                         double min_dist,
                                         double max_dist) {

  // Uniform random sample in image coordinates.
  // Add safety boundary for later inaccurate backprojection

  double boundary = 0.02;
  Eigen::Vector2d outPoint;
  outPoint[0] = Eigen::internal::random(boundary, du-boundary);
  outPoint[1] = Eigen::internal::random(boundary, dv-boundary);

  Eigen::Vector3d ray = BackProject(outPoint, fu, fv, cu, cv);
  ray.normalize();
  double depth = Eigen::internal::random(min_dist, max_dist);
  Eigen::Vector3d point_c = depth * ray;    // rescale

  return ray;
}


int main(int argc, char **argv) {
  srand(0);
  Eigen::Rand::Vmt19937_64 urng{ 0 };

  google::InitGoogleLogging(argv[0]);

  double du = 752.0;              // image dimension
  double dv = 480.0;
  double fu = 458.654880721;      // focal length
  double fv = 457.296696463;
  double cu = 367.215803962;      // principal point
  double cv = 248.375340610;
  // double noise_deviation = 0.8;

  const size_t N = 100;           // number of landmarks



  size_t num_trial = 100;


  std::ofstream pre_rot_error("result/test/reprojection/pre_rot_error.csv");
  std::ofstream post_rot_error("result/test/reprojection/post_rot_error.csv");
  std::ofstream pre_pos_error("result/test/reprojection/pre_pos_error.csv");
  std::ofstream post_pos_error("result/test/reprojection/post_pos_error.csv");


  pre_rot_error << "noise";
  post_rot_error << "noise";
  pre_pos_error << "noise";
  post_pos_error << "noise";

  for (size_t trial=0; trial<num_trial; ++trial) {
    pre_rot_error << "," << std::to_string(trial);
    post_rot_error << "," << std::to_string(trial);
    pre_pos_error << "," << std::to_string(trial);
    post_pos_error << "," << std::to_string(trial);
  }

  pre_rot_error << "\n";
  post_rot_error << "\n";
  pre_pos_error << "\n";
  post_pos_error << "\n";





  for (size_t noise_idx=1; noise_idx<=10; ++noise_idx) {


  double noise_deviation = 0.1 * double(noise_idx);


  pre_rot_error << std::to_string(noise_deviation);
  post_rot_error << std::to_string(noise_deviation);
  pre_pos_error << std::to_string(noise_deviation);
  post_pos_error << std::to_string(noise_deviation);



  for (size_t trial=0; trial<num_trial; ++trial) {


  std::vector<Eigen::Vector3d> landmark_vec;
  std::vector<Eigen::Vector2d> keypoint_vec;
  std::vector<Vec3dParameterBlock*> landmark_para_vec;




  // std::cout << "create simulation scenario... " << std::flush;

  Transformation T_nb;                         // navigation to body
  T_nb.SetRandom(10.0, M_PI);

  Transformation T_bc;                         // body to camera
  T_bc.SetRandom(0.2, M_PI);

  // create random visible point
  for (size_t i=0; i<N; ++i){
    double max_dist = 100;
    double min_dist = double(i%10)*3 + 2.0;

    Eigen::Vector3d landmark_c = CreateRandomVisiblePoint(du, dv, fu, fv, cu, cv, min_dist, max_dist);
    
    Eigen::Vector4d h_landmark_c(landmark_c[0], landmark_c[1], landmark_c[2], 1);
    Eigen::Vector4d h_landmark_n = T_nb.T() *T_bc.T() * h_landmark_c;
    Eigen::Vector3d landmark = h_landmark_n.head<3>();


    Eigen::Vector2d keypoint = Project(landmark_c, fu, fv, cu, cv);
    keypoint += noise_deviation * Eigen::Rand::normal<Eigen::Vector2d>(2, 1, urng);

    landmark_vec.push_back(landmark);
    keypoint_vec.push_back(keypoint);
  }

  // std::cout << " [ OK ] " << std::endl;




  // std::cout << "build the optimization problem... " << std::flush;

  ceres::Problem optimization_problem;
  ceres::Solver::Options optimization_options;
  ceres::Solver::Summary optimization_summary;

  optimization_options.max_num_iterations = 100;
  ceres::LocalParameterization* quat_parameterization_ptr = new ceres::QuaternionParameterization();


  QuatParameterBlock*  rotation_block_ptr = new QuatParameterBlock();
  Vec3dParameterBlock* position_block_ptr = new Vec3dParameterBlock();

  optimization_problem.AddParameterBlock(rotation_block_ptr->parameters(), 4, quat_parameterization_ptr);
  optimization_problem.AddParameterBlock(position_block_ptr->parameters(), 3); 

  for (size_t i=0; i<N; ++i){

    Vec3dParameterBlock* landmark_ptr = new Vec3dParameterBlock();
    landmark_para_vec.push_back(landmark_ptr);

    optimization_problem.AddParameterBlock(landmark_ptr->parameters(), 3);

    ceres::CostFunction* cost_function = new ReprojectionError(keypoint_vec.at(i),
                                                               T_bc.T(),
                                                               fu, fv,
                                                               cu, cv);
    optimization_problem.AddResidualBlock(cost_function, 
                                          NULL, 
                                          rotation_block_ptr->parameters(),
                                          position_block_ptr->parameters(),
                                          landmark_para_vec.at(i)->parameters());

  }

  // std::cout << " [ OK ] " << std::endl;





  /*
  std::cout << "\n\n  ==========================================================" << std::endl;
  std::cout << "    Set landmarks constant and add disturbance to state." << std::endl;
  std::cout << "  ==========================================================\n\n" << std::endl;
  */

  Transformation T_disturb;
  T_disturb.SetRandom(0.5, 0.1);

  Transformation T_nb_init = T_nb * T_disturb; // navigation to body

  rotation_block_ptr->setEstimate(T_nb_init.q());
  position_block_ptr->setEstimate(T_nb_init.t());


  for (size_t i=0; i<N; ++i) {
    landmark_para_vec.at(i)->setEstimate(landmark_vec.at(i));
    optimization_problem.SetParameterBlockConstant(landmark_para_vec.at(i)->parameters());
  }


  ceres::Solve(optimization_options, &optimization_problem, &optimization_summary);
  // std::cout << optimization_summary.FullReport() << "\n";


  // output the optimization result
  /*
  std::cout << "rotation difference before opt.: \t" << 2*(T_nb.q() * T_nb_init.q().inverse()).vec().norm() << "\n";
  std::cout << "rotation difference after opt.:  \t" << 2*(T_nb.q() * rotation_block_ptr->estimate().inverse()).vec().norm() << "\n\n";

  std::cout << "position difference before opt.: \t" << (T_nb.t() - T_nb_init.t()).norm() << "\n";
  std::cout << "position difference after opt.:  \t" << (T_nb.t() - position_block_ptr->estimate()).norm() << "\n\n";
  */


  pre_rot_error << "," << std::to_string(2*(T_nb.q() * T_nb_init.q().inverse()).vec().norm());
  post_rot_error << "," << std::to_string(2*(T_nb.q() * rotation_block_ptr->estimate().inverse()).vec().norm());
  pre_pos_error << "," << std::to_string((T_nb.t() - T_nb_init.t()).norm());
  post_pos_error << "," << std::to_string((T_nb.t() - position_block_ptr->estimate()).norm());


  }

  pre_rot_error << "\n";
  post_rot_error << "\n";
  pre_pos_error << "\n";
  post_pos_error << "\n";


  }

  pre_rot_error.close();
  post_rot_error.close();
  pre_pos_error.close();
  post_pos_error.close();


  return 0;
}