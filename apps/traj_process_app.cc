// This code with generate the ground truth for visualization
// It first needs keyframe time.
// This file also takes care of the transformation between estimation and the groundturth.
// Therefore, the visual odometry output is required.


#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>


struct State {
  std::string t_;

  Eigen::Quaterniond q_;
  Eigen::Vector3d v_;
  Eigen::Vector3d p_;
};


bool ReadTrajFile(std::ifstream& file_stream, std::vector<State*>& vec) {
  std::string data_line_str;
  std::getline(file_stream, data_line_str);


  while (std::getline(file_stream, data_line_str)) {

    std::stringstream data_str_stream(data_line_str); 

    if (data_str_stream.good()) {
      std::string data_str;
      std::getline(data_str_stream, data_str, ',');

      std::string time = data_str;

      double position[3];
      for (size_t i=0; i<3; ++i) {
        std::getline(data_str_stream, data_str, ','); 
        position[i] = std::stod(data_str);
      }

      double velocity[3];
      for (size_t i=0; i<3; ++i) {
        std::getline(data_str_stream, data_str, ','); 
        velocity[i] = std::stod(data_str);
      }

      double rotation[4];
      for (size_t i=0; i<4; ++i) {
        std::getline(data_str_stream, data_str, ','); 
        rotation[i] = std::stod(data_str);
      }

      State* new_state_ptr = new State;

      new_state_ptr->t_ = time;
      new_state_ptr->q_ = Eigen::Quaterniond(rotation[0], rotation[1], rotation[2], rotation[3]);
      new_state_ptr->v_ = Eigen::Vector3d(velocity);
      new_state_ptr->p_ = Eigen::Vector3d(position);

      vec.push_back(new_state_ptr);
    }  

  }

  return true;
}

int main(int argc, char **argv) {


  std::string dataset = std::string(argv[1]);
  std::string traj_name = std::string(argv[2]);


  // Step 1: read two files
  std::ifstream ref_file("result/" + dataset + "/traj_gt.csv");
  std::ifstream input_file("result/" + dataset + "/" + traj_name + ".csv");

  std::vector<State*> ref_state_vec;
  std::vector<State*> input_state_vec;

  ReadTrajFile(ref_file, ref_state_vec);
  ReadTrajFile(input_file, input_state_vec);

  ref_file.close();
  input_file.close();

  // Step 2: 

  std::cout << "ref vec size: \t" << ref_state_vec.size() << std::endl;
  std::cout << "input vec size: \t" << input_state_vec.size() << std::endl;

  Eigen::Vector3d ref_p_mean(0,0,0);
  Eigen::Vector3d input_p_mean(0,0,0);

  double one_over_size = 1.0 / (double) ref_state_vec.size();

  for (size_t i=0; i<ref_state_vec.size(); ++i) {
    ref_p_mean += one_over_size * ref_state_vec.at(i)->p_;
    input_p_mean += one_over_size * input_state_vec.at(i)->p_;
  }

  std::cout << "ref p mean: \t" << ref_p_mean << std::endl;
  std::cout << "input p mean: \t" << input_p_mean << std::endl;

  Eigen::MatrixXd ref_stacked_vec(3,ref_state_vec.size());
  Eigen::MatrixXd input_stacked_vec(3,ref_state_vec.size());

  for (size_t i=0; i<ref_state_vec.size(); ++i) {
    ref_stacked_vec.block<3,1>(0,i) = ref_state_vec.at(i)->p_ - ref_p_mean;
    input_stacked_vec.block<3,1>(0,i) = input_state_vec.at(i)->p_ - input_p_mean;
  }

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(input_stacked_vec*ref_stacked_vec.transpose(), Eigen::ComputeFullU | Eigen::ComputeFullV);

  // the transformation
  Eigen::Matrix3d R;
  Eigen::Vector3d t;

  if ((svd.matrixV() * svd.matrixU().transpose()).determinant() > 0) {
    R = svd.matrixV() * svd.matrixU().transpose();
  }
  else {
    Eigen::Matrix3d sigma = Eigen::Matrix3d::Identity();
    sigma(2,2) = -1;
    R = svd.matrixV() * sigma * svd.matrixU().transpose();
  }

  t = ref_p_mean - R * input_p_mean;

  // Step 4. Output the result
  std::ofstream out_file("result/" + dataset + "/" + traj_name + "_x.csv");
  out_file << "timestamp,p_x,p_y,p_z,v_x,v_y,v_z,q_w,q_x,q_y,q_z\n";

  for (size_t i=0; i<input_state_vec.size(); ++i) {
    
    Eigen::Quaterniond q = Eigen::Quaterniond(R) * input_state_vec.at(i)->q_;
    Eigen::Vector3d v = R * input_state_vec.at(i)->v_;
    Eigen::Vector3d p = R * input_state_vec.at(i)->p_ + t;



    out_file << input_state_vec.at(i)->t_ << ",";
    out_file << std::to_string(p(0)) << ","
                << std::to_string(p(1)) << ","
                << std::to_string(p(2)) << ","
                << std::to_string(v(0)) << ","
                << std::to_string(v(1)) << ","
                << std::to_string(v(2)) << ","
                << std::to_string(q.w()) << ","
                << std::to_string(q.x()) << ","
                << std::to_string(q.y()) << ","
                << std::to_string(q.z()) << std::endl;

  }
  out_file.close();  

  return 0;
}