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
#include <map>

#include <Eigen/Core>
#include <Eigen/Geometry>


std::map<std::string, std::string> euroc_dataset_name = {
  {"MH_01", "MH_01_easy"},
  {"MH_02", "MH_02_easy"},
  {"MH_03", "MH_03_medium"},
  {"MH_04", "MH_04_difficult"},
  {"MH_05", "MH_05_difficult"}
};

struct Data {
  std::string timestamp_;

  Eigen::Quaterniond rotation_;
  Eigen::Vector3d velocity_;
  Eigen::Vector3d position_;
};

int main(int argc, char **argv) {


  std::string dataset = std::string(argv[1]);
  std::string euroc_dataset_path = "/home/lemur/dataset/EuRoC/" + euroc_dataset_name.at(dataset) + "/mav0/";

  std::vector<Data> keyframe_data;

  // Step 1: read out_kf_time.csv
  std::ifstream kf_time_file("data/" + dataset + "/okvis_kf.csv");

  // Read the column names
  // Extract the first line in the file
  std::string first_line_data_str;
  std::getline(kf_time_file, first_line_data_str);

  std::string kf_time_str;
  while (std::getline(kf_time_file, kf_time_str)) {

    std::stringstream kf_time_str_stream(kf_time_str); 

    if (kf_time_str_stream.good()) {
      std::string data_str;
      std::getline(kf_time_str_stream, data_str, ','); 

      Data new_data;
      new_data.timestamp_ = data_str.substr(0,13);
      keyframe_data.push_back(new_data);
    }  

  }

  kf_time_file.close();



  // Step 2: read ground truth
  size_t idx = 0;
  std::ifstream gt_file(euroc_dataset_path + "state_groundtruth_estimate0/data.csv");

  std::getline(gt_file, first_line_data_str);

  std::string gt_str;
  while (std::getline(gt_file, gt_str) && idx < keyframe_data.size()) {

    std::stringstream gt_str_stream(gt_str); 

    if (gt_str_stream.good()) {
      std::string data_str;
      std::getline(gt_str_stream, data_str, ','); 

      if (keyframe_data.at(idx).timestamp_ == data_str.substr(0,13)) {

        double position[3];
        for (size_t i=0; i<3; ++i) {
          std::getline(gt_str_stream, data_str, ','); 
          position[i] = std::stod(data_str);
        }

        double rotation[4];
        for (size_t i=0; i<4; ++i) {
          std::getline(gt_str_stream, data_str, ','); 
          rotation[i] = std::stod(data_str);
        }

        double velocity[3];
        for (size_t i=0; i<3; ++i) {
          std::getline(gt_str_stream, data_str, ','); 
          velocity[i] = std::stod(data_str);
        }

        keyframe_data.at(idx).rotation_ = Eigen::Quaterniond(rotation[0], rotation[1], rotation[2], rotation[3]);
        keyframe_data.at(idx).velocity_ = Eigen::Vector3d(velocity);
        keyframe_data.at(idx).position_ = Eigen::Vector3d(position);

        idx++;
      }
    }  

  }
  
  gt_file.close();

  // step 3. obtain offset term
  Eigen::Quaterniond q0;
  Eigen::Vector3d p0;

  std::ifstream vo_file("result/" + dataset + "/traj_vo.csv");
  std::getline(vo_file, first_line_data_str);

  std::string vo_str;
  std::getline(vo_file, vo_str);

  std::stringstream vo_str_stream(vo_str); 

  if (vo_str_stream.good()) {
    std::string data_str;
    std::getline(vo_str_stream, data_str, ','); 


    double position[3];
    for (size_t i=0; i<3; ++i) {
      std::getline(vo_str_stream, data_str, ','); 
      position[i] = std::stod(data_str);
    }

    double velocity[3];
    for (size_t i=0; i<3; ++i) {
      std::getline(vo_str_stream, data_str, ','); 
      velocity[i] = std::stod(data_str);
    }

    double rotation[4];
    for (size_t i=0; i<4; ++i) {
      std::getline(vo_str_stream, data_str, ','); 
      rotation[i] = std::stod(data_str);
    }

    q0 = Eigen::Quaterniond(rotation[0], rotation[1], rotation[2], rotation[3]);
    p0 = Eigen::Vector3d(position);
  }  



  vo_file.close();




  // step 4. output the transformed ground truth

  Eigen::Quaterniond q0_gt = keyframe_data.front().rotation_;
  Eigen::Vector3d p0_gt = keyframe_data.front().position_;

  std::ofstream out_gt_file("result/" + dataset + "/traj_gt.csv");
  out_gt_file << "timestamp,p_x,p_y,p_z,v_x,v_y,v_z,q_w,q_x,q_y,q_z\n";

  for (size_t i=0; i<keyframe_data.size(); ++i) {
    
    Eigen::Quaterniond q = keyframe_data.at(i).rotation_ * q0_gt.inverse() * q0;
    Eigen::Vector3d v = q0.toRotationMatrix() * q0_gt.inverse().toRotationMatrix() * keyframe_data.at(i).velocity_;
    Eigen::Vector3d p = q0.toRotationMatrix() * q0_gt.inverse().toRotationMatrix() * (keyframe_data.at(i).position_-p0_gt) + p0;



    out_gt_file << std::setprecision(13) << std::stod(keyframe_data.at(i).timestamp_)*1e-3 << ",";
    out_gt_file << std::to_string(p(0)) << ","
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
  out_gt_file.close();

  return 0;
}