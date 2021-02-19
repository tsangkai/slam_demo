
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


  // step 3. output the transformed ground truth

  std::ofstream out_gt_file("result/" + dataset + "/traj_gt.csv");
  out_gt_file << "timestamp,p_x,p_y,p_z,v_x,v_y,v_z,q_w,q_x,q_y,q_z\n";

  for (size_t i=0; i<keyframe_data.size(); ++i) {
    

    out_gt_file << std::setprecision(13) << std::stod(keyframe_data.at(i).timestamp_)*1e-3 << ",";
    out_gt_file << std::to_string(keyframe_data.at(i).position_(0)) << ","
                << std::to_string(keyframe_data.at(i).position_(1)) << ","
                << std::to_string(keyframe_data.at(i).position_(2)) << ","
                << std::to_string(keyframe_data.at(i).velocity_(0)) << ","
                << std::to_string(keyframe_data.at(i).velocity_(1)) << ","
                << std::to_string(keyframe_data.at(i).velocity_(2)) << ","
                << std::to_string(keyframe_data.at(i).rotation_.w()) << ","
                << std::to_string(keyframe_data.at(i).rotation_.x()) << ","
                << std::to_string(keyframe_data.at(i).rotation_.y()) << ","
                << std::to_string(keyframe_data.at(i).rotation_.z()) << std::endl;

  }
  out_gt_file.close();

  return 0;
}