#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cassert>
#include <map>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core/cvstd.hpp>

#include "frontend.h"


std::map<std::string, std::string> euroc_dataset_name = {
  {"MH_01", "MH_01_easy"},
  {"MH_02", "MH_02_easy"},
  {"MH_03", "MH_03_medium"},
  {"MH_04", "MH_04_difficult"},
  {"MH_05", "MH_05_difficult"}
};

int main(int argc, char **argv) {

  std::string dataset = std::string(argv[1]);
  std::string euroc_dataset_path = "/home/lemur/dataset/EuRoC/" + euroc_dataset_name.at(dataset) + "/mav0/";


  std::cout << "Read visual odometry ouput data " << dataset << " ..." << std::endl;
  std::ifstream vo_data_file("data/" + dataset + "/okvis_kf.csv");
  assert(("Could not open visual odometry ouput data.", vo_data_file.is_open()));



  // the BRISK keypoint detector and descriptor extractor
  cv::FileStorage config_file("config/config_fpga_p2_euroc.yaml", cv::FileStorage::READ);
  Frontend frontend(config_file);

  // Read the column names
  // Extract the first line in the file
  std::string line;
  std::getline(vo_data_file, line);

  while (std::getline(vo_data_file, line)) {
    std::stringstream s_stream(line);                // Create a stringstream of the current line

    if (s_stream.good()) {
      std::string time_stamp_str;
      std::getline(s_stream, time_stamp_str, ',');   // get first string delimited by comma
      
      // if (gt_timestamp_start <= time_stamp_str && time_stamp_str <= gt_timestamp_end) {

        Keyframe keyframe;

        keyframe.timestamp_ = time_stamp_str;
        keyframe.img_ = cv::imread(euroc_dataset_path + "cam0/data/" + time_stamp_str + ".png", cv::IMREAD_GRAYSCALE);

        frontend.AddKeyframe(keyframe);

      // }
    }
  }

  std::cout << "Detect and compute keypoints ..." << std::endl;
  cv::Ptr<cv::FeatureDetector> brisk_detector = cv::BRISK::create(40, 0, 1.0f);
  frontend.DetectAndComputeKeypoints(brisk_detector);
  vo_data_file.close();

  std::cout << "Match keypoints ..." << std::endl;
  cv::Ptr<cv::DescriptorMatcher> bf_hamming_matcher = 
      cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);  
  frontend.Match(bf_hamming_matcher);

  std::ofstream feature_obs_file("data/" + dataset + "/feature_obs.csv");
  frontend.OutputLandmarkObservation(feature_obs_file);
  feature_obs_file.close();

  return 0;
}