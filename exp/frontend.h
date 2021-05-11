
#ifndef INCLUDE_FRONTEND_H_
#define INCLUDE_FRONTEND_H_

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <cassert>

#include <opencv2/core.hpp>
#include <opencv2/core/cvstd.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

class Camera {

 public:
  Camera(double du, double dv,
         double fu, double fv, double pu, double pv,
         double dis_para_0, double dis_para_1, double dis_para_2, double dis_para_3) {
    
    du_ = du;
    dv_ = dv;

    fu_ = fu;
    fv_ = fv;

    pu_ = pu;
    pv_ = pv;

    k1_ = dis_para_0;
    k2_ = dis_para_1;
    p1_ = dis_para_2;
    p2_ = dis_para_3;
  }

  cv::Mat K() {
    
    cv::Mat ret_K(3, 3, CV_64F, cv::Scalar(0));
    
    ret_K.at<double>(0,0) = fu_;
    ret_K.at<double>(1,1) = fv_;
    ret_K.at<double>(0,2) = pu_;
    ret_K.at<double>(1,2) = pv_;
    ret_K.at<double>(2,2) = 1;

    return ret_K;
  }

  cv::Mat GetDistortionCoeff() {
    cv::Mat distortion_coeff(4, 1, CV_64F, cv::Scalar(0));

    distortion_coeff.at<double>(0) = k1_;
    distortion_coeff.at<double>(1) = k2_;
    distortion_coeff.at<double>(2) = p1_;
    distortion_coeff.at<double>(3) = p2_;

    return distortion_coeff;
  }

  bool isInside(cv::Point2f pt) {

    if ((pt.x >= (1-boarder_ratio_)*du_) || (pt.x <= boarder_ratio_*du_))
      return false;

    if ((pt.y >= (1-boarder_ratio_)*dv_) || (pt.y <= boarder_ratio_*dv_))
      return false;

    return true;
  }

 private:

  double du_;
  double dv_;

  double boarder_ratio_ = 0.15;

  double fu_;
  double fv_;
  double pu_;
  double pv_;

  double k1_;
  double k2_;
  double p1_;
  double p2_;
};

class FeatureNode {
 public: 
  FeatureNode() {
    landmark_id_ = 0;
  }

  bool AddNeighbor(FeatureNode* feature_node_ptr) {
    neighbors_.push_back(feature_node_ptr);
    return true;
  }

  bool IsNeighborEmpty() {
    return neighbors_.empty();
  }

  size_t GetLandmarkId() {
    return landmark_id_;
  }

  void SetLandmarkId(const size_t new_landmark_id) {
    landmark_id_ = new_landmark_id;
  }  

  bool AssignLandmarkId(const size_t input_landmark_id) {
    if (input_landmark_id == 0) {
      std::cout << "invalid landmark id." << std::endl;
      return false;
    }
    else if (input_landmark_id == landmark_id_) {
      return true;
    }    
    else if (landmark_id_ == 0) {
      landmark_id_ = input_landmark_id;
      
      for (size_t i=0; i<neighbors_.size(); ++i) {
        neighbors_.at(i)->AssignLandmarkId(input_landmark_id);
      }

      return true;
    }
    else {   // input_landmark_id != landmark_id_ 
      std::cout << "The same landmark is assigned 2 different id." << std::endl;
      return false;
    }
  }


  bool ResetLandmarkId() {
    if (landmark_id_ == 0) {
      return true;
    }
    else {
      landmark_id_ = 0;

      for (size_t i=0; i<neighbors_.size(); ++i) {
        neighbors_.at(i)->ResetLandmarkId();
      }

      return true;
    }
  }  

 private:
  size_t landmark_id_;
  std::vector<FeatureNode *> neighbors_;   
};


typedef std::map<size_t, FeatureNode*> LandmarkTable;

struct Keyframe {
  cv::Mat img_;
  std::string timestamp_;

  std::vector<cv::KeyPoint> keypoints_;
  cv::Mat descriptions_;

  LandmarkTable landmark_table_;
};

struct KeypointMatch {

  KeypointMatch(size_t first_time_id, size_t first_kp_id, size_t second_time_id, size_t second_kp_id) {
    assert(("index error!", first_time_id < second_time_id));

    first_time_id_ = first_time_id;
    first_kp_id_ = first_kp_id;
    second_time_id_ = second_time_id;
    second_kp_id_ = second_kp_id;
  }

  size_t first_time_id_;
  size_t first_kp_id_;
  size_t second_time_id_;
  size_t second_kp_id_;
};

class Frontend {
 public:

  Frontend(cv::FileStorage config_file) {
    camera_ptr_ = new Camera((double) config_file["cameras"][0]["image_dimension"][0],
                             (double) config_file["cameras"][0]["image_dimension"][1],
                             (double) config_file["cameras"][0]["focal_length"][0], 
                             (double) config_file["cameras"][0]["focal_length"][1],
                             (double) config_file["cameras"][0]["principal_point"][0],
                             (double) config_file["cameras"][0]["principal_point"][1],
                             (double) config_file["cameras"][0]["distortion_coefficients"][0],
                             (double) config_file["cameras"][0]["distortion_coefficients"][1],
                             (double) config_file["cameras"][0]["distortion_coefficients"][2],
                             (double) config_file["cameras"][0]["distortion_coefficients"][3]);

    landmark_obs_threshold_ = 1;
  }

  bool AddKeyframe(Keyframe keyframe) {
    keyframes_.push_back(keyframe);

    return true;
  }



  bool DetectAndComputeKeypoints(cv::Ptr<cv::FeatureDetector> detector) {
    for (size_t i=0; i<keyframes_.size(); ++i) {

      detector->detectAndCompute(keyframes_.at(i).img_, cv::noArray(),
                                 keyframes_.at(i).keypoints_, 
                                 keyframes_.at(i).descriptions_);   

      /***
      cv::Mat img_w_keypoints;
      cv::drawKeypoints(keyframes_.at(i).img_, keyframes_.at(i).keypoints_, img_w_keypoints);
      cv::imshow(keyframes_.at(i).timestamp_ , img_w_keypoints);
      cv::waitKey();
      ***/


    }

    return true;
  }

  // matches_raw -> matches_distance -> matches_ransac
  bool Match(cv::Ptr<cv::DescriptorMatcher> matcher) {

    for (size_t i=0; i<keyframes_.size(); ++i) {
      for (size_t j=i+1; j<keyframes_.size(); ++j) {
      

        std::vector<cv::DMatch> matches_raw;
        std::vector<cv::DMatch> matches_distance;
        std::vector<cv::DMatch> matches_ransac;
  
        matcher->match(keyframes_.at(i).descriptions_, keyframes_.at(j).descriptions_, matches_raw);

        // matches_raw -> matches_distance
        for (size_t k=0; k<matches_raw.size(); k++) {
          if (matches_raw[k].distance < 40) {
            if (camera_ptr_->isInside(keyframes_.at(i).keypoints_[matches_raw[k].queryIdx].pt)) {
              if (camera_ptr_->isInside(keyframes_.at(j).keypoints_[matches_raw[k].trainIdx].pt)) {
                matches_distance.push_back(matches_raw[k]);

              }
            }
          }
        }

        if (matches_distance.size() < 20)
          continue;

        // matches_distance -> matches_ransac
        std::vector<cv::Point2f> src_points;
        std::vector<cv::Point2f> dst_points;

        std::vector<cv::Point2f> undis_src_point;
        std::vector<cv::Point2f> undis_dst_point;

        for (size_t k=0; k<matches_distance.size(); k++) {
          src_points.push_back(keyframes_.at(i).keypoints_[matches_distance[k].queryIdx].pt);
          dst_points.push_back(keyframes_.at(j).keypoints_[matches_distance[k].trainIdx].pt);
        }

        cv::undistortPoints(src_points, undis_src_point, camera_ptr_->K(), camera_ptr_->GetDistortionCoeff(), cv::noArray(), cv::noArray());
        cv::undistortPoints(dst_points, undis_dst_point, camera_ptr_->K(), camera_ptr_->GetDistortionCoeff(), cv::noArray(), cv::noArray());        


        // RANSAC        
        cv::Mat mask;
        cv::Mat H = cv::findHomography(src_points, dst_points, mask, cv::RANSAC);

        for (size_t k=0; k<matches_distance.size(); k++) {
          if (mask.at<bool>(k,0)) {
            matches_ransac.push_back(matches_distance[k]);

            keypoint_matches_.push_back(KeypointMatch(i, matches_distance[k].queryIdx, j, matches_distance[k].trainIdx));
          }
        }


        std::cout << "frames " <<  i << " and " << j << std::endl;
        /***
        cv::Mat img_w_matches_raw;
        cv::drawMatches(keyframes_.at(i).img_, keyframes_.at(i).keypoints_,
                        keyframes_.at(j).img_, keyframes_.at(j).keypoints_,
                        matches_raw, img_w_matches_raw);
        cv::imshow("raw matches", img_w_matches_raw);

        cv::Mat img_w_matches_distance;
        cv::drawMatches(keyframes_.at(i).img_, keyframes_.at(i).keypoints_,
                        keyframes_.at(j).img_, keyframes_.at(j).keypoints_,
                        matches_distance, img_w_matches_distance);
        cv::imshow("matches after distance thresholding", img_w_matches_distance);

        cv::Mat img_w_matches_ransac;
        cv::drawMatches(keyframes_.at(i).img_, keyframes_.at(i).keypoints_,
                        keyframes_.at(j).img_, keyframes_.at(j).keypoints_,
                        matches_ransac, img_w_matches_ransac);
        cv::imshow("matches after RANSAC", img_w_matches_ransac);

        cv::waitKey();
        ***/

      }  
    }   

    AssignLandmarkId();

    return true;
  }


  bool AssignLandmarkId() {
    if (keypoint_matches_.empty())
      return false;

    // Step 1: set up the graph
    for (size_t i=0; i<keypoint_matches_.size(); ++i) {
      size_t first_time_id = keypoint_matches_.at(i).first_time_id_;
      size_t first_kp_id = keypoint_matches_.at(i).first_kp_id_;
      size_t second_time_id = keypoint_matches_.at(i).second_time_id_;
      size_t second_kp_id = keypoint_matches_.at(i).second_kp_id_;


      if (keyframes_.at(first_time_id).landmark_table_.find(first_kp_id) == keyframes_.at(first_time_id).landmark_table_.end()) {
        keyframes_.at(first_time_id).landmark_table_[first_kp_id] = new FeatureNode();
      }

      if (keyframes_.at(second_time_id).landmark_table_.find(second_kp_id) == keyframes_.at(second_time_id).landmark_table_.end()) {
        keyframes_.at(second_time_id).landmark_table_[second_kp_id] = new FeatureNode();
      }

      FeatureNode* first_feature_ptr = keyframes_.at(first_time_id).landmark_table_[first_kp_id];
      FeatureNode* second_feature_ptr = keyframes_.at(second_time_id).landmark_table_[second_kp_id];

      first_feature_ptr->AddNeighbor(second_feature_ptr);
      second_feature_ptr->AddNeighbor(first_feature_ptr);
    }


    // Step 2: naively assign landmark id
    size_t landmark_count = 0;

    for (size_t i=0; i<keyframes_.size(); ++i) {
      for (LandmarkTable::iterator it=keyframes_.at(i).landmark_table_.begin(); it!=keyframes_.at(i).landmark_table_.end(); ++it) {

        if (it->second->GetLandmarkId()==0) {
          landmark_count++;
          it->second->AssignLandmarkId(landmark_count);
        }
      }
    }

    std::cout << "number of landmarks: " << landmark_count << std::endl;

    
    // Step 3: check each landmark id only appears once in each keyframe
    std::set<size_t> multiply_assigned_id;

    for (size_t i=0; i<keyframes_.size(); ++i) {
      std::map<size_t, size_t> landmark_id_count_table;        // landmark id, counts

      for (LandmarkTable::iterator it=keyframes_.at(i).landmark_table_.begin(); it!=keyframes_.at(i).landmark_table_.end(); ++it) {
        size_t landmark_id = it->second->GetLandmarkId();
        assert(("landmark id 0 is detected!", landmark_id > 0));

        if (landmark_id_count_table.find(landmark_id)==landmark_id_count_table.end()) {
          landmark_id_count_table[landmark_id] = 0;
        }
        
        landmark_id_count_table[landmark_id]++;
      }

      for (std::map<size_t, size_t>::iterator it=landmark_id_count_table.begin(); it!=landmark_id_count_table.end(); ++it) {
        if (it->second >= 2) {
          multiply_assigned_id.insert(it->first);
        }
      }
    }

    std::cout << "multiply assigned id number = " << multiply_assigned_id.size() << std::endl;


    // Step 4: remove those multiply assigned landmark id
    landmark_obs_count_.resize(landmark_count);

    for (size_t i=0; i<keyframes_.size(); ++i) {
      for (LandmarkTable::iterator it=keyframes_.at(i).landmark_table_.begin(); it!=keyframes_.at(i).landmark_table_.end(); ++it) {
        
        size_t landmark_id = it->second->GetLandmarkId();
        
        if (landmark_id!= 0) {
          if (multiply_assigned_id.find(landmark_id)!=multiply_assigned_id.end()) {
            it->second->ResetLandmarkId();
          }
          else {
            landmark_obs_count_.at(landmark_id-1)++;
          }
        }
        // (landmark_id == 0) are those been reset 
      }
    }

    // Step 5: check landmark observation count distribution
    std::map<size_t, size_t> obs_count_table;
    for (size_t i=0; i<landmark_obs_count_.size(); ++i) {
      if (obs_count_table.find(landmark_obs_count_.at(i)) == obs_count_table.end()) {
        obs_count_table[landmark_obs_count_.at(i)] = 0;
      }

      obs_count_table[landmark_obs_count_.at(i)]++;
    }

    return true;
  }

 
  // create a landmark id table, and undistrot the keypoints
  bool OutputLandmarkObservation(std::ostream& output_stream) {

    std::map<size_t, size_t> landmark_id_remap;
    size_t landmark_count = 0;

    double fu = camera_ptr_->K().at<double>(0,0);
    double fv = camera_ptr_->K().at<double>(1,1);
    double pu = camera_ptr_->K().at<double>(0,2);
    double pv = camera_ptr_->K().at<double>(1,2);

    // header
    output_stream << "timestamp,landmark_id,u,v,size" << std::endl; 


    for (size_t i=0; i<keyframes_.size(); ++i) {
      for (LandmarkTable::iterator it=keyframes_.at(i).landmark_table_.begin(); it!=keyframes_.at(i).landmark_table_.end(); ++it) {
        size_t kp_id = it->first;
        size_t lm_id = it->second->GetLandmarkId();


        if (lm_id > 0 && landmark_obs_count_[lm_id-1] > landmark_obs_threshold_) {

          // reassign landmark id
          if (landmark_id_remap.find(lm_id) == landmark_id_remap.end()) {
            landmark_count++;
            landmark_id_remap[lm_id] = landmark_count;
          }

          // undistort
          std::vector<cv::Point2f> point;
          point.push_back(keyframes_.at(i).keypoints_[kp_id].pt);
          std::vector<cv::Point2f> undis_point;
          cv::undistortPoints(point, undis_point, camera_ptr_->K(), camera_ptr_->GetDistortionCoeff(), cv::noArray(), camera_ptr_->K());

          // ouptput stream
          output_stream << keyframes_.at(i).timestamp_ << "," << landmark_id_remap[lm_id] << ","
                        << undis_point.at(0).x << "," << undis_point.at(0).y << ","
                        << keyframes_.at(i).keypoints_[kp_id].size
                        << std::endl;          
        }
      }
    }

    std::cout << "output landmark numbers: " << landmark_count << std::endl;

    return true;
  }


 private:
 
  Camera* camera_ptr_;
  std::vector<Keyframe> keyframes_;

  std::vector<KeypointMatch> keypoint_matches_;
  std::vector<size_t> landmark_obs_count_;

  size_t landmark_obs_threshold_;

};

#endif 