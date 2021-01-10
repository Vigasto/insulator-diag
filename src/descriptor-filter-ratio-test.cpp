#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/ml.hpp"
#include <filesystem>
#include <string>

// Utility script to visualize filtering

int main (int argc, char * const argv[]) {
    cv::CommandLineParser parser(argc, argv, 
        "{@vocab | somevocab_1000.yml | clustered bow}"
        "{@img | ../data/something.jpg | input image}");
    
    cv::FileStorage fs(parser.get<std::string>("@vocab"),
                        cv::FileStorage::READ);
    cv::Mat img = cv::imread(cv::samples::findFile(
        parser.get<std::string>( "@img")));
    cv::Mat vocab;
    fs["vocabulary"] >> vocab;
    fs.release();

    std::shared_ptr<cv::AgastFeatureDetector> detector = 
        cv::AgastFeatureDetector::create(10, true, cv::AgastFeatureDetector::OAST_9_16);
    // std::shared_ptr<cv::DescriptorExtractor> descriptor = 
    //    cv::xfeatures2d::LATCH::create();
    std::shared_ptr<cv::DescriptorExtractor> descriptor = 
       cv::SIFT::create();

    // std::shared_ptr<cv::DescriptorMatcher> matcher = 
    //     cv::BFMatcher::create(cv::NormTypes::NORM_HAMMING);
    std::shared_ptr<cv::DescriptorMatcher> matcher = 
        cv::BFMatcher::create(cv::NormTypes::NORM_L2);
    matcher->add(std::vector<cv::Mat>(1, vocab));
    cv::Mat freq_hist;
    cv::Mat training_descriptors, label;

    //TODO: tuning
    float match_ratio = 0.9;

    std::vector<cv::KeyPoint> kpts, filtered_kpts;
    detector->detect(img, kpts);
    cv::Mat desc;
    descriptor->compute(img, kpts, desc);
    std::vector<std::vector<cv::DMatch>> matches;
    matcher->knnMatch(desc, matches, 2);
    for (auto match : matches){
        int queryIdx = match[0].queryIdx;
        auto dist1 = match[0].distance;
        auto dist2 = match[1].distance;

        if (dist1 < match_ratio * dist2) {
            filtered_kpts.push_back(kpts[queryIdx]);
        }
    }
    cv::Mat img_keypoints;
    cv::drawKeypoints(img, filtered_kpts, img_keypoints);

    cv::imshow("Filtered keypoints", img_keypoints);

    cv::waitKey();
    return 0;
}