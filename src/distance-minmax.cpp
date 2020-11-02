#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/ml.hpp"
#include <filesystem>
#include <string>

int main (int argc, char * const argv[]) {
    cv::CommandLineParser parser(argc, argv, 
        "{@vocab | somevocab_1000.yml | clustered bow}"
        "{@dir | ../data/small | input image dir}");
    
    cv::FileStorage fs(parser.get<std::string>("@vocab"),
                        cv::FileStorage::READ);
    std::string dir = parser.get<std::string>("@dir");
    cv::Mat vocab;
    fs["vocabulary"] >> vocab;
    fs.release();

    std::shared_ptr<cv::SIFT> detector = 
        cv::SIFT::create();
    std::shared_ptr<cv::DescriptorExtractor> descriptor = 
       cv::xfeatures2d::LATCH::create();

    std::shared_ptr<cv::DescriptorMatcher> matcher = 
        cv::BFMatcher::create(cv::NormTypes::NORM_HAMMING);
    matcher->add(std::vector<cv::Mat>(1, vocab));
    cv::Mat freq_hist, img;
    cv::Mat training_descriptors, label;
    float min = 999;
    float max = 0;

    for (const auto & entry : std::filesystem::recursive_directory_iterator(dir)){
        img = cv::imread(entry.path());
        std::vector<cv::KeyPoint> kpts;
        detector->detect(img, kpts);
        cv::Mat desc;
        descriptor->compute(img, kpts, desc);
        std::vector<cv::DMatch> matches;
        matcher->match(desc, matches);
        for (const auto & match : matches){
            //std::cout << match.distance << std::endl;
            if(min > match.distance) min = match.distance;
            if(max < match.distance) max = match.distance;
        }
    }
    std::cout << min << " " << max << std::endl;
}