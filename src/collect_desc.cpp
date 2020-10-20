#include <iostream>
#include <filesystem>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

int main( int argc, char* argv[]) {
    cv::CommandLineParser parser(argc, argv, 
        "{@positive_dir | ../data/positive | positive images directory}"
        "{@negative_dir | ../data/negative | negative images directory}"
        "{@collection_file | training_descriptors | output filename}");
    auto positive_dir = parser.get<std::string>("@positive_dir");
    auto negative_dir = parser.get<std::string>("@negative_dir");
    std::cout << positive_dir << std::endl;

    std::shared_ptr<cv::SIFT> detector = 
        cv::SIFT::create();
    std::shared_ptr<cv::xfeatures2d::LATCH> descriptor = 
        cv::xfeatures2d::LATCH::create();

    cv::Mat training_desriptors(
        1,
        descriptor->descriptorSize(),
        descriptor->descriptorType()
    );
    
    std::vector<cv::KeyPoint> kpts;
    cv::Mat img, desc;

    for (const auto & entry : std::filesystem::recursive_directory_iterator(positive_dir)){
        img = cv::imread(entry.path());
        detector->detect(img, kpts);
        descriptor->compute(img, kpts, desc);
        training_desriptors.push_back(desc);
        std::cout << ".";
    }

    for (const auto & entry : std::filesystem::recursive_directory_iterator(negative_dir)){
        img = cv::imread(entry.path());
        detector->detect(img, kpts);
        descriptor->compute(img, kpts, desc);
        training_desriptors.push_back(desc);
        std::cout << ".";
    }
    std::cout << std::endl;

    std::cout << "Total descriptors: " << training_desriptors.rows << std::endl;
    auto filename = parser.get<std::string>("@collection_file");
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    fs << "training_descriptors" << training_desriptors;
    fs.release();
    return 0;
}