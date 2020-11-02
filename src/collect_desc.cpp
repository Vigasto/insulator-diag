#include <iostream>
#include <filesystem>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

int main( int argc, char* argv[]) {
    cv::CommandLineParser parser(argc, argv, 
        "{@dir | ../data/small | images directory}"
        "{@collection_file | training_descriptors | output filename}");
    auto dir = parser.get<std::string>("@dir");
    std::cout << dir << std::endl;

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

    for (const auto & entry : std::filesystem::recursive_directory_iterator(dir)){
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