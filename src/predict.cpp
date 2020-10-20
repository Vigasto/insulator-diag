#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/ml.hpp"

int main( int argc, char* argv[]) {
    cv::CommandLineParser parser(argc, argv, 
        "{@img | ../data/something.jpg | input image}"
        "{@svm_par | svm_par | svm parameters}"
        "{@vocab| vocabulary_1000.yml | clustered vocab}");
    cv::Mat img = cv::imread(cv::samples::findFile(
        parser.get<std::string>( "@img")));

    cv::FileStorage fs(parser.get<std::string>("@vocab"),
                        cv::FileStorage::READ);
    cv::Mat vocab;
    fs["vocabulary"] >> vocab;
    fs.release();

    std::shared_ptr<cv::SIFT> detector = 
        cv::SIFT::create();
    std::shared_ptr<cv::DescriptorExtractor> descriptor = 
       cv::xfeatures2d::LATCH::create();
    std::shared_ptr<cv::DescriptorMatcher> matcher = 
        cv::BFMatcher::create(cv::NormTypes::NORM_HAMMING);
    cv::BOWImgDescriptorExtractor image_descriptor(descriptor, matcher);
    image_descriptor.setVocabulary(vocab);

    std::shared_ptr<cv::ml::SVM> svm = 
        cv::Algorithm::load<cv::ml::SVM>(parser.get<std::string>("@svm_par"));

    std::vector<cv::KeyPoint> kpts;
    detector->detect(img, kpts);
    cv::Mat description;
    image_descriptor.compute(img, kpts, description);
    if (description.empty()) return -1;
    float prediction = svm->predict(description);
    std::cout << prediction << std::endl;
    return 0;
}