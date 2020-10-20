#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/ml.hpp"
#include <filesystem>

int main( int argc, char* argv[]) {
    cv::CommandLineParser parser(argc, argv, 
        "{@posdir | ../data/positive | input image dir}"
        "{@negdir | ../data/negative | input image dir}"
        "{@svm_par | svm_par | svm parameters}"
        "{@vocab| vocabulary_1000.yml | clustered vocab}");
    std::string posdir = parser.get<std::string>("@posdir");
    std::string negdir = parser.get<std::string>("@negdir");
    cv::Mat img;

    cv::FileStorage fs(parser.get<std::string>("@vocab"),
                        cv::FileStorage::READ);
    cv::Mat vocab;
    fs["vocabulary"] >> vocab;
    fs.release();

    std::shared_ptr<cv::SIFT> detector = 
        cv::SIFT::create();
    std::shared_ptr<cv::DescriptorExtractor> latch = 
       cv::xfeatures2d::LATCH::create();
    std::shared_ptr<cv::DescriptorMatcher> matcher = 
        cv::BFMatcher::create(cv::NormTypes::NORM_HAMMING);
    cv::BOWImgDescriptorExtractor image_descriptor(latch, matcher);
    image_descriptor.setVocabulary(vocab);

    std::shared_ptr<cv::ml::SVM> svm = 
        cv::Algorithm::load<cv::ml::SVM>(parser.get<std::string>("@svm_par"));

    std::vector<cv::KeyPoint> kpts;
    cv::Mat description;

    int tp, fp, tn, fn, p, n;
    tp = fp = tn = fn = p = n = 0;
    
    for (const auto & entry : std::filesystem::recursive_directory_iterator(posdir)){
        img = cv::imread(entry.path());
        std::vector<cv::KeyPoint> kpts;
        detector->detect(img, kpts);
        image_descriptor.compute(img, kpts, description);
        if (!description.empty()){
            int prediction = svm->predict(description);
            if (prediction == 1){
                tp+=1;
                p+=1;
            } else if (prediction == -1) {
                std::cout << "FN: " << entry.path() << std::endl;
                fn+=1;
                p+=1;
            }
        }
    }

    for (const auto & entry : std::filesystem::recursive_directory_iterator(negdir)){
        img = cv::imread(entry.path());
        std::vector<cv::KeyPoint> kpts;
        detector->detect(img, kpts);
        image_descriptor.compute(img, kpts, description);
        if (!description.empty()){
            int prediction = svm->predict(description);
            if (prediction == 1){
                std::cout << "FP: " << entry.path() << std::endl;
                fp+=1;
                n+=1;
            } else if (prediction == -1) {
                tn+=1;
                n+=1;
            }
        }
    }
    std::cout << std::endl;
    std::cout << "TP "<< tp << std::endl;
    std::cout << "TN "<< tn << std::endl;
    std::cout << "FP "<< fp << std::endl;
    std::cout << "FN "<< fn << std::endl;
    std::cout << "P "<< p << std::endl;
    std::cout << "N "<< n << std::endl;

    return 0;
}