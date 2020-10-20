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
        "{@output | svm_param | output parameters}"
        "{@posdir | ../data/positive | input image dir}"
        "{@negdir | ../data/negative | input image dir}");
    
    cv::FileStorage fs(parser.get<std::string>("@vocab"),
                        cv::FileStorage::READ);
    std::string posdir = parser.get<std::string>("@posdir");
    std::string negdir = parser.get<std::string>("@negdir");
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
    cv::Mat freq_hist, img;
    cv::Mat training_descriptors, label;

    for (const auto & entry : std::filesystem::recursive_directory_iterator(posdir)){
        img = cv::imread(entry.path());
        std::vector<cv::KeyPoint> kpts;
        detector->detect(img, kpts);
        image_descriptor.compute(img, kpts, freq_hist);
        if (!freq_hist.empty()){
            training_descriptors.push_back(freq_hist);
            label.push_back(1);
        }
        std::cout << ".";
    }
    for (const auto & entry : std::filesystem::recursive_directory_iterator(negdir)){
        img = cv::imread(entry.path());
        std::vector<cv::KeyPoint> kpts;
        detector->detect(img, kpts);
        image_descriptor.compute(img, kpts, freq_hist);
        if (!freq_hist.empty()){
            training_descriptors.push_back(freq_hist);
            label.push_back(-1);
        }
        std::cout << ".";
    }

    std::shared_ptr<cv::ml::TrainData> train_data = 
        cv::ml::TrainData::create(
            training_descriptors,
            cv::ml::ROW_SAMPLE,
            label);

    std::cout << std::endl << "Training";
    std::shared_ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
    //svm->setType(cv::ml::SVM::C_SVC);
    //svm->setKernel(cv::ml::SVM::RBF);
    // svm->setGamma(8);
    // svm->setDegree(10);
    // svm->setCoef0(1);
    // svm->setC(10);
    // svm->setNu(0.5);
    // svm->setP(0.1);
    // svm->setTermCriteria(cv::cvTermCriteria(CV_TERMCRIT_EPS, 1000, FLT_EPSILON));
    // 
    //svm->train(training_descriptors, cv::ml::ROW_SAMPLE, label);
    svm->trainAuto(train_data);
    svm->save(parser.get<std::string>("@output"));
}