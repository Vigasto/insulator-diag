#include <stdlib.h>

#include <opencv2/opencv.hpp>
#include <iostream>

#include <filesystem>
#include "BOWKmajorityTrainer.h"

int main(int argc, char** argv) {
    cv::CommandLineParser parser(argc, argv, 
        "{@filename | something.yml | descriptor collection file}"
        "{@cluster_num | 1000 | cluster number}"
        "{@output | something_1000.yml | vocab cluster output}");
    auto filename = parser.get<std::string>("@filename"); 
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    cv::Mat training_desc;
    fs["training_descriptors"] >> training_desc;
    fs.release();

    auto cluster_num = parser.get<int>("@cluster_num");
    //cv::BOWKmajorityTrainer trainer(cluster_num, 100);
    cv::BOWKMeansTrainer trainer(cluster_num);
    trainer.add(training_desc);
    std::cout << "Cluster BOW features" << std::endl;
    cv::Mat vocab = trainer.cluster();

    auto output = parser.get<std::string>("@output");
    cv::FileStorage fs1(output, cv::FileStorage::WRITE);
    fs1 << "vocabulary" << vocab;
    fs1.release();
}