#include <stdlib.h>

#include <opencv2/opencv.hpp>
#include <iostream>

#include <filesystem>
#include "BOWKmajorityTrainer.h"

int main(int argc, char** argv) {
    cv::CommandLineParser parser(argc, argv, 
        "{@collection | something.yml | descriptor collection file}"
        "{@pos | something.yml | positive descriptor collection file}"
        "{@neg | something.yml | negative collection file}"
        "{@cluster_num | 1000 | cluster number}"
        "{@output | something_1000.yml | vocab cluster output}");
    auto collection = parser.get<std::string>("@collection"); 
    cv::FileStorage fs(collection, cv::FileStorage::READ);
    cv::Mat training_desc;
    fs["training_descriptors"] >> training_desc;
    fs.release();

    auto cluster_num = parser.get<int>("@cluster_num");
    cv::BOWKmajorityTrainer trainer(cluster_num);
    trainer.add(training_desc);
    std::cout << "Cluster BOW features" << std::endl;
    cv::Mat vocab = trainer.cluster();

    auto output = parser.get<std::string>("@output");
    cv::FileStorage fs1(output, cv::FileStorage::WRITE);
    fs1 << "vocabulary" << vocab;
    fs1.release();

    cv::Mat dist_matrix;

    auto pos = parser.get<std::string>("@pos"); 
    cv::FileStorage fspos(pos, cv::FileStorage::READ);
    cv::Mat pos_desc;
    fspos["training_descriptors"] >> pos_desc;
    fspos.release();

    auto neg = parser.get<std::string>("@neg"); 
    cv::FileStorage fsneg(neg, cv::FileStorage::READ);
    cv::Mat neg_desc;
    fsneg["training_descriptors"] >> neg_desc;
    fsneg.release();

    for (int i=0; i<vocab.rows; i++) {
        cv::Mat current_desc = vocab.row(i);
        double pos_dist = 0, neg_dist = 256;
        double pos_sum = 0, neg_sum = 0;
        for (int j=0; j<pos_desc.rows; j++) {
            cv::Mat current_pos = pos_desc.row(j);
            double dist = cv::norm(current_desc, current_pos, cv::NORM_HAMMING);
            if (dist > pos_dist) pos_dist = dist;
            pos_sum += dist;
        }
        for (int j=0; j<neg_desc.rows; j++) {
            cv::Mat current_neg = neg_desc.row(j);
            double dist = cv::norm(current_desc, current_neg, cv::NORM_HAMMING);
            if (dist < neg_dist) neg_dist = dist;
            neg_sum += dist;
        }
        std::cout 
            << pos_dist << " " 
            << (pos_sum/pos_desc.rows) << " "
            << neg_dist << " "
            << (neg_sum/neg_desc.rows) << std::endl;
    }
}