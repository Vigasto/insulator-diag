#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

int main( int argc, char* argv[]) {
    //Simple script to check sift keypoints
    cv::CommandLineParser parser(argc, argv, 
        "{@input | image.png | input image}" );
    cv::Mat src = cv::imread(cv::samples::findFile(
        parser.get<std::string>( "@input")));
    if (src.empty()) {
        std::cout << "Could not open or find the image" << std::endl;
        std::cout << "Usage: " << argv[0] << "<Input image>" << std::endl;
        return -1;
    }

    std::shared_ptr<cv::SIFT> detector = 
        cv::SIFT::create(0,3,0.04,10.0);
    std::vector<cv::KeyPoint> keypoints;
    detector->detect(src, keypoints);

    cv::Mat img_keypoints;
    cv::drawKeypoints(src, keypoints, img_keypoints);

    cv::imshow("SIFT Keypoints", img_keypoints);

    cv::waitKey();
    return 0;
}