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

    // WARNING: corner keypoints contain no orientation estimate
    // use with other descriptor as last resort only

    std::shared_ptr<cv::AgastFeatureDetector> detector = 
        cv::AgastFeatureDetector::create(10, true, cv::AgastFeatureDetector::AGAST_7_12s);
    std::vector<cv::KeyPoint> keypoints;
    detector->detect(src, keypoints);

    cv::Mat img_keypoints;
    cv::drawKeypoints(src, keypoints, img_keypoints);

    cv::imshow("AGAST Keypoints", img_keypoints);

    cv::waitKey();
    return 0;
}