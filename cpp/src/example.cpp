#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::Mat img(200, 200, CV_8UC1, cv::Scalar(0));
    cv::circle(img, {100,100}, 50, cv::Scalar(255), -1);
    cv::Mat edges;
    cv::Canny(img, edges, 50, 150);
    std::cout << "Edges nonzero: " << cv::countNonZero(edges) << std::endl;
    return 0;
}
