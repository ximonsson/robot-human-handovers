#include <hand/hand_tracker.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

HandTracker::HandTracker ()
{
	// nada for now
}

void HandTracker::find (cv::Mat &img)
{
	// convert RGB image to grayscale
	cv::Mat gray;
	cvtColor (img, gray, CV_RGB2GRAY);

	cv::Mat blur;
	cv::GaussianBlur (gray, blur, cv::Size (5, 5), 0, 0);
	cv::Mat bin;
	cv::threshold (blur, bin, 70, 255, cv::THRESH_BINARY);

	std::vector<std::vector<cv::Point>> contours;
	cv::findContours (bin, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	drawContours (img, contours, -1, cv::Scalar (255, 255, 0), 2);
}
