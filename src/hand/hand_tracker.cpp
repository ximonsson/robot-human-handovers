#include <hand/hand_tracker.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

HandTracker::HandTracker ()
{
	// nada for now
}

void HandTracker::find (cv::Mat &img)
{
	cv::Rect roi (img.cols / 2 - 100, img.rows / 2 - 100, 200, 200);
	cv::Mat gray;

	// convert RGB image to grayscale, then blur and convert to a binary image
	cvtColor (img, gray, CV_RGB2GRAY);
	cv::GaussianBlur (gray, gray, cv::Size (15, 15), 0, 0);
	cv::threshold (gray, gray, 90, 255, cv::THRESH_BINARY);

	// find the contours in the image
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours (gray, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

	// find the largest contours.
	// we are going to hope and assume that this is the hand.
	int largest_contour = 0; double max_area = 0.0f;
	for (int i = 0; i >= 0; i = hierarchy[i][0])
	{
		const std::vector<cv::Point>& c = contours[i];
		double area = fabs (cv::contourArea (cv::Mat (c)));
		if (area > max_area)
		{
			max_area = area;
			largest_contour = i;
		}
	}

	// draw the contours in the original image
	drawContours (img, contours, largest_contour, cv::Scalar (180, 180, 0), 1, cv::LINE_8, hierarchy);

	// convex hull
	std::vector<cv::Point> contour = contours[largest_contour];
	std::vector<int> hull;
	cv::convexHull (contour, hull);
	cv::Point pt0 = contour[hull.back ()];
	for (int i = 0; i < hull.size (); i++)
	{
		cv::Point pt = contour[hull[i]];
		cv::line (img, pt0, pt, cv::Scalar (0, 255, 0), 1, cv::LINE_AA);
		pt0 = pt;
	}

	// find convexity defects
	std::vector<cv::Vec4i> defects;
	cv::convexityDefects (contour, hull, defects);
	for (int i = 0; i < defects.size (); i++)
	{
		circle (img, contour[defects[i][2]], 4, cv::Scalar (0, 0, 255), 2);
	}
}
