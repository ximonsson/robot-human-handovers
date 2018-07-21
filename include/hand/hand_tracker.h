#include <cv.h>

class HandTracker
{
	public:
		HandTracker ();
		void find (cv::Mat&);

	private:
		cv::Point2d palm;
};
