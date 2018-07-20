#include <pcl/point_cloud.h>
#include <pcl/visualization/image_viewer.h>
#include <pcl/io/openni_grabber.h>
#include <cv.h>
#include <opencv2/imgproc.hpp>

typedef pcl::PointXYZRGBA Point;
typedef pcl::PointCloud<Point> Cloud;

pcl::visualization::ImageViewer viewer;

void mat2cloud (cv::Mat mat, Cloud::Ptr cloud)
{
	for (int j = 0; j < mat.rows; j++)
	{
		for (int i = 0; i < mat.cols; i++)
		{
			Point p = cloud->at (i, j);
			int32_t gray = mat.at<uint8_t> (j, i);
			uint32_t rgb = (uint32_t) gray << 16 | (uint32_t) gray << 8 | (uint32_t) gray;
			p.rgb = *reinterpret_cast<float*> (&rgb);
			cloud->at (i, j) = p;
		}
	}
}

cv::Mat cloud2mat (const Cloud::ConstPtr cloud)
{
	cv::Mat mat (cloud->height, cloud->width, CV_8UC4);
	for (int j = 0; j < cloud->height; j++)
	{
		for (int i = 0; i < cloud->width; i++)
		{
			Point p = cloud->at (i, j);
			mat.at<int32_t> (j, i) = *reinterpret_cast<int*> (&p.rgb);
		}
	}
	return mat;
}

Cloud::Ptr track (const Cloud::ConstPtr &cloud)
{
	Cloud::Ptr c (new Cloud (*cloud));
	cv::Mat img = cloud2mat (c);

	cv::Mat gray;
	cvtColor (img, gray, CV_RGB2GRAY);

	mat2cloud (gray, c);
	return c;
}

//void cb (const openni_wrapper::Image &img)
void cb (const Cloud::ConstPtr &img)
{
	if (!viewer.wasStopped ())
	{
		Cloud::Ptr c = track (img);
		viewer.showRGBImage<Point> (c);
		viewer.spinOnce ();
	}
}

void run ()
{
	// create an interface towards the Kinect using OpenNI and start grabbing
	// RGBD frames that are sent to the ObjectTracker::track function as callback
	pcl::Grabber* interface = new pcl::OpenNIGrabber ();
	//boost::function<void (const openni_wrapper::Image&)> f = cb;
	boost::function<void (const Cloud::ConstPtr&)> f = cb;
	interface->registerCallback (f);
	interface->start ();
	while (!viewer.wasStopped ())
	{
		boost::this_thread::sleep (boost::posix_time::seconds (1));
	}
	interface->stop ();
}

int main (int argc, char **argv)
{
	run ();
	return 0;
}
