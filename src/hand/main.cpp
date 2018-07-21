#include <pcl/point_cloud.h>
#include <pcl/visualization/image_viewer.h>
#include <pcl/io/openni_grabber.h>
#include <pcl/filters/passthrough.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <hand/hand_tracker.h>


typedef pcl::PointXYZRGBA Point;
typedef pcl::PointCloud<Point> Cloud;

/**
 * mat2cloud overwrites the RGB data in the points of cloud with the data from mat.
 * mat needs to be a cv::Mat of type CV_8UC4 with the color channels ordered by RGB
 * (note that this is not the standard in OpenCV).
 */
void mat2cloud (cv::Mat mat, Cloud::Ptr cloud)
{
	for (int j = 0; j < mat.rows; j++)
	{
		for (int i = 0; i < mat.cols; i++)
		{
			Point p = cloud->at (i, j);
			uint32_t rgb = mat.at<uint32_t> (j, i);
			p.rgb = *reinterpret_cast<float*> (&rgb);
			cloud->at (i, j) = p;
		}
	}
}

/**
 * rgb2mat converts a pcl::PointCloud to a cv::Mat of type CV_8UC4 using the RGB data within
 * the points.
 */
cv::Mat rgb2mat (const Cloud::ConstPtr cloud)
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

/**
 * depth2mat conversta a pcl::PointCloud to a cv::Mat using the depth data of the cloud.
 * The returned matrix will be of type CV_8UC4 but in grayscale, where closer to the camera will
 * be closer to white. This way it can be used to draw contours in it.
 */
cv::Mat depth2mat (const Cloud::ConstPtr cloud)
{
	cv::Mat mat (cloud->height, cloud->width, CV_8UC4);
	for (int j = 0; j < cloud->height; j++)
	{
		for (int i = 0; i < cloud->width; i++)
		{
			Point p = cloud->at (i, j);
			uint32_t gray = 255;
			if (!pcl_isfinite (p.z) || p.z >= 1.2)
				gray = 0;
			else if (p.z >= 1.0)
				gray = 85;
			else if (p.z >= 0.7)
				gray = 170;
			mat.at<uint32_t> (j, i) = gray << 16 | gray << 8 | gray;
		}
	}
	return mat;
}

#define cloud2mat depth2mat

void passthrough (const Cloud::ConstPtr &cloud, Cloud::Ptr cloud_filtered)
{
	Cloud::Ptr tmp (new Cloud ());
	pcl::PassThrough<Point> pass;

	pass.setInputCloud (cloud);
	pass.setFilterFieldName ("z");
	pass.setFilterLimits (0.0f, 1.0f);
	pass.filter (*tmp);

	pass.setInputCloud (tmp);
	pass.setFilterFieldName ("x");
	pass.setFilterLimits (-0.3f, 0.3f);
	pass.filter (*cloud_filtered);

	pass.setInputCloud (cloud_filtered);
	pass.setFilterFieldName ("y");
	pass.setFilterLimits (-0.3f, 0.3f);
	pass.filter (*tmp);

	*cloud_filtered = *tmp;
}

HandTracker tracker;

Cloud::Ptr track (const Cloud::ConstPtr &cloud)
{
	// convert point cloud to cv matrix
	Cloud::Ptr c (new Cloud (*cloud));
	cv::Mat img = cloud2mat (c);

	// find hand in image
	tracker.find (img);

	// change color info in point cloud for the new on in the matrix
	mat2cloud (img, c);

	return c;
}

pcl::visualization::ImageViewer viewer;

/**
 * cb is the callback function to grabbing new point clouds from the camera.
 */
void cb (const Cloud::ConstPtr &img)
{
	if (!viewer.wasStopped ())
	{
		Cloud::Ptr c = track (img);
		viewer.showRGBImage<Point> (c);
		viewer.spinOnce ();
	}
}

/**
 * run the application, starting a new interface towards the camera using cb as callback
 * and waiting for the viewer to exit.
 */
void run ()
{
	// create an interface towards the Kinect using OpenNI and start grabbing
	// RGBD frames that are sent to the ObjectTracker::track function as callback
	pcl::Grabber* interface = new pcl::OpenNIGrabber ();
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
