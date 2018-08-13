//#include <pcl/visualization/image_viewer.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

#include <libfreenect2/libfreenect2.hpp>

#include <opencv2/core.hpp>

typedef pcl::PointXYZRGBA Point;
typedef pcl::PointCloud<Point> PointCloud;

PointCloud::Ptr freenectFrame2cloud (libfreenect2::Frame registered, libfreenect2::Frame undistorted)
{
	libfreenect2::Freenect2Device::IrCameraParams ir_params = device->getIrCameraParams ();
	float cx = ir_params.cx, cy = ir_params.cy, fx = ir_params.fx, fy = ir_params.fy;
	PointCloud::Ptr cloud (new PointCloud (registered.width, registered.height));

	float *undistorted_data = (float *) undistorted.data;
	float *registered_data = (float *) registered.data;

	for (int xi = 0; xi < registered.width; xi++)
	{
		for (int yi = 0; yi < registered.height; yi++)
		{
			float xu = (xi + 0.5 - cx) / fx;
			float yu = (yi + 0.5 - cy) / fy;
			Point p;
			p.x = xu * undistorted_data[512 * yi + xi];
			p.y = yu * undistorted_data[512 * yi + xi];
			p.z = undistorted_data[512 * yi + xi];
			p.rgb = registered_data[512 * yi + xi];
			cloud->push_back (p);
		}
	}

	return cloud;
}

/**
 * mat2cloud overwrites the RGB data in the points of cloud with the data from mat.
 * mat needs to be a cv::Mat of type CV_8UC4 with the color channels ordered by RGB
 * (note that this is not the standard in OpenCV).
 */
void matRGB2cloud (cv::Mat mat, PointCloud::Ptr cloud)
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

void matDepth2cloud (cv::Mat mat, PointCloud::Ptr cloud)
{
	for (int j = 0; j < mat.rows; j++)
	{
		for (int i = 0; i < mat.cols; i++)
		{
			Point p = cloud->at (i, j);
			p.x = (float) i;
			p.y = (float) j;
			p.z = mat.at<float> (j, i);
			cloud->at (i, j) = p;
		}
	}
}
