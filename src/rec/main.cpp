#include <pcl/point_cloud.h>
//#include <pcl/visualization/image_viewer.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

#include <iostream>
#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/registration.h>
#include <libfreenect2/packet_pipeline.h>
#include <libfreenect2/logger.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

typedef pcl::PointXYZRGBA Point;
typedef pcl::PointCloud<Point> PointCloud;

libfreenect2::Freenect2 freenect2;
libfreenect2::Freenect2Device *device = NULL;
libfreenect2::PacketPipeline *pipeline = NULL;
libfreenect2::SyncMultiFrameListener listener (libfreenect2::Frame::Color | libfreenect2::Frame::Ir | libfreenect2::Frame::Depth);
libfreenect2::Frame undistorted (512, 424, 4), registered (512, 424, 4);

cv::Mat image_rgb, image_depth, image_rgb_f, image_depth_f;

int thresh_min = 80, thresh_max = 250;

int open ()
{
	if (freenect2.enumerateDevices () == 0)
	{
		std::cerr << "no device connected!" << std::endl;
		return -1;
	}
	std::string serial = freenect2.getDefaultDeviceSerialNumber ();
	pipeline = new libfreenect2::OpenCLPacketPipeline ();
	device = freenect2.openDevice (serial, pipeline);
	device->setColorFrameListener (&listener);
	device->setIrAndDepthFrameListener (&listener);
	return 0;
}

int quit ()
{
	device->stop ();
	device->close ();
	return 0;
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

int store_pcd ()
{
	PointCloud::Ptr cloud (new PointCloud (registered.width, registered.height));
	//cloud->is_dense = true;
	matRGB2cloud (image_rgb_f, cloud);
	matDepth2cloud (image_depth_f, cloud);
	pcl::PCDWriter writer;
	writer.write<Point> ("cloud.pcd", *cloud, false);
	return 0;
}

int store ()
{
	FILE *fp = fopen ("depth.bin", "wb");
	fwrite (image_depth.data, image_depth.elemSize (), image_depth.rows * image_depth.cols, fp);
	fclose (fp);
	cv::imwrite ("frame.jpg", image_rgb);
	return store_pcd ();
}

int flags = 0;

#define REC 0x01

int handle_key ()
{
	char c = (char) cv::waitKey (50);
	switch (c)
	{
		case 'q':
			return 1;
		case 's':
			return store ();
		case 'r':
			flags ^= REC;
			break;
	}
	return 0;
}

std::string output_dir = ".";

void store_frame (libfreenect2::Frame *rgb, libfreenect2::Frame *depth)
{
	static int i = 0;
	std::stringstream ss;

	cv::Mat m;
	cv::Mat (rgb->height, rgb->width, CV_8UC4, rgb->data).copyTo (m);
	ss << output_dir << "/color/" << i << ".png";
	cv::imwrite (ss.str (), m);

	cv::Mat (depth->height, depth->width, CV_32FC1, (float*) depth->data).copyTo (m);
	ss.str ("");
	ss << output_dir << "/depth/" << i;
	FILE *fp = fopen (ss.str ().c_str (), "wb");
	fwrite (m.data, m.elemSize (), m.rows * m.cols, fp);
	fclose (fp);

	i++;
}

void run ()
{
	if (!device->start ())
	{
		std::cerr << "device will not start" << std::endl;
		return;
	}
	std::cout << "device serial: " << device->getSerialNumber () << std::endl;
	std::cout << "device firmware: " << device->getFirmwareVersion () << std::endl;

	libfreenect2::Registration *registration =
		new libfreenect2::Registration (device->getIrCameraParams (), device->getColorCameraParams ());

	libfreenect2::FrameMap frames;
	cv::Mat im_bin;

	while (true)
	{
		if (handle_key () != 0)
		{
			break;
		}
		else if (!listener.waitForNewFrame (frames, 10 * 1000))
		{
			std::cout << "timeout!" << std::endl;
			return;
		}

		libfreenect2::Frame *rgb = frames[libfreenect2::Frame::Color];
		libfreenect2::Frame *ir = frames[libfreenect2::Frame::Ir];
		libfreenect2::Frame *depth = frames[libfreenect2::Frame::Depth];

		registration->apply (rgb, depth, &undistorted, &registered);

		cv::Mat (registered.height, registered.width, CV_8UC4, registered.data).copyTo (image_rgb);
		cv::Mat (undistorted.height, undistorted.width, CV_32FC1, (float*) undistorted.data).copyTo (image_depth);

		if (flags & REC)
		{
			store_frame (rgb, depth);
		}

		//cv::Mat (depth->height, depth->width, CV_32FC1, depth->data).copyTo (image_depth);
		//cv::Mat (rgb->height, rgb->width, CV_8UC4, rgb->data).copyTo (image_rgb);

		// normalize depth image
		/*
		for (int i = 0; i < depth->height; i++)
			for (int j = 0; j < depth->width; j ++)
			{
				float v = image_depth.at<float> (i, j);
				v /= 4500;
				image_depth.at<float> (i, j) = v;
			}
		//*/

		cv::Rect roi (100, 100, 200, 200);
		// create binary mask
		image_rgb (roi).copyTo (im_bin);
		cv::cvtColor (im_bin, im_bin, cv::COLOR_RGB2GRAY);
		cv::GaussianBlur (im_bin, im_bin, cv::Size (5, 5), 0, 0);
		cv::threshold (im_bin, im_bin, thresh_min, thresh_max, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

		cv::Mat mask (image_rgb.rows, image_rgb.cols, im_bin.type (), cv::Scalar (0, 0, 0));
		im_bin.copyTo (mask (roi));

		image_rgb_f.setTo (cv::Scalar (0, 0, 0));
		image_rgb.copyTo (image_rgb_f, mask);

		image_depth_f.setTo (cv::Scalar (0, 0, 0));
		image_depth.copyTo (image_depth_f, mask);

		cv::imshow ("color", image_rgb_f);
		cv::imshow ("depth", image_depth_f);
		cv::imshow ("binary", mask);
		listener.release (frames);
	}

	cv::destroyAllWindows ();
	delete registration;
}

int load_frame (const char *dir, int i)
{
	std::stringstream ss;
	ss << dir << "/color/" << i << ".png";
	std::cout << ss.str () << std::endl;
	cv::Mat c = cv::imread (ss.str ());
	cv::imshow ("color", c);

	ss.str ("");
	ss << dir << "/depth/" << i;
	FILE *fp = fopen (ss.str ().c_str (), "rb");
	unsigned char depth_buf [512 * 424 * 4];
	fread (depth_buf, 4, 512 * 424, fp);
	cv::Mat depth (424, 512, CV_32FC1, depth_buf);
	cv::imshow ("depth", depth);

	cv::waitKey (0);
	return 0;
}

int main (int argc, char** argv)
{
	load_frame (argv[1], 0);
	return 0;

	if (open () != 0)
	{
		return -1;
	}

	if (argc > 1)
	{
		output_dir = std::string (argv[1]);
		std::cout << "storing recorded data to: " << output_dir << std::endl;
	}

	run ();
	quit ();
	return 0;
}
