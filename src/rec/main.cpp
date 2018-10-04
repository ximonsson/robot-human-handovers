#include <iostream>
#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/registration.h>
#include <libfreenect2/packet_pipeline.h>
#include <libfreenect2/logger.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "apriltag.h"
#include "tag36h11.h"

libfreenect2::Freenect2 freenect2;
libfreenect2::Freenect2Device *device = NULL;
libfreenect2::PacketPipeline *pipeline = NULL;
libfreenect2::SyncMultiFrameListener listener (
		libfreenect2::Frame::Color | libfreenect2::Frame::Ir | libfreenect2::Frame::Depth);
libfreenect2::Frame undistorted (512, 424, 4), registered (512, 424, 4);

cv::Mat image_rgb, image_depth, image_rgb_f, image_depth_f;

apriltag_detector_t *td;
apriltag_family_t *tf;

int thresh_min = 30, thresh_max = 200;

int open ()
{
	// open device
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

	// init apriltag stuff
	tf = tag36h11_create ();
	tf->black_border = 1; // border size

	// not sure what these settings do...
	td = apriltag_detector_create ();
	apriltag_detector_add_family (td, tf);
	td->quad_decimate = 1.0; // decimate input image
	td->quad_sigma    = 0.0; // apply low pass blur to input
	td->nthreads      = 4; // CPU threads
	td->debug         = false; // debug
	td->refine_edges  = true; // more time trying to align edges
	td->refine_decode = false; // more time trying to decode tags
	td->refine_pose   = false; // more time trying to precisely localize tags

	return 0;
}

int quit ()
{
	// apriltags
	apriltag_detector_destroy (td);
	tag36h11_destroy (tf);

	// stop device
	device->stop ();
	device->close ();

	return 0;
}

int flags = 0;

#define REC      0x01
#define DETECT   0x02
#define SNAPSHOT 0x04

int handle_key ()
{
	char c = (char) cv::waitKey (10);
	switch (c)
	{
		case 'q':
			return 1;
		case 's':
			flags |= SNAPSHOT;
			break;
		case 'r':
			flags ^= REC;
			break;
		case 'd':
			flags ^= DETECT;
			break;
	}
	return 0;
}

std::string output_dir = ".";

void store_frame (std::string dir, libfreenect2::Frame *rgb, libfreenect2::Frame *depth, libfreenect2::Frame *registered)
{
	static int i = 0;
	std::stringstream ss;
	cv::Mat m;

	// store full RGB frame
	cv::Mat (rgb->height, rgb->width, CV_8UC4, rgb->data).copyTo (m);
	ss << dir << "/rgb/" << i << ".jpg";
	cv::imwrite (ss.str (), m);

	// store depth image
	cv::Mat (depth->height, depth->width, CV_32FC1, (float*) depth->data).copyTo (m);
	ss.str ("");
	ss << dir << "/depth/" << i;
	FILE *fp = fopen (ss.str ().c_str (), "wb");
	fwrite (m.data, m.elemSize (), m.rows * m.cols, fp);
	fclose (fp);

	// store fitted registered frame
	cv::Mat (registered->height, registered->width, CV_8UC4, registered->data).copyTo (m);
	ss.str ("");
	ss << dir << "/registered/" << i << ".jpg";
	cv::imwrite (ss.str (), m);

	i++;
}

int detect (cv::Mat &frame)
{
	cv::Mat gray;
	cv::cvtColor (frame, gray, cv::COLOR_BGR2GRAY);

	// Make an image_u8_t header for the Mat data
	image_u8_t im = {
		.width  = gray.cols,
		.height = gray.rows,
		.stride = gray.cols,
		.buf    = gray.data
	};

	zarray_t *detections = apriltag_detector_detect (td, &im);
	std::cout << zarray_size (detections) << " tags detected" << std::endl;
	if (zarray_size (detections) == 0)
		return 1;

	// Draw detection outlines
	for (int i = 0; i < zarray_size (detections); i++)
	{
		apriltag_detection_t *det;
		zarray_get (detections, i, &det);
		cv::line(frame, cv::Point (det->p[0][0], det->p[0][1]),
				cv::Point (det->p[1][0], det->p[1][1]),
				cv::Scalar (0, 0xff, 0), 2);
		cv::line (frame, cv::Point (det->p[0][0], det->p[0][1]),
				cv::Point (det->p[3][0], det->p[3][1]),
				cv::Scalar (0, 0, 0xff), 2);
		cv::line (frame, cv::Point (det->p[1][0], det->p[1][1]),
				cv::Point (det->p[2][0], det->p[2][1]),
				cv::Scalar (0xff, 0, 0), 2);
		cv::line (frame, cv::Point (det->p[2][0], det->p[2][1]),
				cv::Point (det->p[3][0], det->p[3][1]),
				cv::Scalar (0xff, 0, 0), 2);

		std::stringstream ss;
		ss << det->id;
		std::string text = ss.str ();
		int fontface = cv::FONT_HERSHEY_SCRIPT_SIMPLEX;
		double fontscale = 1.0;
		int baseline;
		cv::Size textsize = cv::getTextSize (text, fontface, fontscale, 2, &baseline);
		cv::putText (
				frame,
				text,
				cv::Point (
					det->c[0]-textsize.width/2,
					det->c[1]+textsize.height/2),
				fontface,
				fontscale,
				cv::Scalar (0xff, 0x99, 0), 2);
	}
	zarray_destroy (detections);
	return 0;
}

#define ROI_W 300
#define ROI_H 300

void visualize (libfreenect2::Frame *rgb, libfreenect2::Frame *depth, libfreenect2::Frame *registered)
{
	cv::Mat image_rgb;
	cv::Mat (rgb->height, rgb->width, CV_8UC4, rgb->data).copyTo (image_rgb);
	cv::resize (image_rgb, image_rgb, cv::Size (1280, 720));
	cv::flip (image_rgb, image_rgb, 1);

	if (flags & DETECT)
	{
		detect (image_rgb);
	}

	// draw a rectangle around ROI to guide
	cv::rectangle (
			image_rgb,
			cv::Point (image_rgb.cols / 2 - ROI_W / 2, image_rgb.rows / 2 - ROI_H / 2),
			cv::Point (image_rgb.cols / 2 + ROI_W / 2, image_rgb.rows / 2 + ROI_H / 2),
			cv::Scalar (0, 0, 255),
			2);

	cv::Mat reg;
	cv::Mat (registered->height, registered->width, CV_8UC4, registered->data).copyTo (reg);

	// depth image
	// draw a crosshair in the center to help guide
	cv::Mat d;
	cv::Mat (depth->height, depth->width, CV_32FC1, depth->data).copyTo(d);
	cv::normalize (d, d, 1.0, 0.0, cv::NORM_INF);
	///*
	cv::cvtColor (d, d, cv::COLOR_GRAY2BGRA);
	cv::line (
			d,
			cv::Point (depth->width / 2, depth->height / 2 - 8),
			cv::Point (depth->width / 2, depth->height / 2 + 8),
			cv::Scalar (0, 0, 255),
			2);
	cv::line (
			d,
			cv::Point (depth->width / 2 - 8, depth->height / 2),
			cv::Point (depth->width / 2 + 8, depth->height / 2),
			cv::Scalar (0, 0, 255),
			2);
	//*/

	//cv::imshow ("rgb", image_rgb);
	//cv::imshow ("registered", reg);
	cv::imshow ("opencv depth", d);
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
	cv::namedWindow ("rgb");

	while (true)
	{
		// listen for key events, if exit key was hit exit the main loop
		// else wait for a new frame from the camera
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
		visualize (rgb, &undistorted, &registered);

		// take a snapshot
		if (flags & SNAPSHOT)
		{
			store_frame (output_dir, rgb, &undistorted, &registered);
			flags &= ~SNAPSHOT;
		}

		// recording
		if (flags & REC)
		{
			store_frame (output_dir, rgb, &undistorted, &registered);
		}

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

	libfreenect2::Frame undistorted (512, 424, 4), registered (512, 424, 4);
	libfreenect2::Frame rgb (c.cols, c.rows, 4), depthf (512, 424, 4);
	rgb.data = c.data;
	depthf.data = depth.data;

	cv::waitKey (0);
	return 0;
}

int main (int argc, char** argv)
{
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
