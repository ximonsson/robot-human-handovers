#include <iostream>
#include <opencv2/opencv.hpp>
#include <apriltag.h>
#include <tag36h11.h>
#include <tag36artoolkit.h>
#include <getopt.h>


apriltag_detector_t *td = apriltag_detector_create ();
apriltag_family_t *tf = tag36h11_create ();

void init ()
{
	tf->black_border = 1;
	apriltag_detector_add_family (td, tf);

	// not sure what these settings do...
	td->quad_decimate = 1.0;
	td->quad_sigma    = 0.0;
	td->nthreads      = 4;
	td->debug         = 0;
	td->refine_edges  = 1;
	td->refine_decode = 0;
	td->refine_pose   = 0;
}

zarray_t* detect (cv::Mat frame)
{
	cv::Mat gray;
	cv::cvtColor (frame, gray, cv::COLOR_BGR2GRAY);
	// Make an image_u8_t header for the Mat data
	image_u8_t im =
	{
		.width  = gray.cols,
		.height = gray.rows,
		.stride = gray.cols * static_cast<size_t>(gray.elemSize ()),
		.buf    = gray.data
	};
	zarray_t *detections = apriltag_detector_detect (td, &im);
	return detections;
}

void visualize (cv::Mat frame, zarray_t* detections)
{
	// Draw detection outlines
	for (int i = 0; i < zarray_size (detections); i++)
	{
		apriltag_detection_t *det;
		zarray_get (detections, i, &det);
		cv::line (
				frame,
				cv::Point (det->p[0][0], det->p[0][1]),
				cv::Point (det->p[1][0], det->p[1][1]),
				cv::Scalar (0, 0xff, 0), 2);
		cv::line (
				frame,
				cv::Point (det->p[0][0], det->p[0][1]),
				cv::Point (det->p[3][0], det->p[3][1]),
				cv::Scalar (0, 0, 0xff), 2);
		cv::line (
				frame,
				cv::Point (det->p[1][0], det->p[1][1]),
				cv::Point (det->p[2][0], det->p[2][1]),
				cv::Scalar (0xff, 0, 0), 2);
		cv::line (
				frame,
				cv::Point (det->p[2][0], det->p[2][1]),
				cv::Point (det->p[3][0], det->p[3][1]),
				cv::Scalar (0xff, 0, 0), 2);

		// write the ID of the tag in the middle of the rectangle
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

	//cv::resize (frame, frame, cv::Size (720, 1280));
	// display the image in window and wait for a key press
	cv::imshow ("Tag Detections", frame);
	while (true)
	{
		char c = (char) cv::waitKey (0);
		if (c == 'q')
			break;
	}
}

void grasp_region (cv::Mat ref, cv::Mat trans)
{

}

cv::Mat transformation (apriltag_detection_t *d, cv::Mat ref)
{
	// detect the apriltag again in the reference image and then compute the homography from it
	// to the detected tag in the scene
	zarray_t *detections = detect (ref);
	apriltag_detection_t *refd;
	zarray_get (detections, 0, &refd); // first detected tag - would be poor reference images otherwise

	std::vector<cv::Point2f> src;
	src.push_back (cv::Point2f (refd->p[0][0], refd->p[0][1]));
	src.push_back (cv::Point2f (refd->p[1][0], refd->p[1][1]));
	src.push_back (cv::Point2f (refd->p[2][0], refd->p[2][1]));
	src.push_back (cv::Point2f (refd->p[3][0], refd->p[3][1]));

	std::vector<cv::Point2f> dst;
	src.push_back (cv::Point2f (d->p[0][0], d->p[0][1]));
	src.push_back (cv::Point2f (d->p[1][0], d->p[1][1]));
	src.push_back (cv::Point2f (d->p[2][0], d->p[2][1]));
	src.push_back (cv::Point2f (d->p[3][0], d->p[3][1]));

	zarray_destroy (detections);
	return cv::findHomography (src, dst);
}

void print_detection (apriltag_detection_t *d)
{
	// identifier
	std::cout << d->id << ":";
	// four corners
	for (int i = 0; i < 4; i++)
	{
		std::cout << "(" << d->p[i][0] << "," << d->p[i][1] << ")";
	}
	// center
	std::cout << ":(" << d->c[0] << "," << d->c[1] << ")";
	std::cout << std::endl;
}

const char *imfile;
int display_f, homography_f;

void parse_command_line (int argc, char **argv)
{
	const struct option longopts[] =
	{
		{"image",      required_argument, 0,             'i'},
		{"visualize",  no_argument,       &display_f,    1},
		{"homography", no_argument,       &homography_f, 1},
		{0, 0, 0, 0},
	};

	int opt = 0, index;
	while (opt != -1)
	{
		opt = getopt_long (argc, argv, "i:vm", longopts, &index);
		switch (opt)
		{
			case 0:
				break;

			case 'i':
				imfile = std::string (optarg).c_str ();
				break;
		}
	}
}

int main (int argc, char **argv)
{
	// initialize
	parse_command_line (argc, argv);
	init ();
	cv::Mat frame = cv::imread (imfile); // load image
	//cv::flip (frame, frame, 1); // frames from the kinect camera are mirrored

	// detect tag in the image
	zarray_t *detections = detect (frame);
	if (zarray_size (detections))
	{
		for (int i = 0; i < zarray_size (detections); i++)
		{
			apriltag_detection_t *d;
			zarray_get (detections, i, &d);
			print_detection (d);

			if (homography_f) // calculate transformation of the object
			{
				// load the detected object's reference image
				std::stringstream ss;
				ss << "data/objects/" << d->id << ".jpg";
				cv::Mat ref = cv::imread (ss.str ());
				cv::Mat trans = transformation (d, ref);
				std::cout << trans << std::endl;
				cv::warpPerspective (ref, ref, trans, frame.size ());
				cv::imshow ("transformed", ref);
			}
		}

		if (display_f) // if visualize option was passed
		{
			visualize (frame, detections);
		}
	}
	else
		std::cout << "no tags detected" << std::endl;

	// destroy and exit
	cv::destroyAllWindows ();
	zarray_destroy (detections);
	apriltag_detector_destroy (td);
	tag36h11_destroy (tf);
	return 0;
}
