#include <iostream>
#include <opencv2/opencv.hpp>
#include <apriltag.h>
#include <tag36h11.h>
#include <tag36artoolkit.h>
#include <getopt.h>

apriltag_detector_t *td = apriltag_detector_create ();
apriltag_family_t *tf = tag36h11_create ();

void wait ()
{
	bool done = false;
	while (!done)
	{
		char c = (char) cv::waitKey (0);
		switch (c)
		{
			case 'q':
				done = true;
				break;
		}
	}
}

std::string detection2string (apriltag_detection_t *d)
{
	// identifier
	std::stringstream ss;
	ss << d->id << ":";
	// four corners
	for (int i = 0; i < 4; i++)
	{
		ss << "(" << d->p[i][0] << "," << d->p[i][1] << ")";
	}
	// center
	ss << ":(" << d->c[0] << "," << d->c[1] << ")";
	return ss.str ();
}

cv::Mat load_object_image (int id)
{
	std::stringstream ss;
	ss << "data/objects/" << id << ".jpg";
	cv::Mat ref = cv::imread (ss.str ());
	cv::flip (ref, ref, 1);
	return ref;
}

cv::Mat load_object_mask (int id)
{
	std::stringstream ss;
	ss << "data/objects/" << id << "_mask.jpg";
	cv::Mat mask = cv::imread (ss.str ());
	cv::flip (mask, mask, 1);
	return mask;
}

void visualize_detections (cv::Mat frame, zarray_t* detections)
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

	cv::resize (frame, frame, cv::Size (1280, 720));
	// display the image in window and wait for a key press
	cv::imshow ("Tag Detections", frame);
}

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
		.stride = gray.cols,
		.buf    = gray.data
	};
	zarray_t *detections = apriltag_detector_detect (td, &im);
	return detections;
}

#define ROI_W 500
#define ROI_H 500

cv::Mat detect_handover (cv::Mat frame, zarray_t *&detections)
{
	int roix = frame.cols / 2 - ROI_W / 2;
	int roiy = frame.rows / 2 - ROI_H / 2;
	cv::Mat handover_region = frame (cv::Range (roiy, roiy + ROI_H), cv::Range (roix, roix + ROI_W));
	detections = detect (handover_region);
	return handover_region;
}

cv::Mat find_transformation (apriltag_detection_t *d, cv::Mat ref)
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
	dst.push_back (cv::Point2f (d->p[0][0], d->p[0][1]));
	dst.push_back (cv::Point2f (d->p[1][0], d->p[1][1]));
	dst.push_back (cv::Point2f (d->p[2][0], d->p[2][1]));
	dst.push_back (cv::Point2f (d->p[3][0], d->p[3][1]));

	zarray_destroy (detections);
	return cv::findHomography (src, dst);
}

// skin color minimum and maximum values for range detection
#define SKIN_MIN 80, 133, 77
#define SKIN_MAX 255, 173, 127

// kernel size for erode and dilate operations in skin detection
#define SKIN_KSIZE 4, 4

bool compare_contours (std::vector<cv::Point> c1, std::vector<cv::Point> c2)
{
	double a1 = cv::contourArea (c1);
	double a2 = cv::contourArea (c2);
	return (a1 > a2);
}

cv::Rect find_grasp_region (cv::Mat m)
{
	int an = 10;
	cv::Mat element = cv::getStructuringElement (cv::MORPH_RECT, cv::Size (4, 4));

	// convert the image to YCrCb and extract regions that are within the skin color range
	// erode and dilate the image back to remove any noise that comes from individual pixels
	// falling within the range but do not belong to any skin.
	cv::Mat skin;
	cv::cvtColor (m, skin, cv::COLOR_BGR2YCrCb);
	cv::inRange (skin, cv::Scalar (SKIN_MIN), cv::Scalar (SKIN_MAX), skin);
	cv::erode (skin, skin, cv::Mat::zeros (SKIN_KSIZE, CV_8UC1));
	cv::dilate (skin, skin, cv::Mat::zeros (SKIN_KSIZE, CV_8UC1));

	// find the contours around the skin area and return a bounding rectangle of it
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours (skin, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
	std::sort (contours.begin (), contours.end (), compare_contours);

	return cv::boundingRect (contours[0]);
}

char imfile[255] = {0};
int display_f, pose_f, flip_f;

void parse_command_line (int argc, char **argv)
{
	const struct option longopts[] =
	{
		{"image",      required_argument, 0,             'i'},
		{"visualize",  no_argument,       &display_f,    1},
		{"pose",       no_argument,       &pose_f,       1},
		{"flip",       no_argument,       &flip_f,       1},
		{0, 0, 0, 0},
	};

	int opt = 0, index;
	while (opt != -1)
	{
		opt = getopt_long (argc, argv, "i:", longopts, &index);
		switch (opt)
		{
			case 0:
				break;

			case 'i':
				strcpy (imfile, optarg);
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

	if (flip_f) // frames from the kinect camera need to be mirrored
		cv::flip (frame, frame, 1);

	// detect tag in the image
	zarray_t *detections;
	cv::Mat handover = detect_handover (frame, detections);
	if (zarray_size (detections))
	{
		//cv::imshow ("handover", handover);
		for (int i = 0; i < zarray_size (detections); i++)
		{
			apriltag_detection_t *d;
			zarray_get (detections, i, &d);
			std::cout << detection2string (d) << std::endl;

			// find transformation of object in original image to the one in handover
			// TODO not sure this can count as pose
			cv::Mat ref = load_object_image (d->id);
			cv::Mat trans = find_transformation (d, ref);

			// load the mask and transform it accordingly to extract the object in the scene
			// then invert the transformation of the extracted object into ground zero
			// and lastly extract the grasping region that is latter drawn onto the original image of the object.
			cv::Mat mask = load_object_mask (d->id);
			cv::warpPerspective (mask, mask, trans, handover.size ());
			cv::Mat object;
			handover.copyTo (object, mask);

			cv::Mat itrans;
			cv::invert (trans, itrans);
			cv::warpPerspective (object, object, itrans, object.size ());
			cv::Rect grasp = find_grasp_region (object);

			// draw the grasp region
			cv::rectangle (
					ref,
					cv::Point (grasp.x, grasp.y),
					cv::Point (grasp.x + grasp.width, grasp.y + grasp.height),
					cv::Scalar (0, 0, 255),
					1);
			cv::imshow ("opencv grasp", ref);
			wait ();
		}

		if (display_f) // if visualize option was passed
		{
			visualize_detections (frame, detections);
			wait ();
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
