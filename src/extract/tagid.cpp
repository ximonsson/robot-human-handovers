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

/**
 * Detect any AprilTags in the image.
 * Returns a pointer to zarray_t struct with the results. zarray_destroy needs to be called on the pointer later.
 */
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

/**
 * detection2str returns a string representation of a apriltag_detection_t pointed at by d.
 * The string is a one-line version of the detection in the following format:
 *     ID:(P0x,P0y)(P1x,P1y)(P2x,P2y)(P3x,P3y):(Cx,Cy)
 */
std::string detection2str (apriltag_detection_t *d)
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
	int ret = 1;

	// load scene image
	cv::Mat frame = cv::imread (imfile);
	if (flip_f) // frames from the kinect camera need to be mirrored
		cv::flip (frame, frame, 1);

	// detect tag in the image
	zarray_t *detections = detect (frame);
	// if we detect more than one object it is too confusing and we will abort
	if (zarray_size (detections) > 1)
	{
		std::cerr << "Too many objects in handover region! Aborting..." << std::endl;
		goto quit;
	}
	// else if there are none we abort also
	else if (zarray_size (detections) == 0)
	{
		std::cerr << "No objects in handover region" << std::endl;
		goto quit;
	}

	{ // from here on there should only be one object in the handover region to process
		apriltag_detection_t *d;
		zarray_get (detections, 0, &d);
		std::cout << detection2str (d) << std::endl;
		ret = 0;
	}

quit:
	// destroy and exit
	cv::destroyAllWindows ();
	zarray_destroy (detections);
	apriltag_detector_destroy (td);
	tag36h11_destroy (tf);
	return ret;
}

