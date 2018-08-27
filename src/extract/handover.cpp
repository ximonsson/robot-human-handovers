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

/**
 * homography2str returns a string representation of the obtained homography matrix h.
 * The output is one line with each value separated by a ',', as well as ending with one.
 */
std::string homography2str (cv::Mat h)
{
	std::stringstream ss;
	for (int r = 0; r < h.rows; r++)
	{
		double *p = h.ptr<double> (r);
		for (int c = 0; c < h.cols; c++)
		{
			ss << *p << ",";
			p++;
		}
	}
	return ss.str ();
}

typedef cv::RotatedRect Grasp;

/**
 * grasp2str returns a string representation of the grasp region defined by the Grasp g.
 * The output is a one-line string in the format:
 *     Cx,Cy,W,H,Angle
 */
std::string grasp2str (Grasp g)
{
	std::stringstream ss;
	ss << g.center.x << "," << g.center.y << "," << g.size.width << "," << g.size.height << "," << g.angle;
	return ss.str ();
}

/**
 * Draw the grasping region denoted by g in dst.
 */
void draw_grasp (cv::Mat &dst, Grasp g)
{
	/*
	cv::rectangle (
			dst,
			cv::Point (g.x, g.y),
			cv::Point (g.x + g.width, g.y + g.height),
			cv::Scalar (0, 0, 255),
			1);
		*/
	cv::Point2f points[4];
	g.points (points);
	for (int i = 0; i < 4; i++)
		cv::line (dst, points[i], points[(i + 1) % 4], cv::Scalar (0, 0, 255), 1);
}

/**
 * skin color minimum and maximum values for range detection
 */
#define SKIN_MIN 80, 133, 77
#define SKIN_MAX 255, 173, 127

/**
 * kernel size for erode and dilate operations in skin detection
 */
#define SKIN_KSIZE 4, 4

/**
 * compare_contours returns if c1 > c2 based on cv::contourArea.
 */
bool compare_contours (std::vector<cv::Point> c1, std::vector<cv::Point> c2)
{
	double a1 = cv::contourArea (c1);
	double a2 = cv::contourArea (c2);
	return (a1 > a2);
}

/**
 * Finds the grasping region in the image.
 * The function searches for regions that match skin color and returns a bounding
 * rectangle around the largest part.
 */
Grasp find_grasp_region (cv::Mat &m)
{
	// convert the image to YCrCb and extract regions that are within the skin color range
	// erode and dilate the image back to remove any noise that comes from individual pixels
	// falling within the range but do not belong to any skin.
	cv::Mat skin;
	cv::cvtColor (m, skin, cv::COLOR_BGR2YCrCb);
	cv::inRange (skin, cv::Scalar (SKIN_MIN), cv::Scalar (SKIN_MAX), skin);
	cv::erode (skin, skin, cv::Mat::ones (SKIN_KSIZE, CV_8UC1));
	cv::dilate (skin, skin, cv::Mat::ones (SKIN_KSIZE, CV_8UC1));

	// find the contours around the skin area and return a bounding rectangle of it
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours (skin, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
	std::sort (contours.begin (), contours.end (), compare_contours);

	cv::drawContours (m, contours, 0, cv::Scalar (0, 255, 0), 1);
	return cv::minAreaRect (contours[0]);
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
 * Parameters of the Region Of Interest (ROI)
 * X,Y cordinates denote the upper left's corner's offset from the center of the image.
 */
#define ROI_W 350
#define ROI_H 400
#define ROI_X (-ROI_W >> 1)
#define ROI_Y -100

/**
 * detect_handover takes a frame with a scene and tries to detect the handover in it.
 * This function extracts a region of interest within the scene and tries to detect any objects within it. If
 * it does we count this as a handover moment.
 * In case no handover was detected the zarray_t pointer will point to an object with an empty size.
 *
 * Returns the ROI in the frame.
 */
zarray_t* detect_handover (cv::Mat frame)
{
	int roix = frame.cols / 2 + ROI_X;
	int roiy = frame.rows / 2 + ROI_Y;
	cv::Rect roi (roix, roiy, ROI_W, ROI_H);

	// mask out the handover region we are interested in
	cv::Mat mask = cv::Mat::zeros (frame.rows, frame.cols, CV_8U);
	cv::rectangle (mask, roi, 255, cv::FILLED);
	cv::Mat detection_area;
	frame.copyTo (detection_area, mask);

	return detect (detection_area);
}

/**
 * draw_detections draws the detection found in image frame.
 * A square is drawn around the apriltag with the ID in the middle.
 */
void draw_detections (cv::Mat frame, zarray_t* detections)
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
}

#define OBJ_W 500
#define OBJ_H 500

/**
 * Directory containing object data images.
 */
#define DATA_OBJECT_DIR "data/objects/"

/**
 * load_object_image returns the reference image for the object.
 * The image is resized to the size of the region of interest for convenience.
 */
cv::Mat load_object_image (int id)
{
	std::stringstream ss;
	ss << DATA_OBJECT_DIR << id << ".jpg";
	cv::Mat ref = cv::imread (ss.str ());
	assert (ref.data != NULL); // make sure we have a reference image of the object stored to disk to work with
	//cv::resize (ref, ref, cv::Size (ROI_W, ROI_H), 0, 0, cv::INTER_NEAREST);
	cv::flip (ref, ref, 1);
	return ref;
}

/**
 * load_object_mask returns the mask for the object with ID id.
 * The mask is resized to the size of the ROI for convenience.
 */
cv::Mat load_object_mask (int id)
{
	std::stringstream ss;
	ss << DATA_OBJECT_DIR << id << "_mask.jpg";
	cv::Mat mask = cv::imread (ss.str (), CV_LOAD_IMAGE_GRAYSCALE);
	cv::flip (mask, mask, 1);
	//cv::resize (mask, mask, cv::Size (ROI_W, ROI_H), 0, 0, cv::INTER_NEAREST);
	cv::threshold (mask, mask, 155, 255, cv::THRESH_BINARY); // must have done the masks wrong
	return mask;
}

/**
 * find_transformation finds the transformation that has been applied to find the detection in a frame.
 * This is done by loading the reference image of the AprilTag ID, and then computing the homography between
 * the tag in it to the detection that is supplied.
 *
 * Returns the Homography matrix between the reference AprilTag and the one in the detection.
 */
cv::Mat find_transformation (apriltag_detection_t *d)
{
	// load the reference image and compute the homography from it
	cv::Mat ref = load_object_image (d->id);

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

/**
 * extract_object will return a cv::Mat of the object in the frame masked out.
 * The mask of the object with the supplied ID is loaded from disk and then warped before being applied
 * to the supplied frame to extract only the object in the scene.
 * The returned matrix has the same size as the mask (500x500)
 */
cv::Mat extract_object (cv::Mat f, apriltag_detection_t *d, cv::Mat &H)
{
	// extract a rectangle around the detected object of OBJ_W x OBJ_H
	cv::Mat obj = f (cv::Range (d->c[1] - OBJ_H / 2, d->c[1] + OBJ_H / 2), cv::Range (d->c[0] - OBJ_W / 2, d->c[0] + OBJ_W / 2));

	// detect again within this new image the tag from which will calculate the transformation
	// NOTE maybe we can speed this up by just subtracting from the points in the supplied detection
	zarray_t* detections = detect (obj);
	apriltag_detection_t *detection;
	zarray_get (detections, 0, &detection);
	H = find_transformation (detection);
	zarray_destroy (detections);

	// load the mask for the object,
	// transform the mask to fit the aspect ratio of the object
	// mask out the object
	cv::Mat mask = load_object_mask (d->id);
	cv::warpPerspective (mask, mask, H, mask.size ());
	cv::Mat object; obj.copyTo (object, mask);
	return object;
}

char imfile[255] = {0};
int display_f, pose_f, flip_f;

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
	// if we detect more than one object it is too confusing and we will abort
	// else if there are none we abort also
	zarray_t *detections = detect_handover (frame);
	if (zarray_size (detections) > 1)
	{
		std::cerr << "Too many objects in handover region! Aborting..." << std::endl;
		goto quit;
	}
	else if (zarray_size (detections) == 0)
	{
		std::cerr << "No objects in handover region" << std::endl;
		goto quit;
	}

	{ // from here on there should only be one object in the handover region to process
		apriltag_detection_t *d;
		zarray_get (detections, 0, &d);
		std::cout << detection2str (d) << std::endl;

		// get the object in the scene masked out
		cv::Mat H;
		cv::Mat obj = extract_object (frame, d, H);
		std::cout << homography2str (H) << std::endl;

		// transform to same orientation in original image and extract grasp
		cv::Mat iH; cv::invert (H, iH);
		cv::warpPerspective (obj, obj, iH, obj.size ());
		Grasp grasp = find_grasp_region (obj);
		std::cout << grasp2str (grasp) << std::endl;

		if (display_f)
		{
			draw_grasp (obj, grasp);
			cv::imshow ("opencv handover", obj);
			wait ();
		}
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
