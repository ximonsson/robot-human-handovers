#include <track/object_tracker.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/openni_grabber.h>
#include <pcl/io/openni2_grabber.h>
#include <pcl/filters/passthrough.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <pcl/console/parse.h>

typedef pcl::PointXYZRGBA Point;
typedef pcl::PointCloud<Point> PointCloud;

pcl::visualization::PCLVisualizer viewer ("PCL Object Tracking Viewer");
PointCloud::Ptr model (new PointCloud ());
ObjectTracker tracker;
boost::mutex mut;
PointCloud::ConstPtr scene;

void passthrough (const PointCloud::ConstPtr &cloud, PointCloud::Ptr cloud_filtered)
{
	PointCloud::Ptr tmp (new PointCloud ());
	pcl::PassThrough<pcl::PointXYZRGBA> pass;

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

void visualize (
		const PointCloud::ConstPtr &scene,
		std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> &rototranslations,
		std::vector<pcl::Correspondences> &clustered_correspondences)
{
	if (!viewer.wasStopped ())
	{
		viewer.removeAllPointClouds (); // clear the viewport
		viewer.addPointCloud (scene, "scene_cloud"); // add the scene
		// visualize the found objects
		// we only visualize the one with the most correspondences, we say that everything is else false positives
		for (size_t i = 0; i < rototranslations.size (); i++)
		{
			PointCloud::Ptr rotated_model (new PointCloud ());
			pcl::transformPointCloud (*model, *rotated_model, rototranslations[i]);
			std::stringstream ss_cloud;
			ss_cloud << "instance" << i;
			pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGBA> rotated_model_color_handler (
					rotated_model,
					255,
					0,
					0);
			viewer.addPointCloud (rotated_model, rotated_model_color_handler, ss_cloud.str ());
		}

		// create screenshots
		if (rototranslations.size () > 0)
		{
			std::cout << "taking a screenshot!" << std::endl;
			static int i = 0;
			std::stringstream ss;
			ss << "screen_" << i << ".png";
			viewer.saveScreenshot (ss.str ());

			pcl::PCDWriter writer;
			ss.str (std::string ());
			ss << "screen_" << i << ".pcd";
			writer.write<pcl::PointXYZRGBA> (ss.str (), *scene, false); //*

			i++;
		}

		viewer.spinOnce ();
	}
}

void keyboard_event (const pcl::visualization::KeyboardEvent &ev)
{
	if (ev.keyDown())
		return;

	switch (ev.getKeyCode ())
	{
		case 't':
			tracker.toggle_tracking ();
			break;
	}
}

void process (const PointCloud::ConstPtr &cloud)
{
	PointCloud::Ptr cloud_filtered (new PointCloud ());
	passthrough (cloud, cloud_filtered); // downsample

	// track the object
	std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> rototranslations;
	std::vector<pcl::Correspondences> clustered_correspondences;
	tracker.track (cloud_filtered, rototranslations, clustered_correspondences);
	visualize (cloud_filtered, rototranslations, clustered_correspondences);
}

void cb (const PointCloud::ConstPtr &cloud)
{
	boost::mutex::scoped_lock lock (mut);
	scene = cloud;
}

void run ()
{
	// create an interface towards the Kinect using OpenNI and start grabbing
	// RGBD frames that are sent to the ObjectTracker::track function as callback
	pcl::Grabber* interface = new pcl::io::OpenNI2Grabber ();
	boost::function<void (const PointCloud::ConstPtr&)> f = cb;
	boost::signals2::connection conn = interface->registerCallback (f);
	interface->start ();
	while (!viewer.wasStopped ())
	{
		//boost::this_thread::sleep (boost::posix_time::seconds (1));
		PointCloud::ConstPtr c;
		if (mut.try_lock ())
		{
			scene.swap (c);
			mut.unlock ();
		}
		if (c)
		{
			viewer.removeAllPointClouds ();
			//viewer.addPointCloud (c, "cloud");
			process (c);
		}
	}
	interface->stop ();
	conn.disconnect ();
}



int main (int argc, char **argv)
{
	if (pcl::io::loadPCDFile (argv[1], *model) < 0)
	{
		std::cerr << "error loading object cloud" << std::endl;
		return -1;
	}

	// register for keyboard events
	boost::function<void (const pcl::visualization::KeyboardEvent&)> f = keyboard_event;
	viewer.registerKeyboardCallback (f);

	float hough_threshold (5.0f);
	float hough_bin_size (0.01f);
	float descriptor_radius (0.01f);
	float scene_sampling_radius (0.005f);
	float object_sampling_radius (0.005f);
	float reference_radius (0.015f);

	pcl::console::parse_argument (argc, argv, "--hg_thresh", hough_threshold);
	pcl::console::parse_argument (argc, argv, "--hg_bin", hough_bin_size);
	pcl::console::parse_argument (argc, argv, "--drad", descriptor_radius);
	pcl::console::parse_argument (argc, argv, "--ssrad", scene_sampling_radius);
	pcl::console::parse_argument (argc, argv, "--osrad", object_sampling_radius);
	pcl::console::parse_argument (argc, argv, "--refrad", reference_radius);

	tracker.set_hough_threshold (hough_threshold);
	tracker.set_hough_bin_size (hough_bin_size);
	tracker.set_descriptor_radius (descriptor_radius);
	tracker.set_scene_sampling_radius (scene_sampling_radius);
	tracker.set_object_sampling_radius (object_sampling_radius);
	tracker.set_reference_radius (reference_radius);

	tracker.set_object (model);
	run ();

	return 0;
}
