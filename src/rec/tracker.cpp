#include <pcl/io/openni_grabber.h>
#include <rec/tracker.h>

ObjectTracker::ObjectTracker () : viewer ("PCL OpenNI Viewer")
{
	// nada for now
}

void ObjectTracker::run ()
{
	// create an interface towards the Kinect using OpenNI and start grabbing
	// RGBD frames that are sent to the ObjectTracker::track function as callback
	pcl::Grabber* interface = new pcl::OpenNIGrabber ();
	boost::function <void (const pcl::PointCloud <pcl::PointXYZRGBA>::ConstPtr&)> f =
		boost::bind (&ObjectTracker::track, this, _1);
	interface->registerCallback (f);
	interface->start ();
	while (!viewer.wasStopped ())
	{
		boost::this_thread::sleep (boost::posix_time::seconds (1));
	}
	interface->stop ();
}

// TODO implement tracking of object
void ObjectTracker::track (const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr &img)
{
	if (!viewer.wasStopped ())
		viewer.showCloud (img); // visual the input data
}
