#include <pcl/visualization/cloud_viewer.h>

/**
 *
 */
class ObjectTracker
{
	public:
		ObjectTracker ();
		void run ();
		void set_object (); // TODO
		void track (const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr&);

	private:
		pcl::visualization::CloudViewer viewer;
};
