#include <track/object_tracker.h>

int main ()
{
	ObjectTracker tracker;

	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr model (new pcl::PointCloud<pcl::PointXYZRGBA> ());
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr scene (new pcl::PointCloud<pcl::PointXYZRGBA> ());

	if (pcl::io::loadPCDFile ("obj.pcd", *model) < 0)
	{
		std::cerr << "error loading object cloud" << std::endl;
		return -1;
	}
	else if (pcl::io::loadPCDFile ("scene.pcd", *scene) < 0)
	{
		std::cerr << "error loading scene cloud" << std::endl;
		return -1;
	}


	tracker.set_object (model);
	tracker.track (scene);

	//tracker.run ();
	return 0;
}
