#include <track/object_tracker.h>

int load_pcd (const char* fp, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud)
{
	return pcl::io::loadPCDFile (fp, *cloud);
}

int main ()
{
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr model (new pcl::PointCloud<pcl::PointXYZRGBA> ());
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr scene (new pcl::PointCloud<pcl::PointXYZRGBA> ());

	if (load_pcd ("obj.pcd", model) < 0)
	{
		std::cerr << "error loading object cloud" << std::endl;
		return -1;
	}
	else if (load_pcd ("scene.pcd", scene) < 0)
	{
		std::cerr << "error loading scene cloud" << std::endl;
		return -1;
	}


	ObjectTracker tracker;
	tracker.set_object (model);
	while (!tracker.viewer.wasStopped ())
	{
		tracker.track (scene);
		tracker.viewer.spinOnce ();
	}

	//tracker.run ();
	return 0;
}
