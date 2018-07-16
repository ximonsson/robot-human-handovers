#include <track/object_tracker.h>

int main (int argc, char **argv)
{
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr model (new pcl::PointCloud<pcl::PointXYZRGBA> ());
	if (pcl::io::loadPCDFile (argv[1], *model) < 0)
	{
		std::cerr << "error loading object cloud" << std::endl;
		return -1;
	}

	ObjectTracker tracker;
	tracker.set_object (model);
	tracker.run ();
	return 0;
}
