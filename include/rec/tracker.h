#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/features/shot_omp.h>

/**
 *
 */
class ObjectTracker
{
	public:
		ObjectTracker ();
		void run ();
		void set_object (pcl::PointCloud<pcl::PointXYZRGBA>::Ptr);
		void track (const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr&);

	private:
		pcl::visualization::PCLVisualizer viewer;

		// object features
		pcl::PointCloud<pcl::PointXYZRGBA>::Ptr object_model;
		pcl::PointCloud<pcl::PointXYZRGBA>::Ptr object_keypoints;
		pcl::PointCloud<pcl::Normal>::Ptr object_normals;
		pcl::PointCloud<pcl::SHOT352>::Ptr object_descriptors;

		pcl::NormalEstimationOMP<pcl::PointXYZRGBA, pcl::Normal> normal_estimation;
		pcl::UniformSampling<pcl::PointXYZRGBA> uniform_sampling;
		pcl::SHOTEstimationOMP<pcl::PointXYZRGBA, pcl::Normal, pcl::SHOT352> descriptor_estimation;
};
