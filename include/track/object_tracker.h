#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/board.h>
#include <pcl/recognition/cg/hough_3d.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/correspondence.h>

const int TRACKING = 0x1;

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
		void visualize (const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr&, std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>);
		pcl::visualization::PCLVisualizer viewer;
		void keyboard_event (const pcl::visualization::KeyboardEvent&);

	private:
		// object features
		pcl::PointCloud<pcl::PointXYZRGBA>::Ptr object_model;
		pcl::PointCloud<pcl::PointXYZRGBA>::Ptr object_keypoints;
		pcl::PointCloud<pcl::Normal>::Ptr object_normals;
		pcl::PointCloud<pcl::SHOT352>::Ptr object_descriptors;
		pcl::PointCloud<pcl::ReferenceFrame>::Ptr obj_ref_frame;

		pcl::NormalEstimationOMP<pcl::PointXYZRGBA, pcl::Normal> normal_estimation;
		pcl::UniformSampling<pcl::PointXYZRGBA> uniform_sampling;
		pcl::SHOTEstimationOMP<pcl::PointXYZRGBA, pcl::Normal, pcl::SHOT352> descriptor_estimation;
		pcl::Hough3DGrouping<pcl::PointXYZRGBA, pcl::PointXYZRGBA, pcl::ReferenceFrame, pcl::ReferenceFrame> clusterer;
		pcl::BOARDLocalReferenceFrameEstimation<pcl::PointXYZRGBA, pcl::Normal, pcl::ReferenceFrame> ref_estimation;

		int flags;
};
