#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/filters/model_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>

template<typename Point>
class Clusterer
{
	public:
		typedef pcl::PointCloud<Point> PointCloud;

		Clusterer ()
		{
			c_tol = float (0.01);
			seg_dthresh = float (0.02);
		}

		void extract (const typename PointCloud::ConstPtr &scene, std::vector<pcl::PointIndices> &indices)
		{
			indices.clear ();

			pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
			pcl::ModelCoefficients::Ptr coefs (new pcl::ModelCoefficients);
			typename pcl::search::KdTree<Point>::Ptr tree (new pcl::search::KdTree<Point>);
			typename PointCloud::Ptr cloud (new PointCloud (*scene));
			typename PointCloud::Ptr cloud_plane (new PointCloud);
			typename PointCloud::Ptr cloud_f (new PointCloud);

			// set parameters for segmentation
			pcl::SACSegmentation<Point> seg;
			seg.setOptimizeCoefficients (true);
			seg.setModelType (pcl::SACMODEL_PLANE);
			seg.setMethodType (pcl::SAC_RANSAC);
			seg.setMaxIterations (300);
			seg.setDistanceThreshold (seg_dthresh);

			int i = 0, npoints = (int) cloud->points.size ();
			while (cloud->points.size () > 0.3f * npoints) // while 1/3 of the points remain
			{
				seg.setInputCloud (cloud);
				seg.segment (*inliers, *coefs);
				if (inliers->indices.size () == 0)
				{
					std::cout << "could not estimate a planar model for the given dataset" << std::endl;
					break;
				}

				// extract the planar inliers from the input cloud
				pcl::ExtractIndices<Point> extract;
				extract.setInputCloud (cloud);
				extract.setIndices (inliers);
				extract.setNegative (false);
				// get the points associated with the planar surface
				extract.filter (*cloud_plane);
				// remove the planar inliers, extract the rest
				extract.setNegative (true);
				extract.filter (*cloud_f);
				*cloud = *cloud_f;
			}

			// KdTree for the search method of the extraction
			tree->setInputCloud (cloud);

			pcl::EuclideanClusterExtraction<Point> ec;
			ec.setClusterTolerance (c_tol);
			ec.setMinClusterSize (500);
			ec.setMaxClusterSize (25000);
			ec.setSearchMethod (tree);
			ec.setInputCloud (cloud);
			ec.extract (indices);
		}

		void set_clustering_tolerance (float f) { c_tol = f; }
		void set_segmentation_threshold (float f) { seg_dthresh = f; }

	private:
		float c_tol;
		float seg_dthresh;
};
