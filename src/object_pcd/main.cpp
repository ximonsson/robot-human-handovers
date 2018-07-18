#include <pcl/filters/passthrough.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/openni_grabber.h>
#include <iostream>
#include <pcl/point_types.h>
#include <pcl/filters/model_outlier_removal.h>
#include <pcl/console/parse.h>

typedef pcl::PointXYZRGBA Point;

float seg_dthresh (0.01);
float c_tol (0.02);


double colors[5][3] =
{
	{255, 0, 0},
	{0, 255, 0},
	{0, 0, 255},
	{255, 255, 0},
	{0, 255, 255},
};

class ObjectExtractor
{
	public:
		ObjectExtractor () : viewer ("Object Extraction Viewer")
		{
			cloud_filtered = pcl::PointCloud<Point>::Ptr (new pcl::PointCloud<Point>);
			cloud_f = pcl::PointCloud<Point>::Ptr (new pcl::PointCloud<Point>);
			inliers = pcl::PointIndices::Ptr (new pcl::PointIndices);
			coefs = pcl::ModelCoefficients::Ptr (new pcl::ModelCoefficients);
			cloud_plane = pcl::PointCloud<Point>::Ptr (new pcl::PointCloud<Point>);
			coefs = pcl::ModelCoefficients::Ptr (new pcl::ModelCoefficients);
			tree = pcl::search::KdTree<Point>::Ptr (new pcl::search::KdTree<Point>);

			// set parameters for segmentation
			seg.setOptimizeCoefficients (true);
			seg.setModelType (pcl::SACMODEL_PLANE);
			seg.setMethodType (pcl::SAC_RANSAC);
			seg.setMaxIterations (100);
			seg.setDistanceThreshold (seg_dthresh);
		}

		void visualize_cluster (const pcl::PointCloud<Point>::ConstPtr &cluster, int j)
		{
			std::stringstream ss;
			ss << "cluster_" << j;
			pcl::visualization::PointCloudColorHandlerCustom<Point> color_handler (cluster, colors[j][0], colors[j][1], colors[j][2]);
			viewer.addPointCloud (cluster, color_handler, ss.str ());
		}

		void visualize (const pcl::PointCloud<Point>::ConstPtr &scene)
		{
			if (!viewer.wasStopped ())
			{
				viewer.removeAllPointClouds ();
				viewer.addPointCloud (scene, "scene_cloud");

				// visualize all the clusters
				int j = 0;
				for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); it++)
				{
					pcl::PointCloud<Point>::Ptr cloud_cluster (new pcl::PointCloud<Point>);
					for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); pit++)
					{
						cloud_cluster->points.push_back (cloud_filtered->points[*pit]);
					}
					cloud_cluster->width = cloud_cluster->points.size ();
					cloud_cluster->height = 1;
					cloud_cluster->is_dense = true;

					visualize_cluster (cloud_cluster, j);

					j++;
					if (j == 5) break;
				}
				viewer.spinOnce();
			}
		}

		void extract (const pcl::PointCloud<Point>::ConstPtr &scene)
		{
			pcl::PointCloud<Point>::Ptr filtered_scene (new pcl::PointCloud<Point>);
			pass.setInputCloud (scene);
			pass.setFilterFieldName ("z");
			pass.setFilterLimits (0.0f, 1.0f);
			pass.filter (*cloud_filtered);

			pass.setInputCloud (cloud_filtered);
			pass.setFilterFieldName ("x");
			pass.setFilterLimits (-0.2f, 0.2f);
			pass.filter (*filtered_scene);

			pass.setInputCloud (filtered_scene);
			pass.setFilterFieldName ("y");
			pass.setFilterLimits (-0.3f, 0.3f);
			pass.filter (*cloud_filtered);

			*filtered_scene = *cloud_filtered;

			//vg.setInputCloud (filtered_scene);
			//vg.setLeafSize (0.01f, 0.01f, 0.01f);
			//vg.filter (*cloud_filtered);

			int i = 0, npoints = (int) cloud_filtered->points.size ();
			while (cloud_filtered->points.size () > 0.3f * npoints) // while 1/3 of the points remain
			{
				seg.setInputCloud (cloud_filtered);
				seg.segment (*inliers, *coefs);
				if (inliers->indices.size () == 0)
				{
					std::cout << "could not estimate a planar model for the given dataset" << std::endl;
					break;
				}

				// extract the planar inliers from the input cloud
				pcl::ExtractIndices<Point> extract;
				extract.setInputCloud (cloud_filtered);
				extract.setIndices (inliers);
				extract.setNegative (false);
				// get the points associated with the planar surface
				extract.filter (*cloud_plane);
				// remove the planar inliers, extract the rest
				extract.setNegative (true);
				extract.filter (*cloud_f);
				*cloud_filtered = *cloud_f;
			}

			// KdTree for the search method of the extraction
			tree->setInputCloud (cloud_filtered);

			cluster_indices.clear ();
			pcl::EuclideanClusterExtraction<Point> ec;
			ec.setClusterTolerance (c_tol); // 2cm
			ec.setMinClusterSize (100);
			ec.setMaxClusterSize (25000);
			ec.setSearchMethod (tree);
			ec.setInputCloud (cloud_filtered);
			ec.extract (cluster_indices);

			visualize (filtered_scene);
		}

		void snapshot ()
		{
			pcl::PCDWriter writer;
			int j = 0;
			for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
			{
				pcl::PointCloud<Point>::Ptr cloud_cluster (new pcl::PointCloud<Point>);
				for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
					cloud_cluster->points.push_back (cloud_filtered->points[*pit]); //*
				cloud_cluster->width = cloud_cluster->points.size ();
				cloud_cluster->height = 1;
				cloud_cluster->is_dense = true;

				std::stringstream ss;
				ss << "cloud_cluster_" << j << ".pcd";
				writer.write<Point> (ss.str (), *cloud_cluster, false); //*
				j++;
			}
		}

		void handle_key (const pcl::visualization::KeyboardEvent &event)
		{
			switch (event.getKeyCode ())
			{
				case 's':
					snapshot ();
					break;
			}
		}

		void run ()
		{
			// register for keyboard events
			boost::function<void (const pcl::visualization::KeyboardEvent&)> key_cb =
				boost::bind (&ObjectExtractor::handle_key, this, _1);
			viewer.registerKeyboardCallback (key_cb);

			// create an interface towards the Kinect using OpenNI and start grabbing
			// RGBD frames that are sent to the ObjectTracker::track function as callback
			pcl::Grabber* interface = new pcl::OpenNIGrabber ();
			boost::function<void (const pcl::PointCloud<Point>::ConstPtr&)> f =
				boost::bind (&ObjectExtractor::extract, this, _1);
			interface->registerCallback (f);
			interface->start ();
			while (!viewer.wasStopped ())
			{
				boost::this_thread::sleep (boost::posix_time::seconds (1));
			}
			interface->stop ();
		}

		pcl::visualization::PCLVisualizer viewer;
		pcl::VoxelGrid<Point> vg; // filtering object
		pcl::PointCloud<Point>::Ptr cloud_filtered;
		pcl::PointCloud<Point>::Ptr cloud_f;
		pcl::PointCloud<Point>::Ptr filtered_scene;
		pcl::SACSegmentation<Point> seg;
		pcl::PointIndices::Ptr inliers;
		pcl::ModelCoefficients::Ptr coefs;
		pcl::PointCloud<Point>::Ptr cloud_plane;
		pcl::search::KdTree<Point>::Ptr tree;
		std::vector<pcl::PointIndices> cluster_indices;
		pcl::PassThrough<Point> pass;
};



int main (int argc, char** argv)
{
	pcl::console::parse_argument (argc, argv, "--seg_dthresh", seg_dthresh);
	pcl::console::parse_argument (argc, argv, "--c_tol", c_tol);

	ObjectExtractor oex;
	oex.run();
	return 0;
}
