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
#include <pcl/io/openni2_grabber.h>
#include <iostream>
#include <pcl/point_types.h>
#include <pcl/filters/model_outlier_removal.h>
#include <pcl/console/parse.h>

typedef pcl::PointXYZRGBA Point;
typedef pcl::PointCloud<Point> PointCloud;

float seg_dthresh (0.02);
float c_tol (0.02); // 2cm

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
			cloud_filtered = PointCloud::Ptr (new PointCloud);
			cloud_f = PointCloud::Ptr (new PointCloud);
			inliers = pcl::PointIndices::Ptr (new pcl::PointIndices);
			coefs = pcl::ModelCoefficients::Ptr (new pcl::ModelCoefficients);
			cloud_plane = PointCloud::Ptr (new PointCloud);
			coefs = pcl::ModelCoefficients::Ptr (new pcl::ModelCoefficients);
			tree = pcl::search::KdTree<Point>::Ptr (new pcl::search::KdTree<Point>);

			// set parameters for segmentation
			seg.setOptimizeCoefficients (true);
			seg.setModelType (pcl::SACMODEL_PLANE);
			seg.setMethodType (pcl::SAC_RANSAC);
			seg.setMaxIterations (200);
			seg.setDistanceThreshold (seg_dthresh);
		}

		void visualize_cluster (const PointCloud::ConstPtr &cluster, int j)
		{
			std::stringstream ss;
			ss << "cluster_" << j;
			pcl::visualization::PointCloudColorHandlerCustom<Point> color_handler (cluster, colors[j][0], colors[j][1], colors[j][2]);
			viewer.addPointCloud (cluster, color_handler, ss.str ());
		}

		void visualize (const PointCloud::ConstPtr &scene)
		{
			if (!viewer.wasStopped ())
			{
				viewer.removeAllPointClouds ();
				viewer.addPointCloud (scene, "scene_cloud");

				// visualize all the clusters
				int j = 0;
				for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); it++)
				{
					PointCloud::Ptr cloud_cluster (new PointCloud);
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

		void extract (const PointCloud::ConstPtr &scene)
		{
			// passthrough
			PointCloud::Ptr filtered_scene (new PointCloud);
			pass.setInputCloud (scene);
			pass.setFilterFieldName ("z");
			pass.setFilterLimits (0.3f, 0.9f);
			pass.filter (*cloud_filtered);

			pass.setInputCloud (cloud_filtered);
			pass.setFilterFieldName ("x");
			pass.setFilterLimits (-0.5f, 0.5f);
			pass.filter (*filtered_scene);

			pass.setInputCloud (filtered_scene);
			pass.setFilterFieldName ("y");
			pass.setFilterLimits (-0.5f, 0.5f);
			pass.filter (*cloud_filtered);

			*filtered_scene = *cloud_filtered;

			//vg.setInputCloud (filtered_scene);
			//vg.setLeafSize (0.001f, 0.001f, 0.001f);
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
			ec.setClusterTolerance (c_tol);
			ec.setMinClusterSize (500);
			ec.setMaxClusterSize (10000);
			ec.setSearchMethod (tree);
			ec.setInputCloud (cloud_filtered);
			ec.extract (cluster_indices);

			visualize (filtered_scene);
		}

		void snapshot ()
		{
			boost::mutex::scoped_lock lock (mut);
			pcl::PCDWriter writer;
			int j = 0;
			for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
			{
				PointCloud::Ptr cloud_cluster (new PointCloud);
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


		void cb (const PointCloud::ConstPtr &cloud)
		{
			boost::mutex::scoped_lock lock (mut);
			scene = cloud;
		}

		void run ()
		{
			// register for keyboard events
			boost::function<void (const pcl::visualization::KeyboardEvent&)> key_cb =
				boost::bind (&ObjectExtractor::handle_key, this, _1);
			viewer.registerKeyboardCallback (key_cb);

			// create an interface towards the Kinect using OpenNI and start grabbing
			// RGBD frames that are sent to the ObjectTracker::track function as callback
			pcl::Grabber* grabber = new pcl::io::OpenNI2Grabber ();
			boost::function<void (const PointCloud::ConstPtr&)> f =
				boost::bind (&ObjectExtractor::cb, this, _1);
			boost::signals2::connection conn = grabber->registerCallback (f);
			grabber->start ();
			while (!viewer.wasStopped ())
			{
				PointCloud::ConstPtr c;
				if (mut.try_lock ())
				{
					scene.swap (c);
					mut.unlock ();
				}

				if (c)
				{
					extract (c);
				}
			}
			grabber->stop ();
			conn.disconnect ();
		}

		pcl::visualization::PCLVisualizer viewer;
		pcl::VoxelGrid<Point> vg; // filtering object
		PointCloud::Ptr cloud_filtered;
		PointCloud::Ptr cloud_f;
		PointCloud::ConstPtr scene;
		PointCloud::Ptr filtered_scene;
		pcl::SACSegmentation<Point> seg;
		pcl::PointIndices::Ptr inliers;
		pcl::ModelCoefficients::Ptr coefs;
		PointCloud::Ptr cloud_plane;
		pcl::search::KdTree<Point>::Ptr tree;
		std::vector<pcl::PointIndices> cluster_indices;
		pcl::PassThrough<Point> pass;
		boost::mutex mut;
};



int main (int argc, char** argv)
{
	// extract parameters
	pcl::console::parse_argument (argc, argv, "--seg_dthresh", seg_dthresh);
	pcl::console::parse_argument (argc, argv, "--c_tol", c_tol);

	// run the object extractor
	ObjectExtractor oex;
	oex.run();
	return 0;
}
