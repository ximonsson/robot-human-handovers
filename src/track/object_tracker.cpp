#include <track/object_tracker.h>
#include <pcl/io/openni_grabber.h>
#include <pcl/correspondence.h>
#include <pcl/recognition/cg/hough_3d.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/features/board.h>
#include <pcl/visualization/pcl_visualizer.h>

ObjectTracker::ObjectTracker () : viewer ("PCL OpenNI Viewer")
{
	normal_estimation.setKSearch (10);
	descriptor_estimation.setRadiusSearch (0.02f); // NOTE hardcoded
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

void ObjectTracker::visualize (const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr &scene, std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> rototranslations)
{
	if (!viewer.wasStopped ())
	{
		viewer.removeAllPointClouds ();
		viewer.addPointCloud (scene, "scene_cloud");
	}

	// visualize the found object
	for (size_t i = 0; i < rototranslations.size (); i++)
	{
		pcl::PointCloud<pcl::PointXYZRGBA>::Ptr rotated_model (new pcl::PointCloud<pcl::PointXYZRGBA> ());
		pcl::transformPointCloud (*object_model, *rotated_model, rototranslations[i]);
		std::stringstream ss_cloud;
		ss_cloud << "instance" << i;

		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGBA> rotated_model_color_handler (
				rotated_model,
				255,
				0,
				0);
		viewer.addPointCloud (rotated_model, rotated_model_color_handler, ss_cloud.str ());
	}
}

void ObjectTracker::track (const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr &scene)
{
	// compute the normals of the scene
	pcl::PointCloud<pcl::Normal>::Ptr scene_normals (new pcl::PointCloud<pcl::Normal>);
	normal_estimation.setInputCloud (scene);
	normal_estimation.compute (*scene_normals);

	// downsample cloud for scene and extract keypoints
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr scene_keypoints (new pcl::PointCloud<pcl::PointXYZRGBA>);
	uniform_sampling.setInputCloud (scene);
	uniform_sampling.setRadiusSearch (0.01f); // NOTE hardcoded
	uniform_sampling.filter (*scene_keypoints);

	// compute descriptors
	pcl::PointCloud<pcl::SHOT352>::Ptr scene_descriptors (new pcl::PointCloud<pcl::SHOT352>);
	descriptor_estimation.setInputCloud (scene_keypoints);
	descriptor_estimation.setInputNormals (scene_normals);
	descriptor_estimation.setSearchSurface (scene);
	descriptor_estimation.compute (*scene_descriptors);

	// find object-scene correspondences with KdTree
	pcl::CorrespondencesPtr object_scene_correspondences (new pcl::Correspondences ());
	pcl::KdTreeFLANN <pcl::SHOT352> match_search;
	match_search.setInputCloud (object_descriptors);

	// for each scene keypoint descriptor, find nearest neighbor into the object keypoints descriptor
	// cloud and add it to the correspondences vector
	for (size_t i = 0; i < scene_descriptors->size (); i++)
	{
		std::vector<int> neighbor_indices (1);
		std::vector<float> neighbor_sqr_distances (1);
		if (!pcl_isfinite (scene_descriptors->at (i).descriptor[0]))
		{
			continue; // skipping NaNs
		}
		int neighbors_found = match_search.nearestKSearch (
				scene_descriptors->at (i),
				1,
				neighbor_indices,
				neighbor_sqr_distances);
		// add match only if the squared descriptor distance is less than 0.25
		if (neighbors_found == 1 && neighbor_sqr_distances[0] < .25f)
		{
			pcl::Correspondence c (neighbor_indices[0], static_cast<int> (i), neighbor_sqr_distances[0]);
			object_scene_correspondences->push_back (c);
		}
	}

	// compute (keypoint) reference frames (for Hough)
	pcl::PointCloud<pcl::ReferenceFrame>::Ptr obj_ref_frame (new pcl::PointCloud<pcl::ReferenceFrame>);
	pcl::PointCloud<pcl::ReferenceFrame>::Ptr scene_ref_frame (new pcl::PointCloud<pcl::ReferenceFrame>);

	pcl::BOARDLocalReferenceFrameEstimation<pcl::PointXYZRGBA, pcl::Normal, pcl::ReferenceFrame> ref_estimation;
	ref_estimation.setFindHoles (true);
	ref_estimation.setRadiusSearch (.015f);

	ref_estimation.setInputCloud (object_keypoints);
	ref_estimation.setInputNormals (object_normals);
	ref_estimation.setSearchSurface (object_model);
	ref_estimation.compute (*obj_ref_frame);

	ref_estimation.setInputCloud (scene_keypoints);
	ref_estimation.setInputNormals (scene_normals);
	ref_estimation.setSearchSurface (scene);
	ref_estimation.compute (*scene_ref_frame);

	// cluster
	pcl::Hough3DGrouping<pcl::PointXYZRGBA, pcl::PointXYZRGBA, pcl::ReferenceFrame, pcl::ReferenceFrame> clusterer;
	clusterer.setHoughBinSize (0.01f);
	clusterer.setHoughThreshold (5.0f);
	clusterer.setUseInterpolation (true);
	clusterer.setUseDistanceWeight (false);

	clusterer.setInputCloud (object_keypoints);
	clusterer.setInputRf (obj_ref_frame);
	clusterer.setSceneCloud (scene_keypoints);
	clusterer.setSceneRf (scene_ref_frame);
	clusterer.setModelSceneCorrespondences (object_scene_correspondences);

	std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> rototranslations;
	std::vector<pcl::Correspondences> clustered_correspondences;
	clusterer.recognize (rototranslations, clustered_correspondences);

	// output results
	std::cout << "Model instances found: " << rototranslations.size () << std::endl;
	for (size_t i = 0; i < rototranslations.size (); ++i)
	{
		std::cout << "\n    Instance " << i + 1 << ":" << std::endl;
		std::cout << "        Correspondences belonging to this instance: " << clustered_correspondences[i].size () << std::endl;

		// Print the rotation matrix and translation vector
		Eigen::Matrix3f rotation = rototranslations[i].block<3,3>(0, 0);
		Eigen::Vector3f translation = rototranslations[i].block<3,1>(0, 3);

		printf ("\n");
		printf ("            | %6.3f %6.3f %6.3f | \n", rotation (0,0), rotation (0,1), rotation (0,2));
		printf ("        R = | %6.3f %6.3f %6.3f | \n", rotation (1,0), rotation (1,1), rotation (1,2));
		printf ("            | %6.3f %6.3f %6.3f | \n", rotation (2,0), rotation (2,1), rotation (2,2));
		printf ("\n");
		printf ("        t = < %0.3f, %0.3f, %0.3f >\n", translation (0), translation (1), translation (2));
	}

	visualize (scene, rototranslations);
	while (!viewer.wasStopped ())
		viewer.spinOnce();
}

void ObjectTracker::set_object (pcl::PointCloud<pcl::PointXYZRGBA>::Ptr model)
{
	// set new object and compute it's normals, keypoints and descriptors
	object_model = model;

	object_normals = pcl::PointCloud<pcl::Normal>::Ptr (new pcl::PointCloud<pcl::Normal> ());
	normal_estimation.setInputCloud (object_model);
	normal_estimation.compute (*object_normals);

	object_keypoints = pcl::PointCloud<pcl::PointXYZRGBA>::Ptr (new pcl::PointCloud<pcl::PointXYZRGBA> ());
	uniform_sampling.setInputCloud (object_model);
	uniform_sampling.setRadiusSearch (0.01f); // NOTE hardcoded
	uniform_sampling.filter (*object_keypoints);

	object_descriptors = pcl::PointCloud<pcl::SHOT352>::Ptr (new pcl::PointCloud<pcl::SHOT352> ());
	descriptor_estimation.setInputCloud (object_keypoints);
	descriptor_estimation.setInputNormals (object_normals);
	descriptor_estimation.setSearchSurface (object_model);
	descriptor_estimation.compute (*object_descriptors);
}
