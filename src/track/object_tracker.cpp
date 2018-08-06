/**
 * TODO try and remove the hardcoded levels
 */

#include <track/object_tracker.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>

ObjectTracker::ObjectTracker ()
{
	flags = 0;

	object_sampling_radius = 0.005f;
	scene_sampling_radius  = 0.005f;
	hough_threshold        = 0.8f; // org: 5.0f
	hough_bin_size         = 0.01f;
	descriptor_radius      = 0.01f;
	ref_radius             = 0.015f;

	normal_estimation.setKSearch (5);

	descriptor_estimation.setRadiusSearch (descriptor_radius);

	// cluster settings
	clusterer.setHoughBinSize (hough_bin_size);
	clusterer.setHoughThreshold (hough_threshold);
	clusterer.setUseInterpolation (true);
	clusterer.setUseDistanceWeight (false);

	// reference frame settings
	ref_estimation.setFindHoles (true);
	ref_estimation.setRadiusSearch (ref_radius);
}

void ObjectTracker::toggle_tracking ()
{
	flags ^= TRACKING;
}

void ObjectTracker::set_hough_threshold (float v)
{
	hough_threshold = v;
	clusterer.setHoughThreshold (hough_threshold);
}

void ObjectTracker::set_hough_bin_size (float v)
{
	hough_bin_size = v;
	clusterer.setHoughBinSize (hough_bin_size);
}

void ObjectTracker::set_descriptor_radius (float v)
{
	descriptor_radius = v;
	descriptor_estimation.setRadiusSearch (descriptor_radius);
}

void ObjectTracker::set_scene_sampling_radius (float v)
{
	scene_sampling_radius = v;
}

void ObjectTracker::set_object_sampling_radius (float v)
{
	object_sampling_radius = v;
}

void ObjectTracker::set_reference_radius (float v)
{
	ref_radius = v;
	ref_estimation.setRadiusSearch (ref_radius);
}

void ObjectTracker::track (
		const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr &scene,
		std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> &rototranslations,
		std::vector<pcl::Correspondences> &clustered_correspondences)
{
	if (~flags & TRACKING) // we are not tracking skip
	{
		return;
	}

	// compute the normals of the scene
	pcl::PointCloud<pcl::Normal>::Ptr scene_normals (new pcl::PointCloud<pcl::Normal> ());
	normal_estimation.setInputCloud (scene);
	normal_estimation.compute (*scene_normals);

	// downsample cloud for scene and extract keypoints
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr scene_keypoints (new pcl::PointCloud<pcl::PointXYZRGBA> ());
	uniform_sampling.setInputCloud (scene);
	uniform_sampling.setRadiusSearch (scene_sampling_radius); // NOTE hardcoded
	uniform_sampling.filter (*scene_keypoints);

	// compute descriptors
	pcl::PointCloud<pcl::SHOT352>::Ptr scene_descriptors (new pcl::PointCloud<pcl::SHOT352> ());
	descriptor_estimation.setInputCloud (scene_keypoints);
	descriptor_estimation.setInputNormals (scene_normals);
	descriptor_estimation.setSearchSurface (scene);
	descriptor_estimation.compute (*scene_descriptors);

	// find object-scene correspondences with KdTree
	pcl::CorrespondencesPtr object_scene_correspondences (new pcl::Correspondences ());
	pcl::KdTreeFLANN<pcl::SHOT352> match_search;
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
		if (neighbors_found == 1 && neighbor_sqr_distances[0] < 0.25f)
		{
			pcl::Correspondence c (neighbor_indices[0], static_cast<int> (i), neighbor_sqr_distances[0]);
			object_scene_correspondences->push_back (c);
		}
	}

	std::cout << "correspondences found: " << object_scene_correspondences->size () << std::endl;

	// compute (keypoint) reference frames (for Hough)
	pcl::PointCloud<pcl::ReferenceFrame>::Ptr scene_ref_frame (new pcl::PointCloud<pcl::ReferenceFrame> ());
	ref_estimation.setInputCloud (scene_keypoints);
	ref_estimation.setInputNormals (scene_normals);
	ref_estimation.setSearchSurface (scene);
	ref_estimation.compute (*scene_ref_frame);

	// cluster
	clusterer.setInputCloud (object_keypoints);
	clusterer.setInputRf (obj_ref_frame);
	clusterer.setSceneCloud (scene_keypoints);
	clusterer.setSceneRf (scene_ref_frame);
	clusterer.setModelSceneCorrespondences (object_scene_correspondences);
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
}

void ObjectTracker::set_object (pcl::PointCloud<pcl::PointXYZRGBA>::Ptr model)
{
	// set new object and compute it's normals, keypoints and descriptors
	object_model = model;

	// normals
	object_normals = pcl::PointCloud<pcl::Normal>::Ptr (new pcl::PointCloud<pcl::Normal> ());
	normal_estimation.setInputCloud (object_model);
	normal_estimation.compute (*object_normals);

	// keypoints
	object_keypoints = pcl::PointCloud<pcl::PointXYZRGBA>::Ptr (new pcl::PointCloud<pcl::PointXYZRGBA> ());
	uniform_sampling.setInputCloud (object_model);
	uniform_sampling.setRadiusSearch (object_sampling_radius); // NOTE hardcoded
	uniform_sampling.filter (*object_keypoints);

	// descriptors
	object_descriptors = pcl::PointCloud<pcl::SHOT352>::Ptr (new pcl::PointCloud<pcl::SHOT352> ());
	descriptor_estimation.setInputCloud (object_keypoints);
	descriptor_estimation.setInputNormals (object_normals);
	descriptor_estimation.setSearchSurface (object_model);
	descriptor_estimation.compute (*object_descriptors);

	// reference frame estimation
	obj_ref_frame = pcl::PointCloud<pcl::ReferenceFrame>::Ptr (new pcl::PointCloud<pcl::ReferenceFrame> ());
	ref_estimation.setInputCloud (object_keypoints);
	ref_estimation.setInputNormals (object_normals);
	ref_estimation.setSearchSurface (object_model);
	ref_estimation.compute (*obj_ref_frame);
}
