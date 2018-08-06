#include <pcl/io/openni2_grabber.h>
template <typename PointType>
class OpenNI2Viewer
{
	public:
		typedef pcl::PointCloud<PointType> Cloud;
		typedef typename Cloud::ConstPtr CloudConstPtr;

		OpenNI2Viewer (pcl::io::OpenNI2Grabber& grabber)
			: cloud_viewer_ (new pcl::visualization::PCLVisualizer ("PCL OpenNI2 cloud")), grabber_ (grabber)
		{
		}

		void cloud_callback (const CloudConstPtr& cloud)
		{
			boost::mutex::scoped_lock lock (cloud_mutex_);
			cloud_ = cloud;
		}

		void keyboard_callback (const pcl::visualization::KeyboardEvent& event, void*)
		{
			if (event.getKeyCode ())
				cout << "the key \'" << event.getKeyCode () << "\' (" << event.getKeyCode () << ") was";
			else
				cout << "the special key \'" << event.getKeySym () << "\' was";
			if (event.keyDown ())
				cout << " pressed" << endl;
			else
				cout << " released" << endl;
		}

		void mouse_callback (const pcl::visualization::MouseEvent& mouse_event, void*)
		{
			if (mouse_event.getType () == pcl::visualization::MouseEvent::MouseButtonPress && mouse_event.getButton () == pcl::visualization::MouseEvent::LeftButton)
			{
				cout << "left button pressed @ " << mouse_event.getX () << " , " << mouse_event.getY () << endl;
			}
		}

		/**
		 * @brief starts the main loop
		 */
		void run ()
		{
			cloud_viewer_->registerMouseCallback (&OpenNI2Viewer::mouse_callback, *this);
			cloud_viewer_->registerKeyboardCallback (&OpenNI2Viewer::keyboard_callback, *this);
			//cloud_viewer_->setCameraFieldOfView (1.02259994f);
			boost::function<void (const CloudConstPtr&) > cloud_cb = boost::bind (&OpenNI2Viewer::cloud_callback, this, _1);
			boost::signals2::connection cloud_connection = grabber_.registerCallback (cloud_cb);

			bool cloud_init = false;
			grabber_.start ();
			while (!cloud_viewer_->wasStopped ())
			{
				CloudConstPtr cloud;
				cloud_viewer_->spinOnce ();

				// See if we can get a cloud
				if (cloud_mutex_.try_lock ())
				{
					cloud_.swap (cloud);
					cloud_mutex_.unlock ();
				}

				if (cloud)
				{
					if (!cloud_init)
					{
						cloud_viewer_->setPosition (0, 0);
						cloud_viewer_->setSize (cloud->width, cloud->height);
						cloud_init = !cloud_init;
					}

					if (!cloud_viewer_->updatePointCloud (cloud, "OpenNICloud"))
					{
						cloud_viewer_->addPointCloud (cloud, "OpenNICloud");
					}
				}
			}
			grabber_.stop ();
			cloud_connection.disconnect ();
		}

		boost::shared_ptr<pcl::visualization::PCLVisualizer> cloud_viewer_;
		pcl::io::OpenNI2Grabber& grabber_;
		boost::mutex cloud_mutex_;
		CloudConstPtr cloud_;
};
