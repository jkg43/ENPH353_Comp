#include <iostream>
#include <cmath>
#include <functional>
#include <thread>
#include <mutex>
#include <ros/ros.h>
#include <std_msgs/Float32.h>
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/JointState.h>
#include <drone_internals/Pose.h>
#include <drone_internals/MotorSpeed.h>
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <ignition/math/Vector3.hh>
#include <ignition/math/Quaternion.hh>
#include <ignition/math/Pose3.hh>
#include <tf/transform_broadcaster.h>

// slightly modified from https://deltastep.blogspot.com/2020/01/simulating-drone-in-gazebo-part-2.html


class DronePlugin : public gazebo::ModelPlugin
{
public:
  	DronePlugin() : gazebo::ModelPlugin()
	{
		std::cout << "Starting drone_plugin" << std::endl;
  	}
 
  	virtual ~DronePlugin()
	{
		std::cout << "Closing drone_plugin" << std::endl;
  	}


  	void Load(gazebo::physics::ModelPtr parent, sdf::ElementPtr sdf)
	{
		_model = parent;
   
		if (sdf->HasElement("updateRate"))
		{
			_rate = sdf->GetElement("updateRate")->Get<double>();
		}
		else
		{
			_rate = 100.0;
		}
	
		if (sdf->HasElement("publishTf"))
		{
			_publish_tf = sdf->GetElement("publishTf")->Get<bool>();
		}
		else
		{
			_publish_tf = true;
		}
	
		if (sdf->HasElement("rotorThrustCoeff"))
		{
			_rotor_thrust_coeff = sdf->GetElement("rotorThrustCoeff")->Get<double>();
		}
		else
		{
			_rotor_thrust_coeff = 0.00025;
		}
	
		if (sdf->HasElement("rotorTorqueCoeff"))
		{
			_rotor_torque_coeff = sdf->GetElement("rotorTorqueCoeff")->Get<double>();
		}
		else
		{
			_rotor_torque_coeff = 0.0000074;
		}

		if (!ros::isInitialized())
		{
			int argc = 0;
			char** argv = NULL;
			ros::init(argc, argv, "drone_internals",
			ros::init_options::NoSigintHandler);
		}

		_nh = new ros::NodeHandle("");
		_pose_pub = _nh->advertise<drone_internals::Pose>("pose", 100);
		_cmd_sub = _nh->subscribe("motor_speed_cmd", 100, &DronePlugin::onMotorSpeedsMsg, this);
		_ros_thread = std::thread(std::bind(&DronePlugin::rosThread, this));

		_updateConnection = gazebo::event::Events::ConnectWorldUpdateBegin(std::bind(&DronePlugin::onUpdate, this));
  	}

	void onMotorSpeedsMsg(const drone_internals::MotorSpeed::ConstPtr& msg)
	{
		_cmd_mtx.lock();
		_motor_speed_msg = *msg;
		_cmd_mtx.unlock();
	}

	 void rosThread()
	 {
		ros::Rate rate(_rate);
		while (ros::ok()) {
			ros::spinOnce();
			publishDronePose();
			rate.sleep();
		}
	}

	void publishDronePose()
	{
		_pose_mtx.lock();
		ignition::math::Pose3d pose = _pose;
		_pose_mtx.unlock();
		
		ignition::math::Vector3d rpy = pose.Rot().Euler();
		tf::Quaternion q(pose.Rot().X(), pose.Rot().Y(), pose.Rot().Z(), pose.Rot().W());
		tf::Matrix3x3 m(q);
		double roll, pitch, yaw;
		m.getRPY(roll, pitch, yaw);
		
		drone_internals::Pose pose_msg;
		pose_msg.x = pose.Pos().X();
		pose_msg.y = pose.Pos().Y();
		pose_msg.z = pose.Pos().Z();
		pose_msg.roll = roll;
		pose_msg.pitch = -pitch;
		pose_msg.yaw = yaw;
		_pose_pub.publish(pose_msg);
		
		if (_publish_tf)
		{
			tf::Transform T;
			T.setOrigin(tf::Vector3(pose.Pos().X(), pose.Pos().Y(), pose.Pos().Z()));
			T.setRotation(tf::Quaternion(pose.Rot().X(), pose.Rot().Y(), pose.Rot().Z(), pose.Rot().W()));
			_tf.sendTransform(tf::StampedTransform(T, ros::Time::now(), "world", "drone"));
		}
	}

	void onUpdate()
	{
		_pose_mtx.lock();
		_pose = _model->WorldPose();
		_pose_mtx.unlock();
		
		updateThrust();
	}

	void updateThrust()
	{
		_cmd_mtx.lock();
		drone_internals::MotorSpeed cmd = _motor_speed_msg;
		_cmd_mtx.unlock();
		
		int n = cmd.name.size();
		for (int i = 0; i < n; ++i)
		{
			double thrust = calculateThrust(cmd.velocity[i]);
			double torque = calculateTorque(cmd.velocity[i]);
			gazebo::physics::LinkPtr link = _model->GetLink(cmd.name[i]);
			if (link != NULL)
			{
				link->AddLinkForce(ignition::math::Vector3d(0, 0, thrust));
				link->AddRelativeTorque(ignition::math::Vector3d(0, 0, torque));
			}
		}
	}

	double calculateThrust(double w)
	{
		double thrust = _rotor_thrust_coeff * w * w;
		return thrust;
	}

	double calculateTorque(double w)
	{
		double torque = copysign(_rotor_torque_coeff * w * w, w);
		return torque;
	}


 
private:
	ros::NodeHandle* _nh;
	ros::Publisher _pose_pub;
	ros::Subscriber _cmd_sub;
	tf::TransformBroadcaster _tf;
	std::thread _ros_thread;
	std::mutex _pose_mtx;
	std::mutex _cmd_mtx;
	double _rate;
	bool _publish_tf;
	double _rotor_thrust_coeff;
	double _rotor_torque_coeff;
	drone_internals::MotorSpeed _motor_speed_msg;
	
	gazebo::physics::ModelPtr _model;
	gazebo::event::ConnectionPtr _updateConnection;
	ignition::math::Pose3d _pose;



};
 
GZ_REGISTER_MODEL_PLUGIN(DronePlugin)