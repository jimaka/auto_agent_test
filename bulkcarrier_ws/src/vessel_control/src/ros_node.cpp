#include "vessel_control/koopman_lift.hpp"
#include "vessel_control/mpc_osqp.hpp"
#include "vessel_control/state_assembly.hpp"
#include "vessel_control/trajectory_tracker.hpp"
#include "vessel_control/types.hpp"

#include <vessel_msgs/ControlCmd.h>
#include <vessel_msgs/ControlStatus.h>
#include <vessel_msgs/TrajectoryRef.h>
#include <vessel_msgs/VesselState.h>
#include <vessel_msgs/Wind.h>

#include <ros/ros.h>

namespace {

constexpr const char* MODE_KOOPMAN = "koopman_mpc";
constexpr const char* MODE_BASELINE = "baseline";

}  // namespace

class VesselControlNode {
 public:
  VesselControlNode(ros::NodeHandle& nh, ros::NodeHandle& pnh) : nh_(nh), pnh_(pnh) {
    pnh_.param<std::string>("model_path", model_path_, "");
    pnh_.param<std::string>("control_mode", control_mode_, MODE_KOOPMAN);
    pnh_.param<double>("control_rate", control_rate_, 4.0);

    ins_sub_ = nh_.subscribe("/ins/state", 1, &VesselControlNode::onIns, this);
    traj_sub_ = nh_.subscribe("/trajectory/ref", 1, &VesselControlNode::onTraj, this);
    cmd_pub_ = nh_.advertise<vessel_msgs::ControlCmd>("/control/cmd", 1);
    status_pub_ = nh_.advertise<vessel_msgs::ControlStatus>("/control/status", 1);

    if (control_mode_ == MODE_KOOPMAN) {
      if (!lift_.load(model_path_) || !mpc_.configure(model_path_, vessel_control::DEFAULT_HORIZON)) {
        ROS_WARN("Koopman MPC init failed; switch to baseline externally");
      }
    }

    timer_ = nh_.createTimer(ros::Duration(1.0 / control_rate_),
                             &VesselControlNode::onControlTick, this);
  }

 private:
  void onIns(const vessel_msgs::VesselState::ConstPtr& msg) { last_ins_ = *msg; have_ins_ = true; }

  void onTraj(const vessel_msgs::TrajectoryRef::ConstPtr& msg) { last_traj_ = *msg; have_traj_ = true; }

  void onControlTick(const ros::TimerEvent&) {
    if (!have_ins_) {
      return;
    }

    vessel_control::StateVector x = vessel_control::assemble_state(last_ins_);
    vessel_control::LiftVector z{};
    vessel_control::InputVector u_prev{0.0, 0.0};
    vessel_control::MpcSolution sol;

    if (control_mode_ == MODE_KOOPMAN && lift_.lift(x, z)) {
      sol = mpc_.solve(x, z, u_prev);
    } else {
      sol.success = true;
      sol.status = MODE_BASELINE;
    }

    vessel_msgs::ControlCmd cmd;
    cmd.header.stamp = ros::Time::now();
    cmd.delta_cmd_rad = sol.u0[0];
    cmd.rpm_cmd = sol.u0[1];
    cmd_pub_.publish(cmd);

    vessel_msgs::ControlStatus st;
    st.header.stamp = cmd.header.stamp;
    st.control_mode = control_mode_;
    st.model_id = model_path_;
    st.solve_time_ms = sol.solve_time_ms;
    st.osqp_status = sol.status;
    st.tube_active = true;
    status_pub_.publish(st);
  }

  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;
  ros::Subscriber ins_sub_;
  ros::Subscriber traj_sub_;
  ros::Publisher cmd_pub_;
  ros::Publisher status_pub_;
  ros::Timer timer_;

  std::string model_path_;
  std::string control_mode_;
  double control_rate_{4.0};

  vessel_msgs::VesselState last_ins_;
  vessel_msgs::TrajectoryRef last_traj_;
  bool have_ins_{false};
  bool have_traj_{false};

  vessel_control::KoopmanLift lift_;
  vessel_control::MpcOsqp mpc_;
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "vessel_control_node");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");
  VesselControlNode node(nh, pnh);
  ros::spin();
  return 0;
}
