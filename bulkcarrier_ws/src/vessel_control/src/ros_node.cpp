#include "vessel_control/koopman_lift.hpp"
#include "vessel_control/mpc_osqp.hpp"
#include "vessel_control/state_assembly.hpp"
#include "vessel_control/trajectory_tracker.hpp"
#include "vessel_control/types.hpp"

#include <std_msgs/Float64.h>
#include <vessel_msgs/ControlCmd.h>
#include <vessel_msgs/ControlStatus.h>
#include <vessel_msgs/TrajectoryRef.h>
#include <vessel_msgs/VesselState.h>

#include <ros/ros.h>

namespace {

constexpr const char* MODE_KOOPMAN = "koopman_mpc";
constexpr const char* MODE_BASELINE = "baseline";
constexpr double kDeg2Rad = 3.14159265358979323846 / 180.0;

}  // namespace

class VesselControlNode {
 public:
  VesselControlNode(ros::NodeHandle& nh, ros::NodeHandle& pnh) : nh_(nh), pnh_(pnh) {
    pnh_.param<std::string>("model_path", model_path_, "");
    pnh_.param<std::string>("control_mode", control_mode_, MODE_KOOPMAN);
    pnh_.param<double>("control_rate", control_rate_, 4.0);
    pnh_.param<int>("mpc/horizon_N", horizon_N_, vessel_control::DEFAULT_HORIZON);
    pnh_.param<bool>("tube/enabled", tube_enabled_, true);

    ins_sub_ = nh_.subscribe("/ins/state", 1, &VesselControlNode::onIns, this);
    traj_sub_ = nh_.subscribe("/trajectory/ref", 1, &VesselControlNode::onTraj, this);
    rudder_sub_ = nh_.subscribe("/sensors/rudder_deg", 1, &VesselControlNode::onRudder, this);
    rpm_sub_ = nh_.subscribe("/sensors/shaft_rpm", 1, &VesselControlNode::onRpm, this);
    cmd_pub_ = nh_.advertise<vessel_msgs::ControlCmd>("/control/cmd", 1);
    status_pub_ = nh_.advertise<vessel_msgs::ControlStatus>("/control/status", 1);

    if (control_mode_ == MODE_KOOPMAN) {
      if (!lift_.load(model_path_)) {
        ROS_ERROR("Koopman lift load failed");
      } else {
        mpc_.set_lift(&lift_);
        if (!mpc_.configure(lift_.bundle(), horizon_N_)) {
          ROS_ERROR("MPC configure failed");
        }
      }
    }

    timer_ = nh_.createTimer(ros::Duration(1.0 / control_rate_),
                             &VesselControlNode::onControlTick, this);
  }

 private:
  void onIns(const vessel_msgs::VesselState::ConstPtr& msg) {
    last_ins_ = *msg;
    have_ins_ = true;
  }

  void onTraj(const vessel_msgs::TrajectoryRef::ConstPtr& msg) {
    last_traj_ = *msg;
    have_traj_ = true;
  }

  void onRudder(const std_msgs::Float64::ConstPtr& msg) { rudder_deg_ = msg->data; have_rudder_ = true; }

  void onRpm(const std_msgs::Float64::ConstPtr& msg) { rpm_meas_ = msg->data; have_rpm_ = true; }

  std::vector<vessel_control::StateVector> build_reference_horizon(const ros::Time& t0) const {
    std::vector<vessel_control::StateVector> refs;
    refs.reserve(static_cast<std::size_t>(horizon_N_));
    if (!have_traj_ || last_traj_.points.empty()) {
      return refs;
    }
    const double Ts = lift_.bundle().Ts > 0 ? lift_.bundle().Ts : 0.25;
    for (int k = 0; k < horizon_N_; ++k) {
      ros::Time tk = t0 + ros::Duration(k * Ts);
      vessel_control::StateVector xref{};
      if (vessel_control::lookup_reference(last_traj_, tk, xref)) {
        refs.push_back(xref);
      }
    }
    return refs;
  }

  void onControlTick(const ros::TimerEvent&) {
    if (!have_ins_) {
      return;
    }

    vessel_control::StateVector x = vessel_control::assemble_state(last_ins_);
    vessel_control::LiftVector z{};
    vessel_control::InputVector u_prev{};
    u_prev[0] = have_rudder_ ? rudder_deg_ * kDeg2Rad : 0.0;
    u_prev[1] = have_rpm_ ? rpm_meas_ : last_ins_.u;

    vessel_control::MpcSolution sol;
    sol.success = false;
    sol.status = MODE_BASELINE;

    if (control_mode_ == MODE_KOOPMAN) {
      if (lift_.lift(x, z)) {
        const auto refs = build_reference_horizon(last_ins_.header.stamp);
        sol = mpc_.solve(x, z, u_prev, refs);
      } else {
        sol.status = "lift_failed";
      }
    }

    vessel_msgs::ControlCmd cmd;
    cmd.header.stamp = ros::Time::now();
    cmd.delta_cmd_rad = sol.u0[0];
    cmd.rpm_cmd = sol.u0[1];
    cmd_pub_.publish(cmd);

    vessel_msgs::ControlStatus st;
    st.header.stamp = cmd.header.stamp;
    st.control_mode = control_mode_;
    st.model_id = lift_.bundle().model_id.empty() ? model_path_ : lift_.bundle().model_id;
    st.solve_time_ms = sol.solve_time_ms;
    st.osqp_status = sol.status;
    st.tube_active = tube_enabled_;
    status_pub_.publish(st);
  }

  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;
  ros::Subscriber ins_sub_;
  ros::Subscriber traj_sub_;
  ros::Subscriber rudder_sub_;
  ros::Subscriber rpm_sub_;
  ros::Publisher cmd_pub_;
  ros::Publisher status_pub_;
  ros::Timer timer_;

  std::string model_path_;
  std::string control_mode_;
  double control_rate_{4.0};
  int horizon_N_{vessel_control::DEFAULT_HORIZON};
  bool tube_enabled_{true};

  vessel_msgs::VesselState last_ins_;
  vessel_msgs::TrajectoryRef last_traj_;
  double rudder_deg_{0.0};
  double rpm_meas_{0.0};
  bool have_ins_{false};
  bool have_traj_{false};
  bool have_rudder_{false};
  bool have_rpm_{false};

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
