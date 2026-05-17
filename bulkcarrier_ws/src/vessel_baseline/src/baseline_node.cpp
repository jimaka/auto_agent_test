#include "vessel_baseline/nomoto_mpc.hpp"

#include <vessel_msgs/ControlCmd.h>
#include <vessel_msgs/ControlStatus.h>
#include <vessel_msgs/VesselState.h>

#include <ros/ros.h>

int main(int argc, char** argv) {
  ros::init(argc, argv, "vessel_baseline_node");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");

  double rate = 4.0;
  pnh.param("control_rate", rate, 4.0);
  vessel_baseline::NomotoMpc mpc;

  ros::Publisher cmd_pub = nh.advertise<vessel_msgs::ControlCmd>("/control/cmd", 1);
  ros::Publisher status_pub =
      nh.advertise<vessel_msgs::ControlStatus>("/control/status", 1);

  vessel_msgs::VesselState last_ins;
  bool have_ins = false;
  ros::Subscriber ins_sub = nh.subscribe<vessel_msgs::VesselState>(
      "/ins/state", 1, [&](const vessel_msgs::VesselState::ConstPtr& msg) {
        last_ins = *msg;
        have_ins = true;
      });

  ros::Timer timer = nh.createTimer(
      ros::Duration(1.0 / rate),
      [&](const ros::TimerEvent&) {
        if (!have_ins) return;
        auto sol = mpc.solve(0.0, 0.0, last_ins.u - 5.0, 0.0, last_ins.u);

        vessel_msgs::ControlCmd cmd;
        cmd.header.stamp = ros::Time::now();
        cmd.delta_cmd_rad = sol.u0[0];
        cmd.rpm_cmd = sol.u0[1];
        cmd_pub.publish(cmd);

        vessel_msgs::ControlStatus st;
        st.header = cmd.header;
        st.control_mode = "baseline";
        st.model_id = "nomoto_mpc_v1";
        st.osqp_status = sol.success ? "ok" : "fail";
        status_pub.publish(st);
      });

  ros::spin();
  return 0;
}
