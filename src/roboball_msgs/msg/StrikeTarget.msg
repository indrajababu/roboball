# Predicted ball interception target for the strike planner.
std_msgs/Header header

# Desired paddle pose at impact, in frame_id.
geometry_msgs/Pose impact_pose

# Time-to-impact from header.stamp.
builtin_interfaces/Duration time_to_impact

# Predicted ball velocity at impact (m/s in frame_id).
geometry_msgs/Vector3 ball_velocity_at_impact
