# Estimated state of the ball at a point in time, expressed in `frame_id`.
std_msgs/Header header

# Position (m) and velocity (m/s) in frame_id.
geometry_msgs/Point position
geometry_msgs/Vector3 velocity

# True when the predictor has at least N samples and the fit residual is acceptable.
bool fit_valid
