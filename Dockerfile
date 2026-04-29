# Explicitly target ARM64 for your M4 Mac
FROM --platform=linux/arm64 osrf/ros:humble-desktop-full

# Avoid prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Desktop environment and NoVNC
# Removed vnc4server and added tigervnc
RUN apt-get update && apt-get install -y \
    xfce4 \
    xfce4-goodies \
    novnc \
    websockify \
    tigervnc-standalone-server \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install your specific ROS 2 packages
RUN apt-get update && apt-get install -y \
    ros-humble-libcurl-vendor \
    ros-humble-nav2-regulated-pure-pursuit-controller \
    ros-humble-resource-retriever \
    ros-humble-rqt-reconfigure \
    ros-humble-turtlesim \
    ros-humble-ur-msgs \
    ros-humble-ur-description \
    ros-humble-ur-calibration \
    ros-humble-ur-client-library \
    ros-humble-ur-controllers \
    ros-humble-ur-dashboard-msgs \
    ros-humble-ur-moveit-config \
    ros-humble-ur-robot-driver \
    ros-humble-urdf \
    ros-humble-urdf-parser-plugin \
    ros-humble-urdfdom-py \
    && rm -rf /var/lib/apt/lists/*

# Environment setup
ENV PATH="/opt/ros/humble/bin:$PATH"
ENV DISPLAY=:1

WORKDIR /ros2_ws

# Startup script to launch the virtual desktop and the web streamer
RUN echo "#!/bin/bash\n\
rm -rf /tmp/.X1-lock /tmp/.X11-unix/X1\n\
vncserver :1 -geometry 1280x800 -depth 24 -SecurityTypes None\n\
/usr/share/novnc/utils/launch.sh --vnc localhost:5901 --listen 8080" > /entrypoint.sh && chmod +x /entrypoint.sh

EXPOSE 8080

CMD ["/entrypoint.sh"]