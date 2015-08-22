#!/bin/sh

apt-get update && apt-get upgrade -y && apt-get dist-upgrade -y
apt-get install locales
dpkg-reconfigure locales
apt-get install git gitk -y
apt-get install python2.7 python-pip python-wxgtk3.0 python-serial python-opengl python-pyglet python-numpy python-scipy python-matplotlib libdc1394-utils libdc1394-22 -y
pip install -U pip setuptools
apt-get install make cmake gcc g++ gfortran -y
apt-get install lxde -y
apt-get install build-essential cmake make git gcc g++ gfortran libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev python-dev python-numpy \
libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev libv4l-dev libeigen3-dev -y
apt-get install vnc4server libgl1-mesa-swx11-dev-y

cd /root/
git clone https://github.com/bq/opencv.git
cd opencv
mkdir release
cd release
cmake -D CMAKE_C_FLAGS=-m32 -D CMAKE_CXX_FLAGS=-m32 -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_TBB=ON -D BUILD_NEW_PYTHON_SUPPORT=ON \
-D WITH_V4L=ON -D WITH_FFMPEG=OFF -D WITH_OPENGL=OFF -D BUILD_opencv_gpu=OFF -D BUILD_opencv_gpu=OFF -D BUILD_opencv_ocl=OFF -D BUILD_opencv_nonfree=OFF \
-D BUILD_opencv_stitching=OFF -D BUILD_opencv_superres=OFF -D BUILD_opencv_ts=OFF -D BUILD_opencv_videostab=OFF ..
make
make install

cd /root/
git clone https://github.com/javivi001/horus.git
cd horus
