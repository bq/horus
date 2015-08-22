# Horus installation in Synology DSM ![][synology-logo]

[return to Home](../../README.md)

Firts of all, you need a Synology Nas updated, with ssh access with root credentials. [Guide](http://forum.synology.com/wiki/index.php/Enabling_the_Command_Line_Interface)

Then, you have to Install [Debian Chroot](https://synocommunity.com/package/debian-chroot) adding the Synocomunity repo. As they says:

Step 1:- Log into your NAS as administrator and go to Main Menu → Package Center → Settings and set Trust Level to Synology Inc. and trusted publishers.

Step 2:- In the Package Sources tab, click Add, type SynoCommunity as Name and http://packages.synocommunity.com/ as Location and then press OK to validate.

Step 3:- Go back to the Package Center and enjoy SynoCommunity's packages in the Community tab.


At these point, we have two methods to install:


### The easy one (Automatic and recomended)

Open a ssh connection with your synology Nas and follow these instrucctions:

```bash
ssh root@<yournas>.local -p XXXX

/var/packages/debian-chroot/scripts/start-stop-status chroot

### Now we are in the console of Debian Chroot

cd
wget https://github.com/javivi001/horus/blob/develop/syno-horus.sh
bash syno-horus.sh

### Make coffe and take patience, but everything will work with only a pair of cuestions at beginig
```
Enjoy Horus!!!




### The manual one


Open a ssh connection as the other method:

```bash
ssh root@<yournas>.local -p XXXX

/var/packages/debian-chroot/scripts/start-stop-status chroot

... Now we are in the console of Debian Chroot ...
```


Update packages:

```bash
apt-get update && apt-get upgrade -y && apt-get dist-upgrade -y
```


Install Locales and configure it:

```bash
apt-get install locales
dpkg-reconfigure locales
```


Install Basic comands and eviroment dependences:

```bash
apt-get install git gitk -y
apt-get install python2.7 python-pip python-wxgtk3.0 python-serial python-opengl python-pyglet python-numpy python-scipy python-matplotlib libdc1394-utils libdc1394-22 -y
pip install -U pip setuptools
```


Install a desktop suite:

```bash
apt-get install lxde -y
```


Install request for compilling Open CV:

```bash
apt-get install build-essential cmake make git gcc g++ gfortran libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev python-dev python-numpy \
libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev libv4l-dev libeigen3-dev -y
```

Install an VNC Server that allow OpenGL over X11

```bash
apt-get install vnc4server libgl1-mesa-swx11-dev-y

Start:$ vncserver -geometry 10280x1024 -depth 24 :1 && export DISPLAY=':1'
Stop:$  vncserver -kill :1
```


Let's start compilling and installing custom OpenCV:

```bash
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
```


And now it's time to Horus:

```bash
cd /root/
git clone https://github.com/bq/horus.git

or if you prefer my fort:

git clone https://github.com/javivi001/horus.git
cd horus
python src/horus.py
```

Enjoy Horus!!!


### IMPORTANT : Check yor kernel modules if you have in /lib/modules

You will nedd to add modules (in these order, very important!):

 /lib/modules/videodev.ko
 /lib/modules/v4l2-int-device.ko
 /lib/modules/v4l2-common.ko
 /lib/modules/uvcvideo.ko
 /lib/modules/usbserial.ko
 /lib/modules/ftdi_sio.ko

If you don't have it, you can crosscompile in other pc with the synology chaintools or try to croscompile in the Debian Chroot (these work for me)

Ask me your questions!!!


Regards!!!

[synology-logo]: ../images/synology.png
