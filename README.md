# Horus

Horus is a general solution for 3D scanning. It provides graphic user interfaces for connection, configuration, control, calibration and scanning. It is ready to use with Open Source [Ciclop 3D Scanner](http://diwo.bq.com/en/ciclop-released/) [[es](http://diwo.bq.com/ciclop-released/)]

This project has been developed in Python language and it is distributed under GPL v2 license.

More interest links are shown below:

* [Presentation](http://diwo.bq.com/en/presentacion-ciclop-horus/) [[es](http://diwo.bq.com/presentacion-ciclop-horus/)]
* [Electronics](http://diwo.bq.com/en/zum-scan-released/) [[es](http://diwo.bq.com/zum-scan-released/)]
* [Firmware](http://diwo.bq.com/en/horus-fw-released/) [[es](http://diwo.bq.com/horus-fw-released/)]
* [Software](http://diwo.bq.com/horus-released/) [[es](http://diwo.bq.com/horus-released/)]

# Installing

### GNU/Linux Ubuntu

Add our PPA keys

```bash
sudo add-apt-repository ppa:jesus-arroyo/opencv
sudo add-apt-repository ppa:jesus-arroyo/horus
```

Upgrade your system

```bash
sudo apt-get update
sudo apt-get dist-upgrade
```

Install Horus

```bash
sudo apt-get install horus
```

If user has no access to serial port, execute:

```bash
sudo usermod -a -G dialout $USER
```

Reboot the computer to apply the changes

```bash
sudo reboot
```

### Windows

To install USB Camera drivers follow these instructions [Logitech Camera C270 Drivers](http://support.logitech.com/en_us/product/hd-webcam-c270)

Execute .exe file. This package contains all dependencies

Reboot the computer to apply the changes


# Development

Horus has been developed in [Ubuntu Gnome](http://ubuntugnome.org/). If you are a developer and you want to modify the code, contribute, build packages, etc. you may follow this steps:

## 1. Set up the environment

### Tools

#### Sublime Text 3 IDE
```bash
sudo add-apt-repository ppa:webupd8team/sublime-text-3
sudo apt-get update
sudo apt-get install sublime-text-installer
```

#### Arduino IDE
```bash
sudo apt-get install arduino arduino-core
```

#### Stino plugin
[Stino project on GitHub](https://github.com/Robot-Will/Stino)

#### Git version control
```bash
sudo apt-get install git gitk
```

### Dependencies

Following dependencies are included in deb package, but if you want to install it manually, they are:

#### Python
```bash
sudo apt-get install python-serial python-wxgtk2.8 python-opengl python-pyglet python-numpy python-scipy python-matplotlib
```

#### OpenCV
```bash
sudo add-apt-repository ppa:bqopensource/opencv
sudo apt-get update
sudo apt-get install python-opencv
```

#### AVRDUDE
```bash
sudo apt-get install avrdude
```

#### FTDI drivers
```bash
sudo apt-get install libftdi1
```

#### Video 4 Linux
```bash
sudo apt-get install v4l-utils
```

In order to generate Debian and Windows packages, some extra dependencies are needed

#### Packaging
```bash
sudo apt-get install build-essential pkg-config python-dev python-stdeb p7zip-full curl nsis
```

## 2. Download source code

All source code is available on GitHub. You can download main Horus project by doing:

### Horus
```bash
git clone https://github.com/bq/horus.git
```
or
```bash
git clone git@github.com:bq/horus.git
```

Several improvements and optimizations have been made in GNU/Linux version of OpenCV libraries. If you want to contribute to this custom version, you can download it from:

### Custom OpenCV
```bash
git clone https://github.com/bq/opencv.git
```
or
```bash
git clone git@github.com:bq/opencv.git
```

## 3. Build packages

Horus development comes with a script "package.sh", this script has been designed to run under *nix OSes (Linux, MacOS). For Windows the package.sh script can be run from bash using git.
The "package.sh" script generates a final release package. You should not need it during development, unless you are changing the release process. If you want to distribute your own version of Horus, then the package.sh script will allow you to do that.

### GNU/Linux Ubuntu
```bash
bash package.sh debian     # Generate deb package
bash package.sh debian -s  # Generate sources
bash package.sh debian -i  # Install deb package
bash package.sh debian -u  # Upload to launchpad
```

### Windows
```bash
bash package.sh win32  # Generate exe package
```

### GNU/Linux Fedora

TODO

### Mac OS X

TODO