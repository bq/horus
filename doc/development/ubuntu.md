#Horus development in Ubuntu

[return to Home](../../README.md)

If you are a developer and you want to modify the code, contribute, build packages, etc. you may follow this steps

## 1. Set up the environment

### Tools

#### Atom IDE
Download open source [Atom code editor](https://atom.io/).

#### Git version control
```bash
sudo apt-get install git gitk
```

### Dependencies

Following dependencies are included in deb package, but if you want to install it manually, they are:

#### Python modules
```bash
sudo apt-get install python-serial python-opengl python-pyglet python-numpy python-scipy python-matplotlib
```

##### wxPython

For older ubuntu versions

```bash
sudo apt-get install python-wxgtk2.8
```

For newer ubuntu versions

```bash
sudo apt-get install python-wxgtk3.0
```

#### Custom OpenCV

*NOTE*: first try to remove previous versions of opencv:

```bash
sudo apt-get remove python-opencv
sudo apt-get autoremove
```

```bash
sudo add-apt-repository ppa:bqlabs/horus
sudo apt-get update
sudo apt-get install python-opencv
```

#### AVRDUDE
```bash
sudo apt-get install avrdude  # include libftdi1
```

#### Video 4 Linux
```bash
sudo apt-get install v4l-utils
```

In order to generate Debian and Windows packages, some extra dependencies are needed

#### Packaging dependencies
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

### Custom OpenCV

Several improvements and optimizations have been made in GNU/Linux version of OpenCV libraries. If you want to contribute to this custom version, you can download it from:

```bash
git clone https://github.com/bq/opencv.git
```
or
```bash
git clone git@github.com:bq/opencv.git
```

And build it your own: [instructions](https://github.com/LibreScanner/opencv/wiki/Build)

## 3. Execute source code

In the project directory, execute the command:

```bash
./horus
```

### Unit testing

To run the tests install nose:

```bash
sudo -H pip install -U nose
```

And execute:

```bash
nosetests test
```

## 4. Build packages

Horus development comes with a script *package.sh*. This script generates a final release package. You should not need it during development, unless you are changing the release process. If you want to distribute your own version of Horus, then the *package.sh* script will allow you to do that.

### Version
```bash
bash package.sh version  # Generate version file
```

### GNU/Linux Ubuntu
```bash
bash package.sh debian     # Generate deb package
bash package.sh debian -s  # Generate sources
bash package.sh debian -i  # Install deb package
bash package.sh debian -u  # Upload to launchpad
```

### Windows
```bash
bash package.sh win32        # Generate exe package
bash package.sh win32 /path  # Generate exe using /path for deps
```
