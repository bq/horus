#!/bin/bash

# This script is to package the Horus package for Windows/Linux
# This script should run under Linux and Mac OS X, as well as Windows with Cygwin.

#############################
# CONFIGURATION
#############################

##Select the build target
BUILD_TARGET=${1:-none}
#BUILD_TARGET=win32
#BUILD_TARGET=debian

EXTRA_ARGS=${2}

##Which version name are we appending to the final archive
VERSION=`head -1 pkg/linux/debian/changelog | grep -o '[0-9.]*' | head -1`
TARGET_DIR=Horus-${VERSION}-${BUILD_TARGET}

##Which versions of external programs to use
WIN_PORTABLE_PY_VERSION=2.7.2.1 #TODO: 2.7.6.1


#############################
# Support functions
#############################
function checkTool
{
	if [ -z `which $1` ]; then
		echo "The $1 command must be somewhere in your \$PATH."
		echo "Fix your \$PATH or install it"
		exit 1
	fi
}

function downloadURL
{
	filename=`basename "$1"`
	echo "Checking for $filename"
	if [ ! -f "$filename" ]; then
		echo "Downloading $1"
		curl -L -O "$1"
		if [ $? != 0 ]; then
			echo "Failed to download $1"
			exit 1
		fi
	fi
}

function extract
{
	echo "Extracting $*"
	echo "7z x -y $*" >> log.txt
	7z x -y $* >> log.txt
	if [ $? != 0 ]; then
        echo "Failed to extract $*"
        exit 1
	fi
}

#############################
# Actual build script
#############################
if [ "$BUILD_TARGET" = "none" ]; then
	echo "You need to specify a build target with:"
	echo "$0 win32"
	echo "$0 debian"
	exit 0
fi

# Change working directory to the directory the script is in
# http://stackoverflow.com/a/246128
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR

checkTool git "git: http://git-scm.com/"
checkTool curl "curl: http://curl.haxx.se/"
if [ $BUILD_TARGET = "win32" ]; then
	#Check if we have 7zip, needed to extract and packup a bunch of packages for windows.
	checkTool 7z "7zip: http://www.7-zip.org/"
fi

# Clean sources
rm -rf deb_dist
rm -rf win_dist

#############################
# Debian packaging
#############################

if [ $BUILD_TARGET = "debian" ]; then
	# Generate Debian source package
	python setup.py --command-packages=stdeb.command sdist_dsc \
	#--debian-version 1 \
	#--suite 'trusty' \
	#--section 'misc' \
	#--package 'horus' \
	#--depends 'python,
	#           python-serial,
	#           python-wxgtk2.8,
	#           python-opengl,
	#           python-pyglet,
	#           python-numpy,
	#           python-scipy,
	#           python-matplotlib,
	#           python-opencv,
	#           avrdude,
	#           libftdi1,
	#           v4l-utils' \
	#bdist_deb # Used to generate deb files

	# Copy postinst and postrm files
	cp -a pkg/linux/debian/postinst deb_dist/horus-${VERSION}/debian/postinst
	cp -a pkg/linux/debian/postrm deb_dist/horus-${VERSION}/debian/postrm

	# Modify changelog and control files
	cp -a pkg/linux/debian/changelog deb_dist/horus-${VERSION}/debian/changelog
	cp -a pkg/linux/debian/control deb_dist/horus-${VERSION}/debian/control

	cd deb_dist/horus-${VERSION}
	if [ $EXTRA_ARGS ]; then
		if [ $EXTRA_ARGS = "-s" ]; then
			# Build and sign Debian sources
			debuild -S -sa
		elif [ $EXTRA_ARGS = "-i" ]; then
			# Install Debian package
			dpkg-buildpackage
			sudo dpkg -i ../horus*.deb
			sudo apt-get -f install
		elif [ $EXTRA_ARGS = "-u" ]; then
			# Upload to launchpad
			debuild -S -sa
			PPA=ppa:jesus-arroyo/horus
			RELEASES="trusty utopic"
			ORIG_RELEASE=`head -1 pkg/linux/debian/changelog | sed 's/.*) \(.*\);.*/\1/'`
			for RELEASE in $RELEASES ;
			do
			  cp debian/changelog debian/changelog.backup
			  sed -i "s/${ORIG_RELEASE}/${RELEASE}/;s/bq1/bq1~${RELEASE}1/" debian/changelog
			  debuild -S -sa
			  dput -f ${PPA} ../horus_${VERSION}-bq1~${RELEASE}1_source.changes
			  mv debian/changelog.backup debian/changelog
			done
		fi
		else
			# Build and sign Debian package
			dpkg-buildpackage
		fi

	# Clean directory
	cd ../..
	rm -rf "Horus.egg-info"
fi


#############################
# Rest
#############################

#############################
# Download all needed files.
#############################

if [ $BUILD_TARGET = "win32" ]; then
	mkdir -p win_dist
	cd win_dist
	# Get portable python for windows and extract it. (Linux and Mac need to install python themselfs)
	downloadURL http://ftp.nluug.nl/languages/python/portablepython/v2.7/PortablePython_${WIN_PORTABLE_PY_VERSION}.exe
	downloadURL http://sourceforge.net/projects/pyserial/files/pyserial/2.7/pyserial-2.7.win32.exe
	downloadURL http://sourceforge.net/projects/comtypes/files/comtypes/0.6.2/comtypes-0.6.2.win32.exe
	downloadURL http://sourceforge.net/projects/pyopengl/files/PyOpenGL/3.0.1/PyOpenGL-3.0.1.win32.exe
	downloadURL https://pypi.python.org/packages/any/p/pyparsing/pyparsing-2.0.1.win32-py2.7.exe
	downloadURL http://sourceforge.net/projects/numpy/files/NumPy/1.8.1/numpy-1.8.1-win32-superpack-python2.7.exe
	downloadURL http://sourceforge.net/projects/opencvlibrary/files/opencv-win/2.4.9/opencv-2.4.9.exe
	downloadURL https://downloads.sourceforge.net/project/matplotlib/matplotlib/matplotlib-1.3.0/matplotlib-1.3.0.win32-py2.7.exe
	downloadURL http://sourceforge.net/projects/scipy/files/scipy/0.14.0/scipy-0.14.0-win32-superpack-python2.7.exe
	downloadURL http://videocapture.sourceforge.net/VideoCapture-0.9-5.zip
	mkdir -p pyglet; cd pyglet;
	downloadURL http://pyglet.googlecode.com/files/pyglet-1.1.4.msi; cd .. 
fi

#############################
# Build the packages
#############################

if [ $BUILD_TARGET = "win32" ]; then
	rm -rf ${TARGET_DIR}
	mkdir -p ${TARGET_DIR}

	rm -f log.txt

	# For windows extract portable python to include it.
	extract PortablePython_${WIN_PORTABLE_PY_VERSION}.exe \$_OUTDIR/App
	extract PortablePython_${WIN_PORTABLE_PY_VERSION}.exe \$_OUTDIR/Lib/site-packages
	extract PortablePython_${WIN_PORTABLE_PY_VERSION}.exe \$_OUTDIR/dateutil
	extract pyserial-2.7.win32.exe PURELIB
	extract comtypes-0.6.2.win32.exe PURELIB
	extract PyOpenGL-3.0.1.win32.exe PURELIB
	extract pyparsing-2.0.1.win32-py2.7.exe PURELIB
	extract numpy-1.8.1-win32-superpack-python2.7.exe numpy-1.8.1-sse2.exe
	extract numpy-1.8.1-sse2.exe PLATLIB
	extract scipy-0.14.0-win32-superpack-python2.7.exe scipy-0.14.0-sse2.exe
	extract scipy-0.14.0-sse2.exe PLATLIB
	extract matplotlib-1.3.0.win32-py2.7.exe PLATLIB
	extract opencv-2.4.9.exe opencv/build/python/2.7/x86/cv2.pyd
	extract VideoCapture-0.9-5.zip VideoCapture-0.9-5/Python27/DLLs/vidcap.pyd
	cd pyglet; extract pyglet-1.1.4.msi; cd ..

	mkdir -p ${TARGET_DIR}/python
	mv \$_OUTDIR/App/* ${TARGET_DIR}/python
	mv \$_OUTDIR/Lib/site-packages/wx* ${TARGET_DIR}/python/Lib/site-packages
	mv \$_OUTDIR/dateutil ${TARGET_DIR}/python/Lib
	mv PURELIB/serial ${TARGET_DIR}/python/Lib
	mv PURELIB/comtypes ${TARGET_DIR}/python/Lib
	mv PURELIB/OpenGL ${TARGET_DIR}/python/Lib
	mv PURELIB/pyparsing.py  ${TARGET_DIR}/python/Lib
	mv PLATLIB/numpy ${TARGET_DIR}/python/Lib
	mv PLATLIB/scipy ${TARGET_DIR}/python/Lib
	mv PLATLIB/matplotlib ${TARGET_DIR}/python/Lib
	touch PLATLIB/mpl_toolkits/__init__.py
	mv PLATLIB/mpl_toolkits ${TARGET_DIR}/python/Lib
	mv opencv/build/python/2.7/x86/cv2.pyd ${TARGET_DIR}/python/DLLs
	mv VideoCapture-0.9-5/Python27/DLLs/vidcap.pyd ${TARGET_DIR}/python/DLLs
	mv pyglet ${TARGET_DIR}/python/Lib
	
	rm -rf \$_OUTDIR
	rm -rf opencv
	rm -rf PURELIB
	rm -rf PLATLIB
	rm -rf VideoCapture-0.9-5
	rm -rf Win32
	rm -rf pyglet

	# Clean up portable python a bit, to keep the package size down.
	rm -rf ${TARGET_DIR}/python/PyScripter.*
	rm -rf ${TARGET_DIR}/python/Doc
	rm -rf ${TARGET_DIR}/python/locale
	rm -rf ${TARGET_DIR}/python/tcl
	rm -rf ${TARGET_DIR}/python/Lib/test
	rm -rf ${TARGET_DIR}/python/Lib/site-packages/wx-2.8-msw-unicode/wx/tools
	rm -rf ${TARGET_DIR}/python/Lib/site-packages/wx-2.8-msw-unicode/wx/locale
	#Remove the gle files because they require MSVCR71.dll, which is not included. We also don't need gle, so it's safe to remove it.
	rm -rf ${TARGET_DIR}/python/Lib/OpenGL/DLLS/gle*

	# Add Horus
	mkdir -p ${TARGET_DIR}/doc ${TARGET_DIR}/res ${TARGET_DIR}/src
	cp -a ../doc/* ${TARGET_DIR}/doc
	cp -a ../res/* ${TARGET_DIR}/res
	cp -a ../src/* ${TARGET_DIR}/src
	#Add horus version file
	echo $VERSION > ${TARGET_DIR}/version

	# Add script files
	cp -a ../pkg/${BUILD_TARGET}/*.bat $TARGET_DIR/

	# Package the result
	rm -rf ../pkg/win32/dist
	ln -sf `pwd`/${TARGET_DIR} ../pkg/win32/dist
	makensis -DVERSION=${VERSION} ../pkg/win32/installer.nsi
	if [ $? != 0 ]; then echo "Failed to package NSIS installer"; exit 1; fi
	mv ../pkg/win32/Horus_${VERSION}.exe .
	rm -rf ../pkg/win32/dist
fi
