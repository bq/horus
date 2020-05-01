# Horus

[![R&D](https://img.shields.io/badge/-R%26D-brightgreen.svg)](https://github.com/LibreScanner/horus)
[![License](http://img.shields.io/:license-gpl-blue.svg)](http://opensource.org/licenses/GPL-2.0)
[![Documentation Status](https://readthedocs.org/projects/horus/badge/?version=release-0.2)](http://horus.readthedocs.io/en/release-0.2/?badge=release-0.2)

Horus is a general solution for 3D laser scanning. It provides graphic user interfaces for connection, configuration, control, calibration and scanning with Open Source [Ciclop 3D Scanner](https://github.com/LibreScanner/ciclop).

This is a research project to explore the 3D laser scan with free tools. Feel free to use it for experiments, modify and adapt it to new devices and contribute new features or ideas.

This project has been developed in [Python](https://www.python.org/) language and it is distributed under [GPL v2](https://www.gnu.org/licenses/gpl-2.0.html) license.

## Installation

#### Supported

###### Current version: 0.2rc1

| Logo              | Name     | Instructions                        |
|:-----------------:|:--------:|:-----------------------------------:|
| ![][ubuntu-logo]  | Ubuntu   | [[en]](http://horus.readthedocs.io/en/release-0.2/source/installation/ubuntu.html)  [[es]](http://horus.readthedocs.io/es/release-0.2/source/installation/ubuntu.html) |
| ![][windows-logo] | Windows  |  [[en]](http://horus.readthedocs.io/en/release-0.2/source/installation/windows.html)  [[es]](http://horus.readthedocs.io/es/release-0.2/source/installation/windows.html) |
| ![][macosx-logo]  | Mac OS X |  [[en]](http://horus.readthedocs.io/en/release-0.2/source/installation/macosx.html)  [[es]](http://horus.readthedocs.io/es/release-0.2/source/installation/macosx.html) |

#### Experimental

**Horus 0.2 is not supported for the following distributions**.

However, anyone can test it and contribute to its support.

| Logo               | Name      | Instructions                          |
|:------------------:|:---------:|:-------------------------------------:|
| ![][debian-logo]   | Debian    | [[en]](doc/installation/debian.md)    |
| ![][fedora-logo]   | Fedora    | [[en]](doc/installation/fedora.md)    |

## Documentation

Here you will find the official documentation of the application:

* [User's manual](http://horus.readthedocs.io/en/release-0.2/) [[es](http://horus.readthedocs.io/es/release-0.2/)]

And also all the scientific background of the project in nice Jupyter notebooks:

* [Notebooks](http://nbviewer.jupyter.org/github/Jesus89/3DScanScience/tree/master/notebooks/)
* [Repository](https://github.com/Jesus89/3DScanScience)

## Development

Horus is an Open Source Project. Anyone has the freedom to use, modify, share and distribute this software. If you want to:
* run the source code
* make your own modifications
* contribute to the project
* build packages

follow the next instructions

#### GNU/Linux

Horus has been developed using [Ubuntu Gnome](http://ubuntugnome.org/), that is based on [Debian](https://www.debian.org/), like [Raspbian](https://www.raspbian.org/), [Mint](http://linuxmint.com/), etc. All instructions provided in this section probably work for most of these systems.

* [Ubuntu development](doc/development/ubuntu.md)

NOTE: *deb* and *exe* packages can be generated in *debian like* systems

#### Mac OS X

* [Darwin development](doc/development/darwin.md)

NOTE: *dmg* packages only can be generated in Mac OS X


More interest links are shown below:

* [Presentation](http://diwo.bq.com/en/presentacion-ciclop-horus/) [[es](http://diwo.bq.com/presentacion-ciclop-horus/)]
* [3D Design](http://diwo.bq.com/en/ciclop-released/) [[es](http://diwo.bq.com/ciclop-released/)]
* [Electronics](http://diwo.bq.com/en/zum-scan-released/) [[es](http://diwo.bq.com/zum-scan-released/)]
* [Firmware](http://diwo.bq.com/en/horus-fw-released/) [[es](http://diwo.bq.com/horus-fw-released/)]
* [Software](http://diwo.bq.com/en/horus-released/) [[es](http://diwo.bq.com/horus-released/)]
* [Product documentation](http://diwo.bq.com/en/documentation-ciclop-and-horus/) [[es](http://diwo.bq.com/documentation-ciclop-and-horus/)]
* [Google group](https://groups.google.com/forum/?hl=en#!forum/ciclop-3d-scanner)

[ubuntu-logo]: doc/images/ubuntu.png
[windows-logo]: doc/images/windows.png
[macosx-logo]: doc/images/macosx.png
[debian-logo]: doc/images/debian.png
[raspbian-logo]: doc/images/raspbian.png
[fedora-logo]: doc/images/fedora.png
