# Horus
[![License](http://img.shields.io/:license-gpl-blue.svg?style=flat)](http://opensource.org/licenses/GPL-2.0) [![Build Status](https://travis-ci.org/bq/horus.svg)](https://travis-ci.org/bq/horus) [![Google Group](https://img.shields.io/badge/-Google%20Group-lightgrey.svg)](https://groups.google.com/forum/?hl=en#!forum/ciclop-3d-scanner)

## News of these fork:

01 - Ago - 2015 .- Never more change the possition of the patther on calibration! Just put in front of the camera, adjust camera and the calibration engine will do the rest.

## Whay is Horus?

Horus is a general solution for 3D scanning. It provides graphic user interfaces for connection, configuration, control, calibration and scanning. It is ready to use with Open Source [Ciclop 3D Scanner](https://github.com/bq/ciclop).

This project has been developed in [Python](https://www.python.org/) language and it is distributed under [GPL v2](https://www.gnu.org/licenses/gpl-2.0.html) license.

More interest links are shown below:

* [Presentation](http://diwo.bq.com/en/presentacion-ciclop-horus/) [[es](http://diwo.bq.com/presentacion-ciclop-horus/)]
* [3D Design](http://diwo.bq.com/en/ciclop-released/) [[es](http://diwo.bq.com/ciclop-released/)]
* [Electronics](http://diwo.bq.com/en/zum-scan-released/) [[es](http://diwo.bq.com/zum-scan-released/)]
* [Firmware](http://diwo.bq.com/en/horus-fw-released/) [[es](http://diwo.bq.com/horus-fw-released/)]
* [Software](http://diwo.bq.com/en/horus-released/) [[es](http://diwo.bq.com/horus-released/)]
* [Documentation](http://diwo.bq.com/en/documentation-ciclop-and-horus/) [[es](http://diwo.bq.com/documentation-ciclop-and-horus/)]


## What's these fork?

These fork is my interpretation of what should be, and how should it work Horus, a fantastis software for controlling laser scanner based on Ciclops by BQ.

At the moment I'm working on get the finest calibration possible, and make Horus more cominicative to try to understand it better.

Also I'm working on porting Horus to Synology NAS and Raspberry Pi. I have both working, but I have to finish the instrucctions.


## Installation

###### Last stable version: 0.1.2.4

#### Supported versions

| Logo              | Name     | Versions     | Instructions                        |
|:-----------------:|:--------:|:------------:|:-----------------------------------:|
| ![][ubuntu-logo]  | Ubuntu   | 14.04, 14.10 | [link](doc/installation/ubuntu.md)  |
| ![][windows-logo] | Windows  | 7, 8, 8.1    | [link](doc/installation/windows.md) |
| ![][macosx-logo]  | Mac OS X | 10.9, 10.10  | [link](doc/installation/macosx.md)  |

#### Experimental versions

| Logo               | Name      | Versions | Instructions                         |
|:------------------:|:---------:|:--------:|:------------------------------------:|
| ![][debian-logo]   | Debian    | 8        | [link](doc/installation/debian.md)   |
| ![][raspbian-logo] | Raspbian  | RPi2     | [link](doc/installation/raspbian.md) |
| ![][synology-logo] | Synology  | DSM 5.0  | [link](doc/installation/synology.md) |  

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


[ubuntu-logo]: doc/images/ubuntu.png
[windows-logo]: doc/images/windows.png
[macosx-logo]: doc/images/macosx.png
[debian-logo]: doc/images/debian.png
[raspbian-logo]: doc/images/raspbian.png
[synology-logo]: doc/images/synology.png
