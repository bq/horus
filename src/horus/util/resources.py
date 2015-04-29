#!/usr/bin/python
# -*- coding: utf-8 -*-
#-----------------------------------------------------------------------#
#                                                                       #
# This file is part of the Horus Project                                #
#                                                                       #
# Copyright (C) 2014-2015 Mundo Reader S.L.                             #
# Copyright (C) 2013 David Braam from Cura Project                      #
#                                                                       #
# Date: June 2014                                                       #
# Author: Jesús Arroyo Torrens <jesus.arroyo@bq.com>                    #
#                                                                       #
# This program is free software: you can redistribute it and/or modify  #
# it under the terms of the GNU General Public License as published by  #
# the Free Software Foundation, either version 2 of the License, or     #
# (at your option) any later version.                                   #
#                                                                       #
# This program is distributed in the hope that it will be useful,       #
# but WITHOUT ANY WARRANTY; without even the implied warranty of        #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the          #
# GNU General Public License for more details.                          #
#                                                                       #
# You should have received a copy of the GNU General Public License     #
# along with this program. If not, see <http://www.gnu.org/licenses/>.  #
#                                                                       #
#-----------------------------------------------------------------------#

__author__ = "Jesús Arroyo Torrens <jesus.arroyo@bq.com>"
__license__ = "GNU General Public License v2 http://www.gnu.org/licenses/gpl.html"

import os
import sys
import glob
import gettext

resourceBasePath = os.path.join(os.path.dirname(__file__), "../../../res")

def getPathForResource(dir, subdir, resource_name):
    assert os.path.isdir(dir), "{p} is not a directory".format(p=dir)
    path = os.path.normpath(os.path.join(dir, subdir, resource_name))
    return path

def getPathForVersion(name='version'):
	return getPathForResource(resourceBasePath, '.', name)

def getPathForImage(name):
    return getPathForResource(resourceBasePath, 'images', name)

def getPathForFirmware(name):
    return getPathForResource(resourceBasePath, 'firmware', name)

def getPathForToolsLinux(name):
    return getPathForResource(resourceBasePath, 'tools/linux', name)

def getPathForToolsWindows(name):
    return getPathForResource(resourceBasePath, 'tools/windows', name)

def getPathForMesh(name):
    return getPathForResource(resourceBasePath, 'meshes', name)

"""def getDefaultMachineProfiles():
    path = os.path.normpath(os.path.join(resourceBasePath, 'machine_profiles', '*.ini'))
    return glob.glob(path)"""

def setupLocalization(selectedLanguage = None):
    #Default to english
    languages = ['en']

    if selectedLanguage is not None:
        for item in getLanguageOptions():
            if item[1] == selectedLanguage and item[0] is not None:
                languages = [item[0]]

    locale_path = os.path.normpath(os.path.join(resourceBasePath, 'locale'))
    translation = gettext.translation('horus', locale_path, languages, fallback=True)
    translation.install(unicode=True)

def getLanguageOptions():
    return [
        ['en', u'English'],
        ['es', u'Español'],
        ['fr', u'Français']
    ]
