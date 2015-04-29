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

"""
STL file mesh loader.
STL is the most common file format used for 3D printing right now.
STLs come in 2 flavors.
	Binary, which is easy and quick to read.
	Ascii, which is harder to read, as can come with windows, mac and unix style newlines.
	The ascii reader has been designed so it has great compatibility with all kinds of formats or slightly broken exports from tools.

This module also contains a function to save objects as an STL file.

http://en.wikipedia.org/wiki/STL_(file_format)
"""

__author__ = "Jesús Arroyo Torrens <jesus.arroyo@bq.com>"
__license__ = "GNU General Public License v2 http://www.gnu.org/licenses/gpl.html"

import sys
import os
import struct
import numpy as np

from horus.util import model

def _loadAscii(mesh, stream):
	cnt = 0
	for lines in stream:
		for line in lines.split('\r'):
			if 'vertex' in line:
				cnt += 1
	mesh._prepareFaceCount(int(cnt) / 3)
	stream.seek(5, os.SEEK_SET)
	cnt = 0
	data = [None,None,None]
	for lines in stream:
		for line in lines.split('\r'):
			if 'vertex' in line:
				data[cnt] = line.split()[1:]
				cnt += 1
				if cnt == 3:
					mesh._addFace(float(data[0][0]), float(data[0][1]), float(data[0][2]),
								  float(data[1][0]), float(data[1][1]), float(data[1][2]),
								  float(data[2][0]), float(data[2][1]), float(data[2][2]))
					cnt = 0

def _loadBinary(mesh, stream):
	#Skip the header
	stream.read(80-5)
	count = struct.unpack('<I', stream.read(4))[0]

	dtype = np.dtype([
			('n', np.float32,(3,)),
			('v', np.float32,(9,)),
			('atttr', '<i2',(1,))])

	data = np.fromfile(stream, dtype=dtype , count=count)
	
	mesh.vertexCount = 3 * count
	n = np.zeros((mesh.vertexCount / 3, 9), np.float32)
	n[:,0:3] = n[:,3:6] = n[:,6:9] = data['n']
	mesh.normal = n.reshape(mesh.vertexCount, 3)
	mesh.vertexes = np.reshape(data['v'], (mesh.vertexCount, 3))

def loadScene(filename):
	obj = model.Model(filename)
	m = obj._addMesh()
	with open(filename, "rb") as f:
		if f.read(5).lower() == "solid":
			_loadAscii(m, f)
		else:
			_loadBinary(m, f)
		obj._postProcessAfterLoad()
		return obj