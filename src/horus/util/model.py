#!/usr/bin/python
# -*- coding: utf-8 -*-
#-----------------------------------------------------------------------#
#                                                                       #
# This file is part of the Horus Project                                #
#                                                                       #
# Copyright (C) 2014-2015 Mundo Reader S.L.                             #
# Copyright (C) 2013 David Braam from Cura Project                      #
#                                                                       #
# Date: June, 2014 - April 2015                                         #
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
import time
import math

import numpy as np
np.seterr(all='ignore')

class Model(object):
	"""
	Each object has a Mesh and a 3x3 transformation matrix to rotate/scale the object.
	"""
	def __init__(self, originFilename, isPointCloud=False):
		self._originFilename = originFilename
		if originFilename is None:
			self._name = 'None'
		else:
			self._name = os.path.basename(originFilename)
		if '.' in self._name:
			self._name = os.path.splitext(self._name)[0]
		self._mesh = None
		self._position = np.array([0.0, 0.0, 0.0])
		self._matrix = np.matrix([[1,0,0],[0,1,0],[0,0,1]], np.float64)
		self._min = None
		self._max = None
		self._size = np.array([0.0, 0.0, 0.0])
		self._boundaryCircleSize = 75.0
		self._drawOffset = np.array([0.0, 0.0, 0.0])

		self._isPointCloud = isPointCloud

	def _addMesh(self):
		self._mesh = Mesh(self)
		return self._mesh

	def _postProcessAfterLoad(self):
		if len(self._mesh.vertexes) > 0:
			if not self._isPointCloud:
				self._mesh._calculateNormals()

			self._min = np.array([np.inf,np.inf,np.inf], np.float64)
			self._max = np.array([-np.inf,-np.inf,-np.inf], np.float64)
			self._boundaryCircleSize = 0

			vertexes = self._mesh.vertexes
			vmin = vertexes.min(0)
			vmax = vertexes.max(0)
			for n in xrange(0, 3):
				self._min[n] = min(vmin[n], self._min[n])
				self._max[n] = max(vmax[n], self._max[n])

			#Calculate the boundary circle
			center = vmin + (vmax - vmin) / 2.0
			boundaryCircleSize = round(np.max(np.linalg.norm(vertexes-center, axis=1)), 3)
			self._boundaryCircleSize = max(self._boundaryCircleSize, boundaryCircleSize)

			self._size = self._max - self._min
			if not self._isPointCloud:
				self._drawOffset = (self._max + self._min) / 2
				self._drawOffset[2] = self._min[2]
			self._max -= self._drawOffset
			self._min -= self._drawOffset

	def getName(self):
		return self._name
	def getOriginFilename(self):
		return self._originFilename
	def getPosition(self):
		return self._position
	def setPosition(self, newPos):
		self._position = newPos
	def getMatrix(self):
		return self._matrix

	def getMaximum(self):
		return self._max
	def getMinimum(self):
		return self._min
	def getSize(self):
		return self._size
	def getDrawOffset(self):
		return self._drawOffset
	def getBoundaryCircle(self):
		return self._boundaryCircleSize

	def isPointCloud(self):
		return self._isPointCloud

	def getScale(self):
		return np.array([
			np.linalg.norm(self._matrix[::,0].getA().flatten()),
			np.linalg.norm(self._matrix[::,1].getA().flatten()),
			np.linalg.norm(self._matrix[::,2].getA().flatten())], np.float64);


class Mesh(object):
	"""
	A mesh is a list of 3D triangles build from vertexes. Each triangle has 3 vertexes. It can be also a point cloud.
	A "VBO" can be associated with this object, which is used for rendering this object.
	"""
	def __init__(self, obj):
		self.vertexes = None
		self.colors = None
		self.normal = None
		self.vertexCount = 0
		self.vbo = None
		self._obj = obj

	def _addVertex(self, x, y, z, r=255, g=255, b=255):
		n = self.vertexCount
		self.vertexes[n], self.colors[n] = (x, y, z), (r, g, b)
		self.vertexCount += 1

	def _addFace(self, x0, y0, z0, x1, y1, z1, x2, y2, z2):
		n = self.vertexCount
		self.vertexes[n], self.vertexes[n+1], self.vertexes[n+2] = (x0, y0, z0), (x1, y1, z1), (x2, y2, z2)
		self.vertexCount += 3

	def _prepareVertexCount(self, vertexNumber):
		#Set the amount of vertex before loading data in them. This way we can create the np arrays before we fill them.
		self.vertexes = np.zeros((vertexNumber, 3), np.float32)
		self.colors = np.zeros((vertexNumber, 3), np.int32)
		self.normal = np.zeros((vertexNumber, 3), np.float32)
		self.vertexCount = 0

	def _prepareFaceCount(self, faceNumber):
		#Set the amount of faces before loading data in them. This way we can create the np arrays before we fill them.
		self.vertexes = np.zeros((faceNumber*3, 3), np.float32)
		self.normal = np.zeros((faceNumber*3, 3), np.float32)
		self.vertexCount = 0

	def _calculateNormals(self):
		#Calculate the normals
		tris = self.vertexes.reshape(self.vertexCount / 3, 3, 3)
		normals = np.cross( tris[::,1 ] - tris[::,0]  , tris[::,2 ] - tris[::,0] )
		normals /= np.linalg.norm(normals)
		n = np.concatenate((np.concatenate((normals, normals), axis=1), normals), axis=1)
		self.normal = n.reshape(self.vertexCount, 3)