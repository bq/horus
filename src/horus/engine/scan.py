#!/usr/bin/python
# -*- coding: utf-8 -*-
#-----------------------------------------------------------------------#
#                                                                       #
# This file is part of the Horus Project                                #
#                                                                       #
# Copyright (C) 2014-2015 Mundo Reader S.L.                             #
#                                                                       #
# Date: March, December 2014                                            #
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
import cv2
import time
import Queue
import threading
import numpy as np

from horus.engine.driver import Driver

import horus.util.error as Error
from horus.util.singleton import Singleton


class Scan:
	""" 
		Performs scanning process:

			- Multithread scanning:
				- Capture Thread
				- Process Thread: compute 3D point cloud from 2D points
	"""

	def __init__(self):
		""" """
		self.theta = 0

		self.driver = Driver.Instance()
		self.pcg = PointCloudGenerator.Instance()

		self.run = False
		self.inactive = False
		self.moveMotor = True
		self.generatePointCloud = True

		self.fastScan = False
		self.speedMotor = 200
		self.accelerationMotor = 400

		self.imgType = 'laser'
		self.imgColor  = None
		self.imgLaser  = None
		self.imgGray = None
		self.imgLine = None

		#TODO: Callbacks to Observer pattern
		self.beforeCallback = None
		self.progressCallback = None
		self.afterCallback = None

		self.imagesQueue = Queue.Queue(100)
		self.points3DQueue = Queue.Queue(1000)

	def resetTheta(self):
		self.theta = 0

	def setCallbacks(self, before, progress, after):
		self.beforeCallback = before
		self.progressCallback = progress
		self.afterCallback = after

	def start(self):
		if self.beforeCallback is not None:
			self.beforeCallback()

		self.initializeScan()
		
		self.runCapture = True
		self.runProcess = True
		self.inactive = False

		threading.Thread(target=self._captureThread).start()
		threading.Thread(target=self._processThread, args=(self.progressCallback,self.afterCallback)).start()
		
	def stop(self):
		self.runCapture = False
		self.runProcess = False
		self.inactive = False

	def _stopCapture(self):
		self.runCapture = False
		self.inactive = False

	def _stopProcess(self):
		self.runProcess = False
		self.inactive = False

	def pause(self):
		self.inactive = True

	def resume(self):
		self.inactive = False

	def setFastScan(self, value):
		self.fastScan = value

	def setSpeedMotor(self, value):
		self.speedMotor = value

	def setAccelerationMotor(self, value):
		self.accelerationMotor = value

	def setImageType(self, imgType):
		self.imgType = imgType

	def getImage(self, source=None):
		img = { 'Laser' : self.imgLaser,
				'Gray' : self.imgGray,
				'Line' : self.imgLine,
				'Color' : self.imgColor
			  }[self.imgType]
		if source is not None:
			img = source
		
		if img is not None:
			if self.pcg.viewROI:
				img = self.roi2DVisualization(img)
		return img

	def roi2DVisualization(self, img):
		self.pcg.calculateCenter()
		#params:
		thickness=6
		thickness_hiden=1

		center_up_u=self.pcg.no_trimmed_umin+(self.pcg.no_trimmed_umax- self.pcg.no_trimmed_umin)/2
		center_up_v=self.pcg.upper_vmin+(self.pcg.upper_vmax-self.pcg.upper_vmin)/2
		center_down_u=self.pcg.no_trimmed_umin+(self.pcg.no_trimmed_umax- self.pcg.no_trimmed_umin)/2
		center_down_v= self.pcg.lower_vmax+(self.pcg.lower_vmin-self.pcg.lower_vmax)/2
		axes_up=((self.pcg.no_trimmed_umax- self.pcg.no_trimmed_umin)/2, ((self.pcg.upper_vmax-self.pcg.upper_vmin)/2))
		axes_down=((self.pcg.no_trimmed_umax- self.pcg.no_trimmed_umin)/2, ((self.pcg.lower_vmin-self.pcg.lower_vmax)/2))

		img = img.copy()
		#upper ellipse
		if (center_up_v<self.pcg.cy):
			cv2.ellipse(img, (center_up_u, center_up_v), axes_up, 0, 180, 360, (0, 100, 200), thickness)
			cv2.ellipse(img, (center_up_u, center_up_v), axes_up, 0, 0, 180, (0, 100, 200), thickness_hiden)
		else:
			cv2.ellipse(img, (center_up_u, center_up_v), axes_up, 0, 180, 360, (0, 100, 200), thickness)
			cv2.ellipse(img, (center_up_u, center_up_v), axes_up, 0, 0, 180, (0, 100, 200), thickness)

		#lower ellipse
		cv2.ellipse(img, (center_down_u, center_down_v), axes_down, 0, 180, 360, (0, 100, 200), thickness_hiden)
		cv2.ellipse(img, (center_down_u, center_down_v), axes_down, 0, 0, 180, (0, 100, 200), thickness)

		#cylinder lines

		cv2.line(img, (self.pcg.no_trimmed_umin, center_up_v), (self.pcg.no_trimmed_umin, center_down_v), (0, 100, 200),thickness)
		cv2.line(img, (self.pcg.no_trimmed_umax, center_up_v), (self.pcg.no_trimmed_umax, center_down_v), (0, 100, 200),thickness)

		#view center
		if axes_up[0]<=0 or axes_up[1] <=0:
			axes_up_center=(20,1)
			axes_down_center=(20,1)
		else:
			axes_up_center=(20,axes_up[1]*20/axes_up[0])
			axes_down_center=(20,axes_down[1]*20/axes_down[0])
		#upper center
		cv2.ellipse(img, (self.pcg.center_u, min(center_up_v, self.pcg.center_v) ), axes_up_center, 0, 0, 360, (0, 70, 120), -1)
		#lower center
		cv2.ellipse(img, (self.pcg.center_u, self.pcg.center_v), axes_down_center, 0, 0, 360, (0, 70, 120), -1)

		return img

	def applyROIMask(self, image):
		mask = np.zeros(image.shape,np.uint8)
		mask[self.pcg.vmin:self.pcg.vmax,
			 self.pcg.umin:self.pcg.umax] = image[self.pcg.vmin:self.pcg.vmax,
			 									  self.pcg.umin:self.pcg.umax]

		return mask

	def initializeScan(self):
		self.pcg.resetTheta()
		self.resetTheta()

		self.points3DQueue.queue.clear()
		self.imagesQueue.queue.clear()

		#-- Setup board
		if self.moveMotor:
			self.driver.board.enableMotor()
			self.driver.board.setSpeedMotor(self.speedMotor)
			self.driver.board.setAccelerationMotor(self.setAccelerationMotor)
			time.sleep(0.2)
		else:
			self.driver.board.disableMotor()

		self.driver.board.setLeftLaserOff()
		self.driver.board.setRightLaserOff()

		#-- Setup camera
		self.driver.camera.capture.read()

	#-- Threads

	def _captureThread(self):
		""""""
		pass

	def _processThread(self, progressCallback, afterCallback):
		""""""
		ret = False

		if progressCallback is not None:
			progressCallback(0)

		while self.runProcess:
			if not self.inactive:
				angle = abs(self.pcg.theta * 180.0 / np.pi)
				if progressCallback is not None and abs(self.pcg.degrees) > 0:
					progressCallback(abs(angle/self.pcg.degrees), abs(360.0/self.pcg.degrees))
				if angle <= 360.0:
					if not self.imagesQueue.empty():
						imagesQueueItem = self.imagesQueue.get(timeout=0.1)
						self.imagesQueue.task_done();

						begin = time.time()

						updateTheta=True
						#-- Compute 2D points from images
						points2D, colors = self.compute2DPoints(imagesQueueItem)
						#-- Put 2D points into the queue
						if imagesQueueItem[0]=='right':
							laser=False
						elif imagesQueueItem[0]=='left':
							laser=True
						elif imagesQueueItem[0]=='both_right':
							updateTheta=False
							laser=False
						elif imagesQueueItem[0]=='both_left':
							updateTheta=True
							laser=True

						#-- Compute 3D points
						points3D, colors = self.pcg.compute3DPoints(points2D, colors, laser, updateTheta)

						if points3D is not None and colors is not None:
							if self.generatePointCloud:
								#-- Put point cloud into the queue
								self.points3DQueue.put((points3D, colors))

						end = time.time()

						print "Process: {0} ms".format(int((end-begin)*1000))

						#-- Free objects
						del imagesQueueItem
						del colors
						del points2D
						del points3D
				else:
					if self.generatePointCloud:
						ret = True
						self._stopProcess()
			else:
				time.sleep(0.1)

		if ret:
			response = (True, None)
		else:
			response = (False, Error.ScanError)

		self.imgLaser = None
		self.imgGray = None
		self.imgLine = None
		self.imgColor = None

		self.points3DQueue.queue.clear()
		self.imagesQueue.queue.clear()

		if afterCallback is not None:
			afterCallback(response)

	def getPointCloudIncrement(self):
		""" """
		if not self.points3DQueue.empty():
			pc = self.points3DQueue.get_nowait()
			if pc is not None:
				self.points3DQueue.task_done()
			return pc
		else:
			return None


@Singleton
class SimpleScan(Scan):
	""" 
		Capture geometry

			- Image processing
			- Red line laser detection
	"""

	def _captureThread(self):
		""""""
		if os.name == 'nt':
			flush_both = 3
			flush_single = 1
		else:
			flush_both = 1
			flush_single = 0

		#-- Switch off the lasers
		self.driver.board.setLeftLaserOff()
		self.driver.board.setRightLaserOff()

		if self.pcg.useLeftLaser and not self.pcg.useRightLaser:
			self.driver.board.setLeftLaserOn()

		if not self.pcg.useLeftLaser and self.pcg.useRightLaser:
			self.driver.board.setRightLaserOn()

		while self.runCapture:
			if not self.inactive:
				if abs(self.theta * 180.0 / np.pi) <= 360.0:
					begin = time.time()

					#-- Left laser
					if self.pcg.useLeftLaser and not self.pcg.useRightLaser:
						imageLaserLeft = self.driver.camera.captureImage(flush=True, flushValue=flush_single)

					#-- Right laser
					if not self.pcg.useLeftLaser and self.pcg.useRightLaser:
						imageLaserRight = self.driver.camera.captureImage(flush=True, flushValue=flush_single)

					##-- Both laser
					if self.pcg.useLeftLaser and self.pcg.useRightLaser:
						self.driver.board.setLeftLaserOn()
						self.driver.board.setRightLaserOff()
						imgLaserLeft = self.driver.camera.captureImage(flush=True, flushValue=flush_both)

						self.driver.board.setRightLaserOn()
						self.driver.board.setLeftLaserOff()
						imgLaserRight = self.driver.camera.captureImage(flush=True, flushValue=flush_both)
					
					print "> {0} deg <".format(self.theta * 180.0 / np.pi)
					self.theta -= self.pcg.degrees * self.pcg.rad

					#-- Move motor
					if self.moveMotor:
						self.driver.board.setRelativePosition(self.pcg.degrees)
						self.driver.board.moveMotor()
					else:
						time.sleep(0.05)

					end = time.time()
					print "Capture: {0} ms".format(int((end-begin)*1000))
					
					if self.pcg.useLeftLaser and not self.pcg.useRightLaser:
						self.imagesQueue.put(('left',imageLaserLeft))
						del imageLaserLeft

					if not self.pcg.useLeftLaser and self.pcg.useRightLaser:
						self.imagesQueue.put(('right',imageLaserRight))
						del imageLaserRight

					if self.pcg.useLeftLaser and self.pcg.useRightLaser:
						self.imagesQueue.put(('both_left',imgLaserLeft))
						self.imagesQueue.put(('both_right',imgLaserRight))
						del imgLaserLeft
						del imgLaserRight
				else:
					if self.generatePointCloud:
						self._stopCapture()
					break
			else:
				time.sleep(0.1)

		#-- Disable board
		self.driver.board.setLeftLaserOff()
		self.driver.board.setRightLaserOff()
		self.driver.board.disableMotor()

	def setColor(self, value):
		self.color = value

	def setUseThreshold(self, enable):
		self.thresholdEnable = enable

	def setThresholdValue(self, value):
		self.thresholdValue = value

	def compute2DPoints(self, image_info):
		image=image_info[1]
		self.imgLaser = image
		tempColor = np.ones_like(image)
		tempColor[:,:,0] *= self.color[0]
		tempColor[:,:,1] *= self.color[1]
		tempColor[:,:,2] *= self.color[2]
		self.imgColor = tempColor

		#-- Convert tp Y Cr Cb format
		y,cr,cb = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2YCR_CB))

		#-- Get Cr
		image = cr

		#-- Apply ROI mask
		image = self.applyROIMask(image)

		#-- Threshold image
		if self.thresholdEnable:
			image = cv2.threshold(image, self.thresholdValue, 255.0, cv2.THRESH_TOZERO)[1]

		#-- Peak detection: center of mass
		h, w = image.shape
		W = np.array((np.matrix(np.linspace(0,w-1,w)).T*np.matrix(np.ones(h))).T)
		s = image.sum(axis=1)
		v = np.where(s > 0)[0]
		u = (W*image).sum(axis=1)[v] / s[v]
		
		tempLine = np.zeros_like(self.imgLaser)
		tempLine[v,u.astype(int)] = 255
		self.imgLine = tempLine
		self.imgGray = cv2.merge((image, image, image))

		colors = np.array(np.matrix(np.ones(len(u))).T*np.matrix(self.color)).T

		return (u, v), colors


@Singleton
class TextureScan(Scan):
	""" 
		Capture geometry and texture
	"""

	def _captureThread(self):
		""""""
		imgRaw = None
		imgLaserLeft = None
		imgLaserRight = None

		#-- Switch off the lasers
		self.driver.board.setLeftLaserOff()
		self.driver.board.setRightLaserOff()

		if os.name == 'nt':
			flush = 3
		else:
			flush = 1

		while self.runCapture:
			if not self.inactive:
				if abs(self.theta * 180.0 / np.pi) <= 360.0:
					begin = time.time()

					if self.fastScan: #-- FAST METHOD (only for linux)

						if self.driver.camera.isConnected:
							self.driver.camerareading = True
							#-- Left laser
							if self.pcg.useLeftLaser and not self.pcg.useRightLaser:
								self.driver.board.setLeftLaserOff()
								imgLaserLeft = self.driver.camera.capture.read()[1]
								imgRaw = self.driver.camera.capture.read()[1]
								self.driver.board.setLeftLaserOn()
								self.driver.camerareading = False

								if imgRaw is None or imgLaserLeft is None:
									self.driver.camera._fail()
								else:
									imgRaw = cv2.transpose(imgRaw)
									imgRaw = cv2.flip(imgRaw, 1)
									imgRaw = cv2.cvtColor(imgRaw, cv2.COLOR_BGR2RGB)

									imgLaserRight = None

									imgLaserLeft = cv2.transpose(imgLaserLeft)
									imgLaserLeft = cv2.flip(imgLaserLeft, 1)
									imgLaserLeft = cv2.cvtColor(imgLaserLeft, cv2.COLOR_BGR2RGB)

							#-- Right laser
							if not self.pcg.useLeftLaser and self.pcg.useRightLaser:
								self.driver.board.setRightLaserOff()
								imgLaserRight = self.driver.camera.capture.read()[1]
								imgRaw = self.driver.camera.capture.read()[1]
								self.driver.board.setRightLaserOn()
								self.driver.camerareading = False

								if imgRaw is None or imgLaserRight is None:
									self.driver.camera._fail()
								else:
									imgRaw = cv2.transpose(imgRaw)
									imgRaw = cv2.flip(imgRaw, 1)
									imgRaw = cv2.cvtColor(imgRaw, cv2.COLOR_BGR2RGB)

									imgLaserLeft = None

									imgLaserRight = cv2.transpose(imgLaserRight)
									imgLaserRight = cv2.flip(imgLaserRight, 1)
									imgLaserRight = cv2.cvtColor(imgLaserRight, cv2.COLOR_BGR2RGB)

							##-- Both laser
							if self.pcg.useLeftLaser and self.pcg.useRightLaser:
								imgRaw = self.driver.camera.capture.read()[1]
								self.driver.board.setLeftLaserOn()
								imgLaserLeft = self.driver.camera.capture.read()[1]
								self.driver.board.setLeftLaserOff()
								self.driver.board.setRightLaserOn()
								imgLaserRight = self.driver.camera.capture.read()[1]
								self.driver.board.setRightLaserOff()
								imgRaw = self.driver.camera.capture.read()[1]
								self.driver.camerareading = False

								if imgRaw is None or imgLaserLeft is None or imgLaserRight is None:
									self.driver.camera._fail()
								else:
									imgRaw = cv2.transpose(imgRaw)
									imgRaw = cv2.flip(imgRaw, 1)
									imgRaw = cv2.cvtColor(imgRaw, cv2.COLOR_BGR2RGB)

									imgLaserLeft = cv2.transpose(imgLaserLeft)
									imgLaserLeft = cv2.flip(imgLaserLeft, 1)
									imgLaserLeft = cv2.cvtColor(imgLaserLeft, cv2.COLOR_BGR2RGB)

									imgLaserRight = cv2.transpose(imgLaserRight)
									imgLaserRight = cv2.flip(imgLaserRight, 1)
									imgLaserRight = cv2.cvtColor(imgLaserRight, cv2.COLOR_BGR2RGB)

					else: #-- SLOW METHOD

						#-- Switch off laser
						if self.pcg.useLeftLaser:
							self.driver.board.setLeftLaserOff()
						if self.pcg.useRightLaser:
							self.driver.board.setRightLaserOff()

						#-- Capture images
						imgRaw = self.driver.camera.captureImage(flush=True, flushValue=flush)

						if self.pcg.useLeftLaser:
							self.driver.board.setLeftLaserOn()
							self.driver.board.setRightLaserOff()
							imgLaserLeft = self.driver.camera.captureImage(flush=True, flushValue=flush)
						else:
							imgLaserLeft = None

						if self.pcg.useRightLaser:
							self.driver.board.setRightLaserOn()
							self.driver.board.setLeftLaserOff()
							imgLaserRight = self.driver.camera.captureImage(flush=True, flushValue=flush)
						else:
							imgLaserRight = None

					print "> {0} deg <".format(self.theta * 180.0 / np.pi)
					self.theta -= self.pcg.degrees * self.pcg.rad

					#-- Move motor
					if self.moveMotor:
						self.driver.board.setRelativePosition(self.pcg.degrees)
						self.driver.board.moveMotor()
					else:
						time.sleep(0.05)

					end = time.time()
					print "Capture: {0} ms".format(int((end-begin)*1000))

					if self.pcg.useLeftLaser and not self.pcg.useRightLaser:
						if imgRaw is not None and imgLaserLeft is not None:
							self.imagesQueue.put(('left',imgRaw,imgLaserLeft))
							del imgLaserLeft

					elif self.pcg.useRightLaser and not self.pcg.useLeftLaser:
						if imgRaw is not None and imgLaserRight is not None:
							self.imagesQueue.put(('right',imgRaw,imgLaserRight))
							del imgLaserRight

					elif self.pcg.useRightLaser and self.pcg.useLeftLaser:
						if imgRaw is not None and imgLaserLeft is not None and imgLaserRight is not None:
							self.imagesQueue.put(('both_left',imgRaw,imgLaserLeft))
							self.imagesQueue.put(('both_right',imgRaw,imgLaserRight))
							del imgLaserLeft
							del imgLaserRight
					del imgRaw
				else:
					if self.generatePointCloud:
						self._stopCapture()
					break
			else:
				time.sleep(0.1)

		#-- Disable board
		self.driver.board.setLeftLaserOff()
		self.driver.board.setRightLaserOff()
		self.driver.board.disableMotor()

	def setUseOpen(self, enable):
		self.openEnable = enable

	def setOpenValue(self, value):
		self.openValue = value

	def setUseThreshold(self, enable):
		self.thresholdEnable = enable

	def setThresholdValue(self, value):
		self.thresholdValue = value

	def compute2DPoints(self, images):
		imageColor=images[1]
		imageLaser=images[2]
		self.imgLaser = imageLaser
		self.imgColor = imageColor

		"""#-- Use R channel
		r1,g1,b1 = cv2.split(imageColor)
		r2,g1,b2 = cv2.split(imageLaser)

		image = cv2.subtract(r2, r1)"""

		#-- Use Cr channel
		y1,cr1,cb1 = cv2.split(cv2.cvtColor(imageColor, cv2.COLOR_RGB2YCR_CB))
		y2,cr2,cb2 = cv2.split(cv2.cvtColor(imageLaser, cv2.COLOR_RGB2YCR_CB))
		
		image = cv2.subtract(cr2, cr1)

		#-- Apply ROI mask
		image = self.applyROIMask(image)

		#-- Open image
		if self.openEnable:
			kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(self.openValue,self.openValue))
			image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

		#-- Threshold image
		if self.thresholdEnable:
			image = cv2.threshold(image, self.thresholdValue, 255.0, cv2.THRESH_TOZERO)[1]

		#-- Peak detection: center of mass
		h, w = image.shape
		W = np.array((np.matrix(np.linspace(0,w-1,w)).T*np.matrix(np.ones(h))).T)
		s = image.sum(axis=1)
		v = np.where(s > 0)[0]
		u = (W*image).sum(axis=1)[v] / s[v]

		tempLine = np.zeros_like(imageLaser)
		tempLine[v,u.astype(int)] = 255.0
		self.imgLine = tempLine
		self.imgGray = cv2.merge((image, image, image))

		colors = imageColor[v,u.astype(int)].T

		return (u, v), colors


@Singleton
class PointCloudGenerator:
	""" 
		Contains all scanning algorithms:

			- Point cloud generation and filtering
	"""
	def __init__(self):
		self.theta = 0
		self.driver = Driver.Instance()

		self.rad = np.pi / 180.0

		self.umin = 0
		self.umax = 960
		self.vmin = 0
		self.vmax = 1280

		self.circleResolution = 30
		self.circleArray = np.array([[np.cos(i*2*np.pi/self.circleResolution) for i in xrange(self.circleResolution)],
									 [np.sin(i*2*np.pi/self.circleResolution) for i in xrange(self.circleResolution)],
									 np.zeros(self.circleResolution)])



	def setViewROI(self, value):
		self.viewROI = value

	def setViewCenter(self, value):
		self.viewCenter = value

	def setROIDiameter(self, value):
		self.roiRadius = value / 2.0
		self.calculateROI()

	def setROIHeight(self, value):
		self.roiHeight = value
		self.calculateROI()


	def setDegrees(self, degrees):
		self.degrees = degrees

	def setResolution(self, width, height):
		self.width = width
		self.height = height
		self.W = np.ones((height,width))*np.linspace(0,width-1,width)

	def setUseLaser(self, useLeft, useRight):
		self.useLeftLaser = useLeft
		self.useRightLaser = useRight

	def setLaserAngles(self, angleLeft, angleRight):
		self.alphaLeft = angleLeft * self.rad
		self.alphaRight = angleRight * self.rad

	def setCameraIntrinsics(self, cameraMatrix, distortionVector):
		self.fx = cameraMatrix[0][0]
		self.fy = cameraMatrix[1][1]
		self.cx = cameraMatrix[0][2]
		self.cy = cameraMatrix[1][2]

	def setLaserTriangulation(self, dL, nL, dR, nR):
		self.dL = dL
		self.nL = nL
		self.dR = dR
		self.nR = nR

	def setPlatformExtrinsics(self, rotationMatrix, translationVector):
		self.rotationMatrix = rotationMatrix
		self.translationVector = translationVector

	def resetTheta(self):
		self.theta = 0

	def calculateROI(self):
		if hasattr(self, 'roiRadius') and \
		   hasattr(self, 'roiHeight') and \
		   hasattr(self, 'rotationMatrix') and \
		   hasattr(self, 'translationVector') and \
		   hasattr(self, 'fx') and hasattr(self, 'fy') and \
		   hasattr(self, 'cx') and hasattr(self, 'cy') and \
		   hasattr(self, 'width') and hasattr(self, 'height'):

			#-- Platform system
			bottom = np.matrix(self.roiRadius * self.circleArray)
			top = bottom + np.matrix([0,0,self.roiHeight]).T
			data = np.concatenate((bottom, top), axis=1)

			#-- Camera system
			data =  self.rotationMatrix * data + np.matrix(self.translationVector).T

			#-- Video system
			u = self.fx * data[0] / data[2] + self.cx
			v = self.fy * data[1] / data[2] + self.cy

			umin = int(round(np.min(u)))
			umax = int(round(np.max(u)))
			vmin = int(round(np.min(v)))
			vmax = int(round(np.max(v)))

			#visualization : 
			v_=np.array(v.T)
			#lower cylinder base
			a=v_[:(len(v_)/2)]
			#upper cylinder base
			b=v_[(len(v_)/2):]
			self.lower_vmin=int(round(np.max(a)))
			self.lower_vmax=int(round(np.min(a)))
			self.upper_vmin=int(round(np.min(b)))
			self.upper_vmax=int(round(np.max(b)))

			self.no_trimmed_umin=umin
			self.no_trimmed_umax = int(round(np.max(u)))
			self.no_trimmed_vmin = int(round(np.min(v)))
			self.no_trimmed_vmax = int(round(np.max(v)))

			self.umin = max(umin, 0)
			self.umax = min(umax, self.width)
			self.vmin = max(vmin, 0)
			self.vmax = min(vmax, self.height)

	def calculateCenter(self):
		#-- Platform system
		bottom = np.matrix(0* self.circleArray)
		top = bottom + np.matrix([0,0,0]).T
		data = np.concatenate((bottom, top), axis=1)

		#-- Camera system
		data =  self.rotationMatrix * data + np.matrix(self.translationVector).T

		#-- Video system
		u = self.fx * data[0] / data[2] + self.cx
		v = self.fy * data[1] / data[2] + self.cy

		umin = int(round(np.min(u)))
		umax = int(round(np.max(u)))
		vmin = int(round(np.min(v)))
		vmax = int(round(np.max(v)))

		self.center_u=umin+(umax-umin)/2
		self.center_v=vmin+(vmax-vmin)/2

	def pointCloudGeneration(self, points2D, leftLaser=True):
		""" """
		u, v = points2D

		#-- Obtaining point cloud in camera coordinates
		if leftLaser:
			d = self.dL
			n = self.nL
		else:
			d = self.dR
			n = self.nR

		# x = np.concatenate(((u-self.cx)/self.fx, (v-self.cy)/self.fy, np.ones(len(u)))).reshape(3,len(u))

		x = np.concatenate(((u-self.cx)/self.fx, (v-self.cy)/self.fy, np.ones(len(u)))).reshape(3,len(u))

		Xc = d/np.dot(n,x)*x

		#-- Move point cloud to world coordinates
		R = np.matrix(self.rotationMatrix).T
		t = np.matrix(self.translationVector).T

		Xwo = R*Xc - R*t

		#-- Rotate point cloud
		c = np.cos(self.theta)
		s = np.sin(self.theta)
		Rz = np.matrix([[c, -s, 0],[s, c, 0], [0, 0, 1]])
		Xw = Rz * Xwo

		#-- Return result
		if Xw.size > 0:
			return np.array(Xw)
		else:
			return None

	def pointCloudFilter(self, points, colors):
		""" """
		#-- Point Cloud Filter
		rho = np.sqrt(points[0,:]**2 + points[1,:]**2)
		z = points[2,:]

		idx = np.where((z >= 0) &
					   (z <= self.roiHeight) &
					   (rho >= -self.roiRadius) &
					   (rho <= self.roiRadius))[0]

		return points[:,idx], colors[:,idx]

	def compute3DPoints(self, points2D, colors, leftLaser, updateTheta):
		""" """
		#-- Point Cloud Generation
		points3D = self.pointCloudGeneration(points2D, leftLaser)

		if points3D is not None:
			#-- Point Cloud Filter
			points3D, colors = self.pointCloudFilter(points3D, colors)

		if updateTheta: 
			#-- Update Theta
			self.theta -= self.degrees * self.rad

		return points3D, colors
