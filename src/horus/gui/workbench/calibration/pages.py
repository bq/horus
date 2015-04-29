#!/usr/bin/python
# -*- coding: utf-8 -*-
#-----------------------------------------------------------------------#
#                                                                       #
# This file is part of the Horus Project                                #
#                                                                       #
# Copyright (C) 2014-2015 Mundo Reader S.L.                             #
#                                                                       #
# Date: August, November 2014                                           #
# Author: Jesús Arroyo Torrens <jesus.arroyo@bq.com>   	                #
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

import cv2
import wx._core
import numpy as np

import random
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg

import horus.util.error as Error
from horus.util import profile, resources

from horus.gui.util.imageView import ImageView, VideoView

from horus.gui.workbench.calibration.page import Page

from horus.engine.driver import Driver
from horus.engine import calibration

class CameraIntrinsicsMainPage(Page):

	def __init__(self, parent, afterCancelCallback=None, afterCalibrationCallback=None):
		Page.__init__(self, parent,
							title=_("Camera Intrinsics"),
							subTitle=_("Press space bar to perform captures"),
							left=_("Cancel"),
							right=_("Calibrate"),
							buttonLeftCallback=self.onCancel,
							buttonRightCallback=self.onCalibrate,
							panelOrientation=wx.HORIZONTAL,
							viewProgress=True)

		self.driver = Driver.Instance()
		self.cameraIntrinsics = calibration.CameraIntrinsics.Instance()

		self.afterCancelCallback = afterCancelCallback
		self.afterCalibrationCallback = afterCalibrationCallback

		#-- Video View
		self.videoView = VideoView(self._panel, self.getFrame, 50)
		self.videoView.SetBackgroundColour(wx.BLACK)

		#-- Image Grid Panel
		self.imageGridPanel = wx.Panel(self._panel)
		self.rows, self.columns = 2, 6
		self.panelGrid = []
		self.gridSizer = wx.GridSizer(self.rows, self.columns, 3, 3)
		for panel in xrange(self.rows*self.columns):
			self.panelGrid.append(ImageView(self.imageGridPanel))
			self.panelGrid[panel].Bind(wx.EVT_KEY_DOWN, self.onKeyPress)
			self.panelGrid[panel].SetBackgroundColour((221, 221, 221))
			self.panelGrid[panel].setImage(wx.Image(resources.getPathForImage("void.png")))
			self.gridSizer.Add(self.panelGrid[panel], 0, wx.ALL|wx.EXPAND)
		self.imageGridPanel.SetSizer(self.gridSizer)

		#-- Layout
		self.addToPanel(self.videoView, 1)
		self.addToPanel(self.imageGridPanel, 3)

		#-- Events
		self.Bind(wx.EVT_SHOW, self.onShow)
		self.videoView.Bind(wx.EVT_KEY_DOWN, self.onKeyPress)
		self.imageGridPanel.Bind(wx.EVT_KEY_DOWN, self.onKeyPress)
		
		self.videoView.SetFocus()
		self.Layout()

	def initialize(self):
		self._rightButton.Hide()
		self.subTitleText.SetLabel(_("Press space bar to perform captures"))
		self.currentGrid = 0
		self.gauge.SetValue(0)
		for panel in xrange(self.rows*self.columns):
			self.panelGrid[panel].SetBackgroundColour((221, 221, 221))
			self.panelGrid[panel].setImage(wx.Image(resources.getPathForImage("void.png")))

	def onShow(self, event):
		if event.GetShow():
			self.gauge.SetValue(0)
			self.videoView.play()
			calibration.CameraIntrinsics.Instance().clearImageStack()
			self.GetParent().Layout()
			self.Layout()
		else:
			try:
				self.initialize()
				self.videoView.stop()
			except:
				pass

	def getFrame(self):
		frame = self.driver.camera.captureImage(mirror=True)
		if frame is not None:
			retval, frame = self.cameraIntrinsics.detectChessboard(frame)
			if retval:
				self.videoView.SetBackgroundColour((45,178,0))
			else:
				self.videoView.SetBackgroundColour((217,0,0))
		return frame

	def onKeyPress(self, event):
		if event.GetKeyCode() == 32: #-- spacebar
			if self.driver.isConnected:
				self.videoView.pause()
				frame = self.driver.camera.captureImage(mirror=False, flush=True)
				if frame is not None:
					retval, frame = self.cameraIntrinsics.detectChessboard(frame, capture=True)
					frame = cv2.flip(frame, 1) #-- Mirror
					self.addFrameToGrid(retval, frame)
					self.gauge.SetValue(7*self.currentGrid)
				self.videoView.play()

	def addFrameToGrid(self, retval, image):
		if self.currentGrid < (self.columns*self.rows):
			if retval:
				self.panelGrid[self.currentGrid].setFrame(image)
				self.panelGrid[self.currentGrid].SetBackgroundColour((45,178,0))
				self.currentGrid += 1
			else:
				self.panelGrid[self.currentGrid].setFrame(image)
				self.panelGrid[self.currentGrid].SetBackgroundColour((217,0,0))

		if self.currentGrid is (self.columns*self.rows):
			self.subTitleText.SetLabel(_("Press Calibrate to continue"))
			self.buttonRightCallback()
			# self._rightButton.Enable()

	def onCalibrate(self):
		self.cameraIntrinsics.setCallbacks(self.beforeCalibration,
										   lambda p: wx.CallAfter(self.progressCalibration,p),
										   lambda r: wx.CallAfter(self.afterCalibration,r))
		self.cameraIntrinsics.start()

	def beforeCalibration(self):
		self.videoView.pause()
		self._rightButton.Disable()
		self.gauge.SetValue(95)
		if not hasattr(self, 'waitCursor'):
			self.waitCursor = wx.BusyCursor()

	def progressCalibration(self, progress):
		self.gauge.SetValue(max(95, progress))

	def afterCalibration(self, result):
		self._rightButton.Enable()
		if self.afterCalibrationCallback is not None:
			self.afterCalibrationCallback(result)
		if hasattr(self, 'waitCursor'):
			del self.waitCursor

	def onCancel(self):
		boardUnplugCallback = self.driver.board.unplugCallback
		cameraUnplugCallback = self.driver.camera.unplugCallback
		self.driver.board.setUnplugCallback(None)
		self.driver.camera.setUnplugCallback(None)
		if not hasattr(self, 'waitCursor'):
			self.waitCursor = wx.BusyCursor()
		self.onCalibration = False
		self.cameraIntrinsics.cancel()
		if self.afterCancelCallback is not None:
			self.afterCancelCallback()
		del self.waitCursor
		self.driver.board.setUnplugCallback(boardUnplugCallback)
		self.driver.camera.setUnplugCallback(cameraUnplugCallback)


class CameraIntrinsicsResultPage(Page):

	def __init__(self, parent, buttonRejectCallback=None, buttonAcceptCallback=None):
		Page.__init__(self, parent,
							title=_("Camera Intrinsics"),
							left=_("Reject"),
							right=_("Accept"),
							buttonLeftCallback=buttonRejectCallback,
							buttonRightCallback=buttonAcceptCallback,
							panelOrientation=wx.HORIZONTAL)

		self.cameraIntrinsics = calibration.CameraIntrinsics.Instance()

		#-- 3D Plot Panel
		self.plotPanel = CameraIntrinsics3DPlot(self._panel)

		#-- Layout
		self.addToPanel(self.plotPanel, 2)

		#-- Events
		self.Bind(wx.EVT_SHOW, self.onShow)

	def onShow(self, event):
		if event.GetShow():
			self.GetParent().Layout()
			self.Layout()

	def processCalibration(self, response):
		self.plotPanel.Hide()
		self.plotPanel.clear()
		ret, result = response

		if ret:
			mtx, dist, rvecs, tvecs = result
			self.GetParent().GetParent().controls.panels['camera_intrinsics_panel'].setParameters((mtx, dist))
			self.plotPanel.add(rvecs, tvecs)
			self.plotPanel.Show()
			self.Layout()
		else:
			if result == Error.CalibrationError:
				dlg = wx.MessageDialog(self, _("Camera Intrinsics Calibration has failed. Please try again."), _(result), wx.OK|wx.ICON_ERROR)
				dlg.ShowModal()
				dlg.Destroy()


class CameraIntrinsics3DPlot(wx.Panel):
	def __init__(self, parent):
		wx.Panel.__init__(self, parent)

		self.initialize()

	def initialize(self):
		self.fig = Figure(facecolor=(0.7490196,0.7490196,0.7490196,1), tight_layout=True)
		self.canvas = FigureCanvasWxAgg(self, -1, self.fig)
		self.canvas.SetExtraStyle(wx.EXPAND)

		self.ax = self.fig.gca(projection='3d', axisbg=(0.7490196,0.7490196,0.7490196,1))

		# Parameters of the pattern
		self.rows = profile.getProfileSettingInteger('pattern_rows')
		self.columns = profile.getProfileSettingInteger('pattern_columns')
		self.squareWidth = profile.getProfileSettingInteger('square_width')
		
		self.printCanvas()

		self.Bind(wx.EVT_SIZE, self.onSize)
		self.Layout()

	def onSize(self,event):
		self.canvas.SetClientSize(self.GetClientSize())
		self.Layout()
		event.Skip()

	def printCanvas(self):
		self.ax.plot([0,50], [0,0], [0,0], linewidth=2.0, color='red')
		self.ax.plot([0,0], [0,0], [0,50], linewidth=2.0, color='green')
		self.ax.plot([0,0], [0,50], [0,0], linewidth=2.0, color='blue')

		self.ax.set_xlabel('X')
		self.ax.set_ylabel('Z')
		self.ax.set_zlabel('Y')
		self.ax.set_xlim(-150, 150)
		self.ax.set_ylim(0, 500)
		self.ax.set_zlim(-150, 150)
		self.ax.invert_xaxis()
		self.ax.invert_yaxis()
		self.ax.invert_zaxis()

	def add(self, rvecs, tvecs):
		h = self.columns * self.squareWidth 
		w = self.rows * self.squareWidth

		p = np.array([[0,0,0],[w,0,0],[w,h,0],[0,h,0],[0,0,0]])
		n = np.array([[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]])

		c = np.array([[30,0,0],[0,30,0],[0,0,30]])

		for ind, transvector in enumerate(rvecs):

			R = cv2.Rodrigues(transvector)[0]
			t = tvecs[ind]

			points = (np.dot(R, p.T) + np.array([t,t,t,t,t]).T)[0]
			normals = np.dot(R, n.T)

			X = np.array([points[0], normals[0]])
			Y = np.array([points[1], normals[1]])
			Z = np.array([points[2], normals[2]])

			coords = (np.dot(R, c.T) + np.array([t,t,t]).T)[0]

			CX = coords[0]
			CY = coords[1]
			CZ = coords[2]

			color = (random.random(),random.random(),random.random(), 0.8)

			self.ax.plot_surface(X, Z, Y, linewidth=0, color=color)

			self.ax.plot([t[0][0],CX[0]], [t[2][0],CZ[0]], [t[1][0],CY[0]], linewidth=1.0, color='red')
			self.ax.plot([t[0][0],CX[1]], [t[2][0],CZ[1]], [t[1][0],CY[1]], linewidth=1.0, color='green')
			self.ax.plot([t[0][0],CX[2]], [t[2][0],CZ[2]], [t[1][0],CY[2]], linewidth=1.0, color='blue')
			self.canvas.draw()
		
		self.Layout()

	def clear(self):
		self.ax.cla()
		self.printCanvas()


class LaserTriangulationMainPage(Page):

	def __init__(self, parent, afterCancelCallback=None, afterCalibrationCallback=None):
		Page.__init__(self, parent,
							title=_("Laser Triangulation"),
							subTitle=_("Put the pattern on the platform as shown in the picture and press Calibrate to continue"),
							left=_("Cancel"),
							right=_("Calibrate"),
							buttonLeftCallback=self.onCancel,
							buttonRightCallback=self.onCalibrate,
							panelOrientation=wx.HORIZONTAL,
							viewProgress=True)

		self.driver = Driver.Instance()
		self.cameraIntrinsics = calibration.CameraIntrinsics.Instance()
		self.laserTriangulation = calibration.LaserTriangulation.Instance()


		self.onCalibration=False

		self.afterCancelCallback = afterCancelCallback
		self.afterCalibrationCallback = afterCalibrationCallback

		#-- Image View
		imageView = ImageView(self._panel)
		imageView.setImage(wx.Image(resources.getPathForImage("pattern-position-right.jpg")))

		#-- Video View
		self.videoView = VideoView(self._panel, self.getFrame, 50)
		self.videoView.SetBackgroundColour(wx.BLACK)

		#-- Layout
		self.addToPanel(imageView, 3)
		self.addToPanel(self.videoView, 2)

		#-- Events
		self.Bind(wx.EVT_SHOW, self.onShow)

		self.Layout()

	def initialize(self):
		self.gauge.SetValue(0)
		self._rightButton.Enable()

	def onShow(self, event):
		if event.GetShow():
			self.videoView.play()
			self.GetParent().Layout()
			self.Layout()
		else:
			try:
				self.initialize()
				self.videoView.stop()
			except:
				pass

	def getFrame(self):
		if self.onCalibration:
			frame = self.laserTriangulation.getImage()
		else:
			frame = self.driver.camera.captureImage()

		#if frame is not None:
		#	retval, frame = self.cameraIntrinsics.detectChessboard(frame)
		return frame

	def onCalibrate(self):
		self.onCalibration=True
		self.laserTriangulation.setImage(self.driver.camera.captureImage())

		self.laserTriangulation.setCallbacks(self.beforeCalibration,
											 lambda p: wx.CallAfter(self.progressCalibration,p),
											 lambda r: wx.CallAfter(self.afterCalibration,r))
		self.laserTriangulation.start()

	def beforeCalibration(self):
		self._rightButton.Disable()
		self.gauge.SetValue(0)
		self.waitCursor = wx.BusyCursor()

	def progressCalibration(self, progress):
		self.gauge.SetValue(progress)

	def afterCalibration(self, result):
		self.onCalibrationFinished(result)
		self.onCalibration=False

	def onCalibrationFinished(self, result):
		self._rightButton.Enable()
		if self.afterCalibrationCallback is not None:
			self.afterCalibrationCallback(result)
		if hasattr(self, 'waitCursor'):
			del self.waitCursor

	def onCancel(self):
		boardUnplugCallback = self.driver.board.unplugCallback
		cameraUnplugCallback = self.driver.camera.unplugCallback
		self.driver.board.setUnplugCallback(None)
		self.driver.camera.setUnplugCallback(None)
		if not hasattr(self, 'waitCursor'):
			self.waitCursor = wx.BusyCursor()
		self.onCalibration = False
		self.laserTriangulation.cancel()
		if self.afterCancelCallback is not None:
			self.afterCancelCallback()
		del self.waitCursor
		self.driver.board.setUnplugCallback(boardUnplugCallback)
		self.driver.camera.setUnplugCallback(cameraUnplugCallback)


class LaserTriangulationResultPage(Page):

	def __init__(self, parent, buttonRejectCallback=None, buttonAcceptCallback=None):
		Page.__init__(self, parent,
							title=_("Laser Triangulation"),
							left=_("Reject"),
							right=_("Accept"),
							buttonLeftCallback=buttonRejectCallback,
							buttonRightCallback=buttonAcceptCallback,
							panelOrientation=wx.HORIZONTAL)

		vbox = wx.BoxSizer(wx.VERTICAL)

		self.laserTriangulation = calibration.LaserTriangulation.Instance()
		self.plotPanel = LaserTriangulation3DPlot(self._panel)

		#-- Layout
		self.addToPanel(self.plotPanel, 3)

		#-- Events
		self.Bind(wx.EVT_SHOW, self.onShow)

	def onShow(self, event):
		if event.GetShow():
			self.GetParent().Layout()
			self.Layout()

	def processCalibration(self, response):
		ret, result = response

		if ret:
			dL = result[0][0]
			nL = result[0][1]
			stdL = result[0][2]
			dR = result[1][0]
			nR = result[1][1]
			stdR = result[1][2]

			self.GetParent().GetParent().controls.panels['laser_triangulation_panel'].setParameters((dL, nL, dR, nR))
			self.plotPanel.clear()
			self.plotPanel.add((dL, nL,stdL, dR, nR,stdR))
			self.plotPanel.Show()
			self.Layout()
		else:
			if result == Error.CalibrationError:
				dlg = wx.MessageDialog(self, _("Laser Triangulation Calibration has failed. Please try again."), _(result), wx.OK|wx.ICON_ERROR)
				dlg.ShowModal()
				dlg.Destroy()


class LaserTriangulation3DPlot(wx.Panel):

	def __init__(self, parent):
		wx.Panel.__init__(self, parent)

		self.initialize()

	def initialize(self):
		fig = Figure(facecolor=(0.7490196,0.7490196,0.7490196,1), tight_layout=True)
		self.canvas = FigureCanvasWxAgg(self, -1, fig)
		self.canvas.SetExtraStyle(wx.EXPAND)
		self.ax = fig.gca(projection='3d', axisbg=(0.7490196,0.7490196,0.7490196,1))

		self.Bind(wx.EVT_SIZE, self.onSize)
		self.Layout()

	def onSize(self,event):
		self.canvas.SetClientSize(self.GetClientSize())
		self.canvas.draw()
		self.Layout()

	def add(self, args):
		dL, nL, stdL, dR, nR, stdR = args

		rL = np.cross(np.array([0,0,1]), nL)
		sL = np.cross(rL, nL)
		RL = np.array([rL, sL, nL])

		rR = np.cross(np.array([0,0,1]), nR)
		sR = np.cross(rR, nR)
		RR = np.array([rR, sR, nR])

		self.addPlane(RL, dL*nL)
		self.addPlane(RR, dR*nR)

		self.ax.plot([0,50], [0,0], [0,0], linewidth=2.0, color='red')
		self.ax.plot([0,0], [0,0], [0,50], linewidth=2.0, color='green')
		self.ax.plot([0,0], [0,50], [0,0], linewidth=2.0, color='blue')

		self.ax.set_xlabel('X')
		self.ax.set_ylabel('Z')
		self.ax.set_zlabel('Y')

		self.ax.text(-100,0,0, str(round(stdL, 5)), fontsize=15)
		self.ax.text(100,0,0, str(round(stdR, 5)), fontsize=15)

		self.ax.set_xlim(-150, 150)
		self.ax.set_ylim(0, 400)
		self.ax.set_zlim(-150, 150)

		self.ax.invert_xaxis()
		self.ax.invert_yaxis()
		self.ax.invert_zaxis()

		self.canvas.draw()
		self.Layout()

	def addPlane(self, R, t):
		w = 200
		h = 300

		p = np.array([[-w/2,-h/2,0],[-w/2,h/2,0],[w/2,h/2,0],[w/2,-h/2,0],[-w/2,-h/2,0]])
		n = np.array([[0,0,1],[0,0,1],[0,0,1],[0,0,1],[0,0,1]])

		self.ax.plot([0,t[0]], [0,t[2]], [0,t[1]], linewidth=2.0, color='yellow')

		points = np.dot(R.T, p.T) + np.array([t,t,t,t,t]).T 
		normals = np.dot(R.T, n.T)

		X = np.array([points[0], normals[0]])
		Y = np.array([points[1], normals[1]])
		Z = np.array([points[2], normals[2]])

		self.ax.plot_surface(X, Z, Y, linewidth=0, color=(1,0,0,0.8))

		self.canvas.draw()

	def clear(self):
		self.ax.cla()


class SimpleLaserTriangulationMainPage(Page):

	def __init__(self, parent, afterCancelCallback=None, afterCalibrationCallback=None):
		Page.__init__(self, parent,
							title=_("Simple Laser Triangulation"),
							subTitle=_("Put the pattern on the platform as shown in the picture and press Calibrate to continue"),
							left=_("Cancel"),
							right=_("Calibrate"),
							buttonLeftCallback=self.onCancel,
							buttonRightCallback=self.onCalibrate,
							panelOrientation=wx.HORIZONTAL,
							viewProgress=True)

		self.driver = Driver.Instance()
		self.cameraIntrinsics = calibration.CameraIntrinsics.Instance()
		self.laserTriangulation = calibration.LaserTriangulation.Instance()

		self.afterCancelCallback = afterCancelCallback
		self.afterCalibrationCallback = afterCalibrationCallback

		#-- Image View
		imageView = ImageView(self._panel)
		imageView.setImage(wx.Image(resources.getPathForImage("pattern-position-right.jpg")))

		#-- Video View
		self.videoView = VideoView(self._panel, self.getFrame, 50)
		self.videoView.SetBackgroundColour(wx.BLACK)

		#-- Layout
		self.addToPanel(imageView, 3)
		self.addToPanel(self.videoView, 2)

		#-- Events
		self.Bind(wx.EVT_SHOW, self.onShow)

		self.Layout()

	def initialize(self):
		self.gauge.SetValue(0)
		self._rightButton.Enable()

	def onShow(self, event):
		if event.GetShow():
			self.videoView.play()
			self.GetParent().Layout()
			self.Layout()
		else:
			try:
				self.initialize()
				self.videoView.stop()
			except:
				pass

	def getFrame(self):
		frame = self.driver.camera.captureImage()
		if frame is not None:
			retval, frame = self.cameraIntrinsics.detectChessboard(frame)
		return frame

	def onCalibrate(self):
		self.laserTriangulation.setCallbacks(self.beforeCalibration,
											 lambda p: wx.CallAfter(self.progressCalibration,p),
											 lambda r: wx.CallAfter(self.afterCalibration,r))
		self.laserTriangulation.start()

	def beforeCalibration(self):
		self._rightButton.Disable()
		self.gauge.SetValue(0)
		if not hasattr(self, 'waitCursor'):
			self.waitCursor = wx.BusyCursor()

	def progressCalibration(self, progress):
		self.gauge.SetValue(progress)

	def afterCalibration(self, result):
		self._rightButton.Enable()
		if hasattr(self, 'waitCursor'):
			del self.waitCursor
		if self.afterCalibrationCallback is not None:
			self.afterCalibrationCallback(result)

	def onCancel(self):
		boardUnplugCallback = self.driver.board.unplugCallback
		cameraUnplugCallback = self.driver.camera.unplugCallback
		self.driver.board.setUnplugCallback(None)
		self.driver.camera.setUnplugCallback(None)
		if not hasattr(self, 'waitCursor'):
			self.waitCursor = wx.BusyCursor()
		self.onCalibration = False
		self.laserTriangulation.cancel()
		if self.afterCancelCallback is not None:
			self.afterCancelCallback()
		del self.waitCursor
		self.driver.board.setUnplugCallback(boardUnplugCallback)
		self.driver.camera.setUnplugCallback(cameraUnplugCallback)


class SimpleLaserTriangulationResultPage(Page):

	def __init__(self, parent, buttonRejectCallback=None, buttonAcceptCallback=None):
		Page.__init__(self, parent,
							title=_("Laser Triangulation"),
							left=_("Reject"),
							right=_("Accept"),
							buttonLeftCallback=buttonRejectCallback,
							buttonRightCallback=buttonAcceptCallback,
							panelOrientation=wx.HORIZONTAL)

		vbox = wx.BoxSizer(wx.VERTICAL)

		self.laserTriangulation = calibration.LaserTriangulation.Instance()

		self.leftLaserImageSequence = SimpleLaserTriangulationImageSequence(self._panel, "Left Laser Image Sequence")
		self.rightLaserImageSequence = SimpleLaserTriangulationImageSequence(self._panel, "Right Laser Image Sequence")

		#-- Layout
		vbox.Add(self.leftLaserImageSequence, 1, wx.ALL|wx.EXPAND, 3)
		vbox.Add(self.rightLaserImageSequence, 1, wx.ALL|wx.EXPAND, 3)

		self.addToPanel(vbox, 3)

		#-- Events
		self.Bind(wx.EVT_SHOW, self.onShow)

	def onShow(self, event):
		if event.GetShow():
			self.GetParent().Layout()
			self.Layout()

	def processCalibration(self, response):
		ret, result = response

		if ret:
			vectors, parameters, images = result
			self.GetParent().GetParent().controls.panels['laser_triangulation_panel'].setParameters((parameters, vectors[0], vectors[1]))
			self.leftLaserImageSequence.imageLas.setFrame(images[0][0])
			self.leftLaserImageSequence.imageGray.setFrame(images[0][1])
			self.leftLaserImageSequence.imageBin.setFrame(images[0][2])
			self.leftLaserImageSequence.imageLine.setFrame(images[0][3])
			self.rightLaserImageSequence.imageLas.setFrame(images[1][0])
			self.rightLaserImageSequence.imageGray.setFrame(images[1][1])
			self.rightLaserImageSequence.imageBin.setFrame(images[1][2])
			self.rightLaserImageSequence.imageLine.setFrame(images[1][3])
			self.Layout()
		else:
			if result == Error.CalibrationError:
				dlg = wx.MessageDialog(self, _("Laser Triangulation Calibration has failed. Please try again."), _(result), wx.OK|wx.ICON_ERROR)
				dlg.ShowModal()
				dlg.Destroy()


class SimpleLaserTriangulationImageSequence(wx.Panel):

	def __init__(self, parent, title="Title"):
		wx.Panel.__init__(self, parent)

		vbox = wx.BoxSizer(wx.VERTICAL)
		hbox = wx.BoxSizer(wx.HORIZONTAL)

		titleText = wx.StaticText(self, label=title)
		titleText.SetFont((wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.NORMAL, wx.FONTWEIGHT_BOLD)))

		panel = wx.Panel(self)
		self.imageLas = ImageView(panel)
		self.imageGray = ImageView(panel)
		self.imageBin = ImageView(panel)
		self.imageLine = ImageView(panel)

		self.imageLas.SetBackgroundColour('#AAAAAA')
		self.imageGray.SetBackgroundColour('#AAAAAA')
		self.imageBin.SetBackgroundColour('#AAAAAA')
		self.imageLine.SetBackgroundColour('#AAAAAA')

		#-- Layout
		vbox.Add(titleText, 0, wx.ALL|wx.EXPAND, 5)
		hbox.Add(self.imageLas, 1, wx.ALL|wx.EXPAND, 3)
		hbox.Add(self.imageGray, 1, wx.ALL|wx.EXPAND, 3)
		hbox.Add(self.imageBin, 1, wx.ALL|wx.EXPAND, 3)
		hbox.Add(self.imageLine, 1, wx.ALL|wx.EXPAND, 3)
		panel.SetSizer(hbox)
		vbox.Add(panel, 1, wx.ALL|wx.EXPAND, 3)

		self.SetSizer(vbox)
		self.Layout()


class PlatformExtrinsicsMainPage(Page):

	def __init__(self, parent, afterCancelCallback=None, afterCalibrationCallback=None):
		Page.__init__(self, parent,
							title=_("Platform Extrinsics"),
							subTitle=_("Put the pattern on the platform as shown in the picture and press Calibrate to continue"),
							left=_("Cancel"),
							right=_("Calibrate"),
							buttonLeftCallback=self.onCancel,
							buttonRightCallback=self.onCalibrate,
							panelOrientation=wx.HORIZONTAL,
							viewProgress=True)

		self.driver = Driver.Instance()
		self.cameraIntrinsics = calibration.CameraIntrinsics.Instance()
		self.platformExtrinsics = calibration.PlatformExtrinsics.Instance()

		self.onCalibration=False

		self.afterCancelCallback = afterCancelCallback
		self.afterCalibrationCallback = afterCalibrationCallback

		#-- Image View
		imageView = ImageView(self._panel)
		imageView.setImage(wx.Image(resources.getPathForImage("pattern-position-left.jpg")))

		#-- Video View
		self.videoView = VideoView(self._panel, self.getFrame, 50)
		self.videoView.SetBackgroundColour(wx.BLACK)

		#-- Layout
		self.addToPanel(imageView, 3)
		self.addToPanel(self.videoView, 2)

		#-- Events
		self.Bind(wx.EVT_SHOW, self.onShow)

		self.Layout()

	def initialize(self):
		self.gauge.SetValue(0)
		self._rightButton.Enable()

	def onShow(self, event):
		if event.GetShow():
			self.videoView.play()
			self.GetParent().Layout()
			self.Layout()
		else:
			try:
				self.initialize()
				self.videoView.stop()
			except:
				pass

	def getFrame(self):
		if self.onCalibration:
			frame = self.platformExtrinsics.getImage()
		else:
			frame = self.driver.camera.captureImage()
		if frame is not None:
			retval, frame = self.cameraIntrinsics.detectChessboard(frame)

		return frame

	def onCalibrate(self):
		self.onCalibration=True
		self.platformExtrinsics.setImage(self.driver.camera.captureImage())

		self.platformExtrinsics.setCallbacks(self.beforeCalibration,
											 lambda p: wx.CallAfter(self.progressCalibration,p),
											 lambda r: wx.CallAfter(self.afterCalibration,r))
		self.platformExtrinsics.start()

	def beforeCalibration(self):
		self._rightButton.Disable()
		self.gauge.SetValue(0)
		self.waitCursor = wx.BusyCursor()

	def progressCalibration(self, progress):
		self.gauge.SetValue(progress)

	def afterCalibration(self, result):
		self.onCalibrationFinished(result)
		self.onCalibration=False

	def onCalibrationFinished(self, result):
		self._rightButton.Enable()
		if self.afterCalibrationCallback is not None:
			self.afterCalibrationCallback(result)
		if hasattr(self, 'waitCursor'):
			del self.waitCursor

	def onCancel(self):
		boardUnplugCallback = self.driver.board.unplugCallback
		cameraUnplugCallback = self.driver.camera.unplugCallback
		self.driver.board.setUnplugCallback(None)
		self.driver.camera.setUnplugCallback(None)
		if not hasattr(self, 'waitCursor'):
			self.waitCursor = wx.BusyCursor()
		self.onCalibration = False
		self.platformExtrinsics.cancel()
		if self.afterCancelCallback is not None:
			self.afterCancelCallback()
		del self.waitCursor
		self.driver.board.setUnplugCallback(boardUnplugCallback)
		self.driver.camera.setUnplugCallback(cameraUnplugCallback)


class PlatformExtrinsicsResultPage(Page):

	def __init__(self, parent, buttonRejectCallback=None, buttonAcceptCallback=None):
		Page.__init__(self, parent,
							title=_("Platform Extrinsics"),
							left=_("Reject"),
							right=_("Accept"),
							buttonLeftCallback=buttonRejectCallback,
							buttonRightCallback=buttonAcceptCallback,
							panelOrientation=wx.HORIZONTAL)

		vbox = wx.BoxSizer(wx.VERTICAL)

		self.platformExtrinsics = calibration.PlatformExtrinsics.Instance()
		self.plotPanel = PlatformExtrinsics3DPlot(self._panel)

		#-- Layout
		self.addToPanel(self.plotPanel, 3)

		#-- Events
		self.Bind(wx.EVT_SHOW, self.onShow)

	def onShow(self, event):
		if event.GetShow():
			self.GetParent().Layout()
			self.Layout()

	def processCalibration(self, response):
		ret, result = response

		if ret:
			R = result[0]
			t = result[1]
			self.GetParent().GetParent().controls.panels['platform_extrinsics_panel'].setParameters((R, t))
			self.plotPanel.clear()
			self.plotPanel.add(result)
			self.plotPanel.Show()
			self.Layout()
		else:
			if result == Error.CalibrationError:
				dlg = wx.MessageDialog(self, _("Platform Extrinsics Calibration has failed. Please try again."), _(result), wx.OK|wx.ICON_ERROR)
				dlg.ShowModal()
				dlg.Destroy()


class PlatformExtrinsics3DPlot(wx.Panel):

	def __init__(self, parent):
		wx.Panel.__init__(self, parent)

		self.initialize()

	def initialize(self):
		fig = Figure(facecolor=(0.7490196,0.7490196,0.7490196,1), tight_layout=True)
		self.canvas = FigureCanvasWxAgg(self, -1, fig)
		self.canvas.SetExtraStyle(wx.EXPAND)
		self.ax = fig.gca(projection='3d', axisbg=(0.7490196,0.7490196,0.7490196,1))

		self.Bind(wx.EVT_SIZE, self.onSize)
		self.Layout()

	def onSize(self,event):
		self.canvas.SetClientSize(self.GetClientSize())
		self.canvas.draw()
		self.Layout()

	def add(self, args):
		R, t, center, point, normal, [x,y,z], circle = args

		# plot the surface, data, and synthetic circle
		self.ax.scatter(x, z, y, c='b', marker='o')
		#self.ax.scatter(center[0], center[2], center[1], c='b', marker='o')
		self.ax.plot(circle[0], circle[2], circle[1], c='r')

		d = profile.getProfileSettingFloat('pattern_distance')

		self.ax.plot([t[0],t[0]+50*R[0][0]], [t[2],t[2]+50*R[2][0]], [t[1],t[1]+50*R[1][0]], linewidth=2.0, color='red')
		self.ax.plot([t[0],t[0]+50*R[0][1]], [t[2],t[2]+50*R[2][1]], [t[1],t[1]+50*R[1][1]], linewidth=2.0, color='green')
		self.ax.plot([t[0],t[0]+d*R[0][2]], [t[2],t[2]+d*R[2][2]], [t[1],t[1]+d*R[1][2]], linewidth=2.0, color='blue')

		self.ax.plot([0,50], [0,0], [0,0], linewidth=2.0, color='red')
		self.ax.plot([0,0], [0,0], [0,50], linewidth=2.0, color='green')
		self.ax.plot([0,0], [0,50], [0,0], linewidth=2.0, color='blue')

		self.ax.set_xlabel('X')
		self.ax.set_ylabel('Z')
		self.ax.set_zlabel('Y')

		self.ax.set_xlim(-150, 150)
		self.ax.set_ylim(0, 400)
		self.ax.set_zlim(-150, 150)

		self.ax.invert_xaxis()
		self.ax.invert_yaxis()
		self.ax.invert_zaxis()

		self.canvas.draw()
		self.Layout()

	def clear(self):
		self.ax.cla()