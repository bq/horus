#!/usr/bin/python
# -*- coding: utf-8 -*-
#-----------------------------------------------------------------------#
#                                                                       #
# This file is part of the Horus Project                                #
#                                                                       #
# Copyright (C) 2014-2015 Mundo Reader S.L.                             #
#                                                                       #
# Date: October 2014                                                    #
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

import wx._core
import time

from horus.gui.util.imageView import ImageView
from horus.gui.util.patternDistanceWindow import PatternDistanceWindow

from horus.gui.wizard.wizardPage import WizardPage

import horus.util.error as Error
from horus.util import profile, resources

from horus.engine.driver import Driver
from horus.engine import calibration


class CalibrationPage(WizardPage):
	def __init__(self, parent, buttonPrevCallback=None, buttonNextCallback=None):
		WizardPage.__init__(self, parent,
							title=_("Calibration"),
							buttonPrevCallback=buttonPrevCallback,
							buttonNextCallback=buttonNextCallback)

		self.driver = Driver.Instance()
		self.cameraIntrinsics = calibration.CameraIntrinsics.Instance()
		self.laserTriangulation = calibration.LaserTriangulation.Instance()
		self.platformExtrinsics = calibration.PlatformExtrinsics.Instance()
		self.phase = 'none'

		self.patternLabel = wx.StaticText(self.panel, label=_("Put the pattern on the platform as shown in the picture and press \"Calibrate\""))
		self.patternLabel.Wrap(400)
		self.imageView = ImageView(self.panel)
		self.imageView.setImage(wx.Image(resources.getPathForImage("pattern-position-right.jpg")))
		self.calibrateButton = wx.Button(self.panel, label=_("Calibrate"))
		self.cancelButton = wx.Button(self.panel, label=_("Cancel"))
		self.gauge = wx.Gauge(self.panel, range=100, size=(-1, 30))
		self.resultLabel = wx.StaticText(self.panel, size=(-1, 30))

		self.cancelButton.Disable()
		self.resultLabel.Hide()
		self.skipButton.Enable()
		self.nextButton.Disable()

		#-- Layout
		vbox = wx.BoxSizer(wx.VERTICAL)
		vbox.Add(self.patternLabel, 0, wx.ALL|wx.CENTER, 5)
		vbox.Add(self.imageView, 1, wx.ALL|wx.EXPAND, 5)
		vbox.Add(self.resultLabel, 0, wx.ALL|wx.CENTER, 5)
		vbox.Add(self.gauge, 0, wx.ALL|wx.EXPAND, 5)
		hbox = wx.BoxSizer(wx.HORIZONTAL)
		hbox.Add(self.cancelButton, 1, wx.ALL|wx.EXPAND, 5)
		hbox.Add(self.calibrateButton, 1, wx.ALL|wx.EXPAND, 5)
		vbox.Add(hbox, 0, wx.ALL|wx.EXPAND, 2)
		self.panel.SetSizer(vbox)

		self.Layout()

		self.calibrateButton.Bind(wx.EVT_BUTTON, self.onCalibrationButtonClicked)
		self.cancelButton.Bind(wx.EVT_BUTTON, self.onCancelButtonClicked)
		self.Bind(wx.EVT_SHOW, self.onShow)

		self.videoView.setMilliseconds(20)
		self.videoView.setCallback(self.getFrame)


	def onShow(self, event):
		if event.GetShow():
			self.updateStatus(self.driver.isConnected)
		else:
			try:
				self.videoView.stop()
			except:
				pass

	def getFrame(self):
		if self.phase is 'platformCalibration':
			frame = self.platformExtrinsics.getImage()
		elif self.phase is 'laserTriangulation':
			frame = self.laserTriangulation.getImage()
		else: # 'none'
			frame = self.driver.camera.captureImage()

		if frame is not None and self.phase is not 'laserTriangulation':
			retval, frame = self.cameraIntrinsics.detectChessboard(frame)

		return frame

	def onUnplugged(self):
		self.videoView.stop()
		self.laserTriangulation.cancel()
		self.platformExtrinsics.cancel()
		self.enableNext = True

	def onCalibrationButtonClicked(self, event):
		self.phase = 'laserTriangulation'
		self.laserTriangulation.setCallbacks(self.beforeCalibration,
											 lambda p: wx.CallAfter(self.progressLaserCalibration,p),
											 lambda r: wx.CallAfter(self.afterLaserCalibration,r))
		if profile.getProfileSettingFloat('pattern_distance') == 0:
			PatternDistanceWindow(self)
		else:
			self.laserTriangulation.start()

	def onCancelButtonClicked(self, event):
		boardUnplugCallback = self.driver.board.unplugCallback
		cameraUnplugCallback = self.driver.camera.unplugCallback
		self.driver.board.setUnplugCallback(None)
		self.driver.camera.setUnplugCallback(None)
		self.phase = 'none'
		self.resultLabel.SetLabel(_("Calibration canceled. To try again press \"Calibrate\""))
		self.platformExtrinsics.cancel()
		self.laserTriangulation.cancel()
		self.skipButton.Enable()
		self.onFinishCalibration()
		self.driver.board.setUnplugCallback(boardUnplugCallback)
		self.driver.camera.setUnplugCallback(cameraUnplugCallback)

	def beforeCalibration(self):
		self.calibrateButton.Disable()
		self.cancelButton.Enable()
		self.prevButton.Disable()
		self.skipButton.Disable()
		self.nextButton.Disable()
		self.enableNext = False
		self.gauge.SetValue(0)
		self.resultLabel.Hide()
		self.gauge.Show()
		self.Layout()
		self.waitCursor = wx.BusyCursor()

	def progressLaserCalibration(self, progress):
		self.gauge.SetValue(progress*0.6)

	def afterLaserCalibration(self, response):
		self.phase='platformCalibration'
		ret, result = response

		if ret:
			profile.putProfileSetting('distance_left', result[0][0])
			profile.putProfileSettingNumpy('normal_left', result[0][1])
			profile.putProfileSetting('distance_right', result[1][0])
			profile.putProfileSettingNumpy('normal_right', result[1][1])
			self.platformExtrinsics.setCallbacks(None,
												 lambda p: wx.CallAfter(self.progressPlatformCalibration,p),
												 lambda r: wx.CallAfter(self.afterPlatformCalibration,r))
			self.platformExtrinsics.start()
		else:
			if result == Error.CalibrationError:
				self.resultLabel.SetLabel(_("Error in lasers: please connect the lasers and try again"))
				dlg = wx.MessageDialog(self, _("Laser Calibration failed. Please try again"), _(result), wx.OK|wx.ICON_ERROR)
				dlg.ShowModal()
				dlg.Destroy()
				self.skipButton.Enable()
				self.onFinishCalibration()

	def progressPlatformCalibration(self, progress):
		self.gauge.SetValue(60 + progress*0.4)	

	def afterPlatformCalibration(self, response):
		self.phase = 'none'
		ret, result = response
		
		if ret:
			profile.putProfileSettingNumpy('rotation_matrix', result[0])
			profile.putProfileSettingNumpy('translation_vector', result[1])
		else:
			if result == Error.CalibrationError:
				self.resultLabel.SetLabel(_("Error in pattern: please check the pattern and try again"))
				dlg = wx.MessageDialog(self, _("Platform Calibration failed. Please try again"), _(result), wx.OK|wx.ICON_ERROR)
				dlg.ShowModal()
				dlg.Destroy()

		if ret:
			self.skipButton.Disable()
			self.nextButton.Enable()
			self.resultLabel.SetLabel(_("All OK. Please press next to continue"))
		else:
			self.skipButton.Enable()
			self.nextButton.Disable()

		self.onFinishCalibration()

	def onFinishCalibration(self):
		self.enableNext = True
		self.gauge.Hide()
		self.resultLabel.Show()
		self.calibrateButton.Enable()
		self.cancelButton.Disable()
		self.prevButton.Enable()
		self.panel.Fit()
		self.panel.Layout()
		self.Layout()
		if hasattr(self, 'waitCursor'):
			del self.waitCursor

	def updateStatus(self, status):
		if status:
			if profile.getPreference('workbench') != 'Calibration workbench':
				profile.putPreference('workbench', 'Calibration workbench')
				self.GetParent().parent.workbenchUpdate(False)
			self.videoView.play()
			self.calibrateButton.Enable()
			self.skipButton.Enable()
			self.driver.board.setLeftLaserOff()
			self.driver.board.setRightLaserOff()
		else:
			self.videoView.stop()
			self.gauge.SetValue(0)
			self.gauge.Show()
			self.prevButton.Enable()
			self.skipButton.Disable()
			self.nextButton.Disable()
			self.calibrateButton.Disable()
			self.cancelButton.Disable()