#!/usr/bin/python
# -*- coding: utf-8 -*-
#-----------------------------------------------------------------------#
#                                                                       #
# This file is part of the Horus Project                                #
#                                                                       #
# Copyright (C) 2014-2015 Mundo Reader S.L.                             #
#                                                                       #
# Date: June, November 2014                                             #
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

import horus.util.error as Error
from horus.util import resources, profile

from horus.gui.util.imageView import VideoView
from horus.gui.util.sceneView import SceneView
from horus.gui.util.customPanels import ExpandableControl

from horus.gui.workbench.workbench import WorkbenchConnection
from horus.gui.workbench.scanning.panels import ScanParameters, RotativePlatform, ImageAcquisition, \
												ImageSegmentation, PointCloudGeneration

from horus.engine.scan import SimpleScan, TextureScan

class ScanningWorkbench(WorkbenchConnection):

	def __init__(self, parent):
		WorkbenchConnection.__init__(self, parent)

		self.scanning = False
		self.showVideoViews = False

		self.simpleScan = SimpleScan.Instance()
		self.textureScan = TextureScan.Instance()

		self.currentScan = self.simpleScan

		self.load()

		self.pointCloudTimer = wx.Timer(self)
		self.Bind(wx.EVT_TIMER, self.onPointCloudTimer, self.pointCloudTimer)

	def load(self):
		#-- Toolbar Configuration
		self.playTool   = self.toolbar.AddLabelTool(wx.NewId(), _("Play"), wx.Bitmap(resources.getPathForImage("play.png")), shortHelp=_("Play"))
		self.stopTool   = self.toolbar.AddLabelTool(wx.NewId(), _("Stop"), wx.Bitmap(resources.getPathForImage("stop.png")), shortHelp=_("Stop"))
		self.pauseTool  = self.toolbar.AddLabelTool(wx.NewId(), _("Pause"), wx.Bitmap(resources.getPathForImage("pause.png")), shortHelp=_("Pause"))
		self.toolbar.Realize()

		#-- Disable Toolbar Items
		self.enableLabelTool(self.playTool  , False)
		self.enableLabelTool(self.stopTool  , False)
		self.enableLabelTool(self.pauseTool , False)

		#-- Bind Toolbar Items
		self.Bind(wx.EVT_TOOL, self.onPlayToolClicked  , self.playTool)
		self.Bind(wx.EVT_TOOL, self.onStopToolClicked  , self.stopTool)
		self.Bind(wx.EVT_TOOL, self.onPauseToolClicked , self.pauseTool)

		self.scrollPanel = wx.lib.scrolledpanel.ScrolledPanel(self._panel, size=(290,-1))
		self.scrollPanel.SetupScrolling(scroll_x=False, scrollIntoView=False)
		self.scrollPanel.SetAutoLayout(1)

		self.controls = ExpandableControl(self.scrollPanel)

		self.controls.addPanel('scan_parameters', ScanParameters(self.controls))
		self.controls.addPanel('rotative_platform', RotativePlatform(self.controls))
		self.controls.addPanel('image_acquisition', ImageAcquisition(self.controls))
		self.controls.addPanel('image_segmentation', ImageSegmentation(self.controls))
		self.controls.addPanel('point_cloud_generation', PointCloudGeneration(self.controls))

		self.splitterWindow = wx.SplitterWindow(self._panel)

		self.videoView = VideoView(self.splitterWindow, self.getFrame, 10)
		self.videoView.SetBackgroundColour(wx.BLACK)

		self.scenePanel = wx.Panel(self.splitterWindow)
		self.sceneView = SceneView(self.scenePanel)
		self.gauge = wx.Gauge(self.scenePanel, size=(-1, 30))
		self.gauge.Hide()

		vbox = wx.BoxSizer(wx.VERTICAL)
		vbox.Add(self.sceneView, 1, wx.ALL|wx.EXPAND, 0)
		vbox.Add(self.gauge, 0, wx.ALL|wx.EXPAND, 0)
		self.scenePanel.SetSizer(vbox)

		self.splitterWindow.SplitVertically(self.videoView, self.scenePanel)
		self.splitterWindow.SetMinimumPaneSize(200)

		#-- Layout
		vsbox = wx.BoxSizer(wx.VERTICAL)
		vsbox.Add(self.controls, 0, wx.ALL|wx.EXPAND, 0)
		self.scrollPanel.SetSizer(vsbox)
		vsbox.Fit(self.scrollPanel)

		self.addToPanel(self.scrollPanel, 0)
		self.addToPanel(self.splitterWindow, 1)

		#- Video View Selector
		_choices = []
		choices = profile.getProfileSettingObject('img_type').getType()
		for i in choices:
			_choices.append(_(i))
		self.videoViewsDict = dict(zip(_choices, choices))

		self.buttonShowVideoViews = wx.BitmapButton(self.videoView, wx.NewId(), wx.Bitmap(resources.getPathForImage("views.png"), wx.BITMAP_TYPE_ANY), (10,10))
		self.comboVideoViews = wx.ComboBox(self.videoView, value=_(profile.getProfileSetting('img_type')), choices=_choices, style=wx.CB_READONLY, pos=(60,10))

		self.buttonShowVideoViews.Hide()
		self.comboVideoViews.Hide()

		self.buttonShowVideoViews.Bind(wx.EVT_BUTTON, self.onShowVideoViews)
		self.comboVideoViews.Bind(wx.EVT_COMBOBOX, self.onComboBoVideoViewsSelect)

		self.updateCallbacks()
		self.Layout()

	def updateCallbacks(self):
		self.controls.updateCallbacks()

	def enableRestore(self, value):
		self.controls.enableRestore(value)

	def onShow(self, event):
		if event.GetShow():
			self.updateStatus(self.currentScan.driver.isConnected)
			self.pointCloudTimer.Stop()
		else:
			try:
				self.pointCloudTimer.Stop()
				self.videoView.stop()
			except:
				pass

	def onShowVideoViews(self, event):
		self.showVideoViews = not self.showVideoViews
		if self.showVideoViews:
			self.comboVideoViews.Show()
		else:
			self.comboVideoViews.Hide()

	def onComboBoVideoViewsSelect(self, event):
		value = self.videoViewsDict[self.comboVideoViews.GetValue()]
		self.currentScan.setImageType(value)
		profile.putProfileSetting('img_type', value)

	def getFrame(self):
		if self.scanning:
			return self.currentScan.getImage()
		else:
			return self.currentScan.getImage(self.driver.camera.captureImage())

	def onPointCloudTimer(self, event):
		pointCloud = self.currentScan.getPointCloudIncrement()
		if pointCloud is not None:
			if pointCloud[0] is not None and pointCloud[1] is not None:
				if len(pointCloud[0]) > 0:
					self.sceneView.appendPointCloud(pointCloud[0], pointCloud[1])

	def onPlayToolClicked(self, event):
		if self.currentScan.inactive:
			#-- Resume
			self.enableLabelTool(self.pauseTool , True)
			self.enableLabelTool(self.playTool, False)
			self.currentScan.resume()
			self.pointCloudTimer.Start(milliseconds=50)
		else:
			result = True
			if self.sceneView._object is not None:
				dlg = wx.MessageDialog(self, _("Your current model will be erased.\nDo you really want to do it?"), _("Clear Point Cloud"), wx.YES_NO | wx.ICON_QUESTION)
				result = dlg.ShowModal() == wx.ID_YES
				dlg.Destroy()
			if result:
				value = profile.getProfileSetting('scan_type')
				if value == 'Simple Scan':
					self.currentScan = self.simpleScan
				elif value == 'Texture Scan':
					self.currentScan = self.textureScan
				self.gauge.SetValue(0)
				self.gauge.Show()
				self.scenePanel.Layout()
				self.Layout()
				self.currentScan.setCallbacks(self.beforeScan, self.progressScan, lambda r: wx.CallAfter(self.afterScan,r))
				self.currentScan.start()

	def beforeScan(self):
		self.scanning = True
		self.buttonShowVideoViews.Show()
		self.enableLabelTool(self.disconnectTool, False)
		self.enableLabelTool(self.playTool, False)
		self.enableLabelTool(self.stopTool, True)
		self.enableLabelTool(self.pauseTool , True)
		self.sceneView.createDefaultObject()
		self.sceneView.setShowDeleteMenu(False)
		self.videoView.setMilliseconds(200)
		self.combo.Disable()
		self.GetParent().menuFile.Enable(self.GetParent().menuLaunchWizard.GetId(), False)
		self.GetParent().menuFile.Enable(self.GetParent().menuLoadModel.GetId(), False)
		self.GetParent().menuFile.Enable(self.GetParent().menuSaveModel.GetId(), False)
		self.GetParent().menuFile.Enable(self.GetParent().menuClearModel.GetId(), False)
		self.GetParent().menuFile.Enable(self.GetParent().menuOpenProfile.GetId(), False)
		self.GetParent().menuFile.Enable(self.GetParent().menuSaveProfile.GetId(), False)
		self.GetParent().menuFile.Enable(self.GetParent().menuResetProfile.GetId(), False)
		self.GetParent().menuFile.Enable(self.GetParent().menuExit.GetId(), False)
		self.GetParent().menuEdit.Enable(self.GetParent().menuPreferences.GetId(), False)
		self.GetParent().menuHelp.Enable(self.GetParent().menuWelcome.GetId(), False)
		panel = self.controls.panels['scan_parameters']
		section = panel.sections['scan_parameters']
		section.disable('scan_type')
		section.disable('use_laser')
		panel = self.controls.panels['rotative_platform']
		section = panel.sections['motor_scanning']
		section.disable('feed_rate_scanning')
		section.disable('acceleration_scanning')
		panel = self.controls.panels['image_acquisition']
		section = panel.sections['camera_scanning']
		section.disable('framerate_scanning')
		section.disable('resolution_scanning')
		self.enableRestore(False)
		self.pointCloudTimer.Start(milliseconds=50)

	def progressScan(self, progress, range=100):
		self.gauge.SetRange(range)
		self.gauge.SetValue(progress)

	def afterScan(self, response):
		ret, result = response
		if ret:
			dlg = wx.MessageDialog(self, _("Scanning has finished. If you want to save your point cloud go to File > Save Model"), _("Scanning finished!"), wx.OK|wx.ICON_INFORMATION)
			dlg.ShowModal()
			dlg.Destroy()
			self.scanning = False
			self.onScanFinished()

	def onStopToolClicked(self, event):
		paused = self.currentScan.inactive
		self.currentScan.pause()
		dlg = wx.MessageDialog(self, _("Your current scanning will be stopped.\nDo you really want to do it?"), _("Stop Scanning"), wx.YES_NO | wx.ICON_QUESTION)
		result = dlg.ShowModal() == wx.ID_YES
		dlg.Destroy()

		if result:
			self.scanning = False
			self.currentScan.stop()
			self.onScanFinished()
		else:
			if not paused:
				self.currentScan.resume()

	def onScanFinished(self):
		self.buttonShowVideoViews.Hide()
		self.comboVideoViews.Hide()
		self.enableLabelTool(self.disconnectTool, True)
		self.enableLabelTool(self.playTool, True)
		self.enableLabelTool(self.stopTool, False)
		self.enableLabelTool(self.pauseTool , False)
		self.sceneView.setShowDeleteMenu(True)
		self.combo.Enable()
		self.GetParent().menuFile.Enable(self.GetParent().menuLaunchWizard.GetId(), True)
		self.GetParent().menuFile.Enable(self.GetParent().menuLoadModel.GetId(), True)
		self.GetParent().menuFile.Enable(self.GetParent().menuSaveModel.GetId(), True)
		self.GetParent().menuFile.Enable(self.GetParent().menuClearModel.GetId(), True)
		self.GetParent().menuFile.Enable(self.GetParent().menuOpenProfile.GetId(), True)
		self.GetParent().menuFile.Enable(self.GetParent().menuSaveProfile.GetId(), True)
		self.GetParent().menuFile.Enable(self.GetParent().menuResetProfile.GetId(), True)
		self.GetParent().menuFile.Enable(self.GetParent().menuExit.GetId(), True)
		self.GetParent().menuEdit.Enable(self.GetParent().menuPreferences.GetId(), True)
		self.GetParent().menuHelp.Enable(self.GetParent().menuWelcome.GetId(), True)
		panel = self.controls.panels['scan_parameters']
		section = panel.sections['scan_parameters']
		section.enable('scan_type')
		section.enable('use_laser')
		panel = self.controls.panels['rotative_platform']
		section = panel.sections['motor_scanning']
		section.enable('feed_rate_scanning')
		section.enable('acceleration_scanning')
		panel = self.controls.panels['image_acquisition']
		section = panel.sections['camera_scanning']
		section.enable('framerate_scanning')
		section.enable('resolution_scanning')
		self.enableRestore(True)
		self.pointCloudTimer.Stop()
		self.videoView.setMilliseconds(10)
		self.gauge.Hide()
		self.scenePanel.Layout()
		self.Layout()

	def onPauseToolClicked(self, event):
		self.enableLabelTool(self.pauseTool , False)
		self.enableLabelTool(self.playTool, True)
		self.currentScan.pause()
		self.pointCloudTimer.Stop()

	def updateToolbarStatus(self, status):
		if status:
			if self.IsShown():
				self.videoView.play()
			self.enableLabelTool(self.playTool  , True)
			self.enableLabelTool(self.stopTool  , False)
			self.enableLabelTool(self.pauseTool , False)
			self.controls.enableContent()
		else:
			self.videoView.stop()
			self.enableLabelTool(self.playTool  , False)
			self.enableLabelTool(self.stopTool  , False)
			self.enableLabelTool(self.pauseTool , False)
			self.controls.disableContent()

	def updateProfileToAllControls(self):
		self.controls.updateProfile()
