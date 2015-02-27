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

import wx.lib.scrolledpanel

from horus.util import resources, profile

from horus.gui.util.patternDistanceWindow import PatternDistanceWindow
from horus.gui.util.imageView import VideoView
from horus.gui.util.customPanels import ExpandableControl

from horus.gui.workbench.workbench import WorkbenchConnection
from horus.gui.workbench.calibration.panels import CameraSettingsPanel, PatternSettingsPanel, \
                                                   LaserSettingsPanel, CameraIntrinsicsPanel, \
                                                   LaserTriangulationPanel, PlatformExtrinsicsPanel
from horus.gui.workbench.calibration.pages import CameraIntrinsicsMainPage, CameraIntrinsicsResultPage, \
                                                  LaserTriangulationMainPage, LaserTriangulationResultPage, \
                                                  PlatformExtrinsicsMainPage, PlatformExtrinsicsResultPage

from horus.engine.driver import Driver
from horus.engine.calibration import CameraIntrinsics

class CalibrationWorkbench(WorkbenchConnection):

    def __init__(self, parent):
        WorkbenchConnection.__init__(self, parent)

        self.calibrating = False

        self.load()

    def load(self):
        #-- Toolbar Configuration
        self.toolbar.Realize()

        self.scrollPanel = wx.lib.scrolledpanel.ScrolledPanel(self._panel, size=(290,-1))
        self.scrollPanel.SetupScrolling(scroll_x=False, scrollIntoView=False)
        self.scrollPanel.SetAutoLayout(1)

        self.controls = ExpandableControl(self.scrollPanel)

        self.videoView = VideoView(self._panel, self.getFrame, 10)
        self.videoView.SetBackgroundColour(wx.BLACK)

        #-- Add Scroll Panels
        self.controls.addPanel('camera_settings', CameraSettingsPanel(self.controls))
        self.controls.addPanel('pattern_settings', PatternSettingsPanel(self.controls))
        self.controls.addPanel('laser_settings', LaserSettingsPanel(self.controls))
        self.controls.addPanel('camera_intrinsics_panel', CameraIntrinsicsPanel(self.controls, buttonStartCallback=self.onCameraIntrinsicsStartCallback))
        self.controls.addPanel('laser_triangulation_panel', LaserTriangulationPanel(self.controls, buttonStartCallback=self.onLaserTriangulationStartCallback))
        self.controls.addPanel('platform_extrinsics_panel', PlatformExtrinsicsPanel(self.controls, buttonStartCallback=self.onPlatformExtrinsicsStartCallback))

        #-- Add Calibration Pages
        self.cameraIntrinsicsMainPage = CameraIntrinsicsMainPage(self._panel,
                                                                 afterCancelCallback=self.onCancelCallback,
                                                                 afterCalibrationCallback=self.onCameraIntrinsicsAfterCalibrationCallback)

        self.cameraIntrinsicsResultPage = CameraIntrinsicsResultPage(self._panel,
                                                                     buttonRejectCallback=self.onCancelCallback,
                                                                     buttonAcceptCallback=self.onCameraIntrinsicsAcceptCallback)

        self.laserTriangulationMainPage = LaserTriangulationMainPage(self._panel,
                                                                     afterCancelCallback=self.onCancelCallback,
                                                                     afterCalibrationCallback=self.onLaserTriangulationAfterCalibrationCallback)

        self.laserTriangulationResultPage = LaserTriangulationResultPage(self._panel,
                                                                         buttonRejectCallback=self.onCancelCallback,
                                                                         buttonAcceptCallback=self.onLaserTriangulationAcceptCallback)

        self.platformExtrinsicsMainPage = PlatformExtrinsicsMainPage(self._panel,
                                                                     afterCancelCallback=self.onCancelCallback,
                                                                     afterCalibrationCallback=self.onPlatformExtrinsicsAfterCalibrationCallback)

        self.platformExtrinsicsResultPage = PlatformExtrinsicsResultPage(self._panel,
                                                                         buttonRejectCallback=self.onCancelCallback,
                                                                         buttonAcceptCallback=self.onPlatformExtrinsicsAcceptCallback)

        self.cameraIntrinsicsMainPage.Hide()
        self.cameraIntrinsicsResultPage.Hide()
        self.laserTriangulationMainPage.Hide()
        self.laserTriangulationResultPage.Hide()
        self.platformExtrinsicsMainPage.Hide()
        self.platformExtrinsicsResultPage.Hide()

        #-- Layout
        vsbox = wx.BoxSizer(wx.VERTICAL)
        vsbox.Add(self.controls, 0, wx.ALL|wx.EXPAND, 0)
        self.scrollPanel.SetSizer(vsbox)
        vsbox.Fit(self.scrollPanel)

        self.addToPanel(self.scrollPanel, 0)
        self.addToPanel(self.videoView, 1)

        self.addToPanel(self.cameraIntrinsicsMainPage, 1)
        self.addToPanel(self.cameraIntrinsicsResultPage, 1)
        self.addToPanel(self.laserTriangulationMainPage, 1)
        self.addToPanel(self.laserTriangulationResultPage, 1)
        self.addToPanel(self.platformExtrinsicsMainPage, 1)
        self.addToPanel(self.platformExtrinsicsResultPage, 1)

        self.Layout()

    def initialize(self):
        self.controls.initialize()

    def getFrame(self):
        frame = Driver.Instance().camera.captureImage()
        self.cameraIntrinsics = CameraIntrinsics.Instance()
        if frame is not None:
            retval, frame = self.cameraIntrinsics.detectChessboard(frame)
        return frame

    def onCameraIntrinsicsStartCallback(self):
        self.calibrating = True
        self.enableLabelTool(self.disconnectTool, False)
        self.controls.setExpandable(False)
        self.controls.panels['camera_intrinsics_panel'].buttonsPanel.Disable()
        self.combo.Disable()
        self.videoView.stop()
        self.videoView.Hide()
        self.cameraIntrinsicsMainPage.Show()
        self.cameraIntrinsicsMainPage.videoView.SetFocus()
        self.Layout()

    def onLaserTriangulationStartCallback(self):
        self.calibrating = True
        self.enableLabelTool(self.disconnectTool, False)
        self.controls.setExpandable(False)
        self.controls.panels['laser_triangulation_panel'].buttonsPanel.Disable()
        self.combo.Disable()
        self.videoView.stop()
        self.videoView.Hide()
        self.laserTriangulationMainPage.Show()
        self.Layout()

    def onPlatformExtrinsicsStartCallback(self):
        if profile.getProfileSettingFloat('pattern_distance') == 0:
            a=PatternDistanceWindow(self)
            self.updateProfileToAllControls()
        else:
            self.calibrating = True
            self.enableLabelTool(self.disconnectTool, False)
            self.controls.setExpandable(False)
            self.controls.panels['platform_extrinsics_panel'].buttonsPanel.Disable()
            self.combo.Disable()
            self.videoView.stop()
            self.videoView.Hide()
            self.platformExtrinsicsMainPage.Show()
            self.Layout()

    def onCancelCallback(self):
        self.calibrating = False
        self.enableLabelTool(self.disconnectTool, True)
        self.controls.setExpandable(True)
        self.controls.panels['camera_intrinsics_panel'].buttonsPanel.Enable()
        self.controls.panels['laser_triangulation_panel'].buttonsPanel.Enable()
        self.controls.panels['platform_extrinsics_panel'].buttonsPanel.Enable()
        self.controls.updateProfile()
        self.combo.Enable()
        self.cameraIntrinsicsMainPage.Hide()
        self.cameraIntrinsicsResultPage.Hide()
        self.laserTriangulationMainPage.Hide()
        self.laserTriangulationResultPage.Hide()
        self.platformExtrinsicsMainPage.Hide()
        self.platformExtrinsicsResultPage.Hide()
        self.videoView.play()
        self.videoView.Show()
        self.Layout()

    def onCameraIntrinsicsAfterCalibrationCallback(self, result):
        self.cameraIntrinsicsResultPage.processCalibration(result)
        if result[0]:
            self.cameraIntrinsicsMainPage.Hide()
            self.cameraIntrinsicsResultPage.Show()
        else:
            self.cameraIntrinsicsMainPage.initialize()
        self.Layout()

    def onCameraIntrinsicsAcceptCallback(self):
        self.videoView.play()
        self.calibrating = False
        self.enableLabelTool(self.disconnectTool, True)
        self.controls.setExpandable(True)
        self.controls.panels['camera_intrinsics_panel'].buttonsPanel.Enable()
        self.controls.panels['camera_intrinsics_panel'].updateAllControlsToProfile()
        self.combo.Enable()
        self.cameraIntrinsicsResultPage.Hide()
        self.videoView.Show()
        self.Layout()

    def onLaserTriangulationAfterCalibrationCallback(self, result):
        self.laserTriangulationResultPage.processCalibration(result)
        if result[0]:
            self.laserTriangulationMainPage.Hide()
            self.laserTriangulationResultPage.Show()
        else:
            self.laserTriangulationMainPage.initialize()
        self.Layout()

    def onLaserTriangulationAcceptCallback(self):
        self.videoView.play()
        self.calibrating = False
        self.enableLabelTool(self.disconnectTool, True)
        self.controls.setExpandable(True)
        self.controls.panels['laser_triangulation_panel'].buttonsPanel.Enable()
        self.controls.panels['laser_triangulation_panel'].updateAllControlsToProfile()
        self.combo.Enable()
        self.laserTriangulationResultPage.Hide()
        self.videoView.Show()
        self.Layout()

    def onPlatformExtrinsicsAfterCalibrationCallback(self, result):
        self.platformExtrinsicsResultPage.processCalibration(result)
        if result[0]:
            self.platformExtrinsicsMainPage.Hide()
            self.platformExtrinsicsResultPage.Show()
        else:
            self.platformExtrinsicsMainPage.initialize()
        self.Layout()

    def onPlatformExtrinsicsAcceptCallback(self):
        self.videoView.play()
        self.calibrating = False
        self.enableLabelTool(self.disconnectTool, True)
        self.controls.setExpandable(True)
        self.controls.panels['platform_extrinsics_panel'].buttonsPanel.Enable()
        self.controls.panels['platform_extrinsics_panel'].updateAllControlsToProfile()
        self.combo.Enable()
        self.platformExtrinsicsResultPage.Hide()
        self.videoView.Show()
        self.Layout()

    def updateToolbarStatus(self, status):
        if status:
            if self.IsShown():
                self.videoView.play()
            self.controls.panels['camera_intrinsics_panel'].buttonsPanel.Enable()
            self.controls.panels['laser_triangulation_panel'].buttonsPanel.Enable()
            self.controls.panels['platform_extrinsics_panel'].buttonsPanel.Enable()
            self.controls.enableContent()
        else:
            self.videoView.stop()
            self.controls.panels['camera_intrinsics_panel'].buttonsPanel.Disable()
            self.controls.panels['laser_triangulation_panel'].buttonsPanel.Disable()
            self.controls.panels['platform_extrinsics_panel'].buttonsPanel.Disable()
            self.controls.disableContent()
            self.calibrating = False
            self.combo.Enable()
            self.controls.setExpandable(True)
            self.cameraIntrinsicsMainPage.Hide()
            self.cameraIntrinsicsResultPage.Hide()
            self.laserTriangulationMainPage.Hide()
            self.laserTriangulationResultPage.Hide()
            self.platformExtrinsicsMainPage.Hide()
            self.platformExtrinsicsResultPage.Hide()
            self.videoView.Show()

    def updateProfileToAllControls(self):
        self.controls.updateProfile()
