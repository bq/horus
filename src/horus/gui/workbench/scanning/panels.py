#!/usr/bin/python
# -*- coding: utf-8 -*-
#-----------------------------------------------------------------------#
#                                                                       #
# This file is part of the Horus Project                                #
#                                                                       #
# Copyright (C) 2014 Mundo Reader S.L.                                  #
#                                                                       #
# Date: October 2014                                                    #
# Author: Jesús Arroyo Torrens <jesus.arroyo@bq.com>                    #
#                                                                       #
# This program is free software: you can redistribute it and/or modify  #
# it under the terms of the GNU General Public License as published by  #
# the Free Software Foundation, either version 3 of the License, or     #
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
__license__ = "GNU General Public License v3 http://www.gnu.org/licenses/gpl.html"


import wx._core

from horus.gui.util.itemControls import *

from horus.util import profile

from horus.engine.driver import Driver
from horus.engine.scan import PointCloudGenerator

class SettingsPanel(wx.Panel):
    def __init__(self, parent):
        """"""
        wx.Panel.__init__(self, parent=parent, size=(275, 0))
        self.initialize()

    def initialize(self):
        self.driver = Driver.Instance()
        self.pcg = PointCloudGenerator.Instance()
        self.main = self.GetParent().GetParent().GetParent()

        if hasattr(self, 'controls'):
            del self.controls[:]
        self.controls = []

        #-- Graphic elements
        control = Control(self, _('Scanner Settings'))
        control.append(Slider, 'brightness_scanning', self.driver.camera.setBrightness)
        control.append(Slider, 'contrast_scanning', self.driver.camera.setContrast)
        control.append(Slider, 'saturation_scanning', self.driver.camera.setSaturation)
        control.append(Slider, 'exposure_scanning', self.driver.camera.setExposure)
        control.append(ComboBox, 'framerate_scanning', lambda v: (self.driver.camera.setFrameRate(int(v)), self.reloadVideo()))
        control.append(ComboBox, 'resolution_scanning', lambda v: self.driver.camera.setResolution(int(v.split('x')[0]), int(v.split('x')[1])))
        control.append(CheckBox, 'use_distortion_scanning', lambda v: (self.driver.camera.setUseDistortion(v), self.reloadVideo()))
        control.append(Button, 'restore_default', self.restoreDefault)
        self.controls.append(control)
        
        control = Control(self, _('Image Processing'), bold=False)
        control.append(CheckBox, 'use_open', lambda v: self.pcg.setUseOpen(bool(v)))
        control.append(Slider, 'open_value', lambda v: self.pcg.setOpenValue(int(v)))
        control.append(CheckBox, 'use_threshold', lambda v: self.pcg.setUseThreshold(bool(v)))
        control.append(Slider, 'threshold_value', lambda v: self.pcg.setThresholdValue(int(v)))
        self.controls.append(control)

        control = Control(self, _('Algorithm'), bold=False)
        control.append(CheckBox, 'use_compact', lambda v: self.pcg.setUseCompact(bool(v)))
        self.controls.append(control)

        control = Control(self, _('3D ROI'), bold=False)
        control.append(CheckBox, 'view_roi', lambda v: self.main.sceneView.QueueRefresh())
        control.append(Slider, 'roi_radius', lambda v: (self.pcg.setROIRadius(int(v)), self.main.sceneView.QueueRefresh()))
        control.append(Slider, 'roi_height', lambda v: (self.pcg.setROIHeight(int(v)), self.main.sceneView.QueueRefresh()))
        self.controls.append(control)

        control = Control(self, _('Laser'), bold=False)
        control.append(ComboBox, 'use_laser', lambda v: self.pcg.setUseLaser(v==_("Use Left Laser"), v==_("Use Right Laser"))) #TODO: use combo choices
        self.controls.append(control)

        control = Control(self, _('Motor'), bold=False)
        control.append(TextBox, 'step_degrees_scanning', lambda v: (self.driver.board.setRelativePosition(self.getValueFloat(v)), self.pcg.setDegrees(self.getValueFloat(v))))
        control.append(TextBox, 'feed_rate_scanning', lambda v: self.driver.board.setSpeedMotor(self.getValueInteger(v)))
        control.append(TextBox, 'acceleration_scanning', lambda v: self.driver.board.setAccelerationMotor(self.getValueInteger(v)))
        control.append(Button, 'restore_default', self.restoreDefault)
        self.controls.append(control)

        # - Layout
        vbox = wx.BoxSizer(wx.VERTICAL)
        for control in self.controls:
            vbox.Add(control, 0, wx.ALL|wx.EXPAND, 0)
        self.SetSizer(vbox)
        self.Layout()

        #-- Callbacks
        for control in self.controls:
            control.setUndoCallbacks(self.main.appendToUndo, self.main.releaseUndo)

    #TODO: move
    def getValueInteger(self, value):
        try:
            return int(eval(value, {}, {}))
        except:
            return 0

    def getValueFloat(self, value): 
        try:
            return float(eval(value.replace(',', '.'), {}, {}))
        except:
            return 0.0

    def restoreDefault(self):
        dlg = wx.MessageDialog(self, _("This will reset scanner settings to defaults.\nUnless you have saved your current profile, all settings will be lost!\nDo you really want to reset?"), _("Scanner Settings reset"), wx.YES_NO | wx.ICON_QUESTION)
        result = dlg.ShowModal() == wx.ID_YES
        dlg.Destroy()
        if result:
            for control in self.controls:
                control.resetProfile()
            self.main.enableLabelTool(self.main.undoTool, False)
            self.reloadVideo()

    def reloadVideo(self):
    	self.main.videoView.stop()
        if self.main.playing:
            self.main.videoView.play()

    def updateProfileToAllControls(self):
        for control in self.controls:
            control.updateProfile()