#!/usr/bin/python
# -*- coding: utf-8 -*-
#-----------------------------------------------------------------------#
#                                                                       #
# This file is part of the Horus Project                                #
#                                                                       #
# Copyright (C) 2014-2015 Mundo Reader S.L.                             #
#                                                                       #
# Date: August, November 2014                                           #
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

from horus.gui.util.customPanels import ExpandablePanel, SectionItem, Slider, ComboBox, \
                                        CheckBox, Button, TextBox, ToggleButton, CallbackButton

from horus.util import profile

from horus.engine.driver import Driver


class CameraControl(ExpandablePanel):
    """"""
    def __init__(self, parent):
        """"""
        ExpandablePanel.__init__(self, parent, _("Camera Control"))
        
        self.driver = Driver.Instance()
        self.main = self.GetParent().GetParent().GetParent().GetParent()

        self.initialize()
        
    def initialize(self):
        self.clearSections()
        section = self.createSection('camera_control')
        section.addItem(Slider, 'brightness_control', self.driver.camera.setBrightness)
        section.addItem(Slider, 'contrast_control', self.driver.camera.setContrast)
        section.addItem(Slider, 'saturation_control', self.driver.camera.setSaturation)
        section.addItem(Slider, 'exposure_control', self.driver.camera.setExposure)
        section.addItem(ComboBox, 'framerate_control', lambda v: self.driver.camera.setFrameRate(int(v)))
        section.addItem(ComboBox, 'resolution_control', lambda v: self.driver.camera.setResolution(int(v.split('x')[0]), int(v.split('x')[1])))
        section.addItem(CheckBox, 'use_distortion_control', self.driver.camera.setUseDistortion, tooltip=_("Use the distortion vector to remove the distortion caused by the camera from the image. This process slows the video feed from the camera."))


class LaserControl(ExpandablePanel):
    """"""
    def __init__(self, parent):
        """"""
        ExpandablePanel.__init__(self, parent, _("Laser Control"), hasUndo=False, hasRestore=False)
        
        self.driver = Driver.Instance()

        self.initialize()
        
    def initialize(self):
        self.clearSections()
        section = self.createSection('laser_control')
        section.addItem(ToggleButton, 'left_button', (self.driver.board.setLeftLaserOn, self.driver.board.setLeftLaserOff))
        section.addItem(ToggleButton, 'right_button', (self.driver.board.setRightLaserOn, self.driver.board.setRightLaserOff))

class LDRControl(ExpandablePanel):
    """"""
    def __init__(self, parent):
        """"""
        ExpandablePanel.__init__(self, parent, _("LDR Value"), hasUndo=False, hasRestore=False)
        
        self.driver = Driver.Instance()

        self.initialize()
        
    def initialize(self):
        self.clearSections()
        section = self.createSection('ldr_control')
        section.addItem(LDRSection, 'ldr_value', lambda v, c: self.driver.board.sendRequest(v, nonblocking=True, callback=c, readLines=True))



class MotorControl(ExpandablePanel):
    """"""
    def __init__(self, parent):
        """"""
        ExpandablePanel.__init__(self, parent, _("Motor Control"), hasUndo=False)
        
        self.driver = Driver.Instance()

        self.initialize()
        
    def initialize(self):
        self.clearSections()
        section = self.createSection('motor_control')
        section.addItem(TextBox, 'step_degrees_control', lambda v: self.driver.board.setRelativePosition(self.getValueFloat(v)))
        section.addItem(TextBox, 'feed_rate_control', lambda v: self.driver.board.setSpeedMotor(self.getValueInteger(v)))
        section.addItem(TextBox, 'acceleration_control', lambda v: self.driver.board.setAccelerationMotor(self.getValueInteger(v)))
        section.addItem(CallbackButton, 'move_button', lambda c: self.driver.board.moveMotor(nonblocking=True, callback=c))
        section.addItem(ToggleButton, 'enable_button', (self.driver.board.enableMotor, self.driver.board.disableMotor))

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


class GcodeControl(ExpandablePanel):
    """"""
    def __init__(self, parent):
        """"""
        ExpandablePanel.__init__(self, parent, _("Gcode Control"), hasUndo=False, hasRestore=False)
        
        self.driver = Driver.Instance()

        self.initialize()
        
    def initialize(self):
        self.clearSections()
        section = self.createSection('gcode_control')
        section.addItem(GcodeSection, 'gcode_gui', lambda v, c: self.driver.board.sendRequest(v, nonblocking=True, callback=c, readLines=True))


class GcodeSection(SectionItem):
    def __init__(self, parent, name, engineCallback=None):
        """"""
        SectionItem.__init__(self, parent, name, engineCallback)

        #-- Elements
        self.request = wx.TextCtrl(self, size=(10,10))
        self.control = wx.Button(self, label=self.setting.getLabel())
        self.response = wx.TextCtrl(self, size=(10,250), style=wx.TE_MULTILINE)

        #-- Layout
        vbox =wx.BoxSizer(wx.VERTICAL)
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.Add(self.request, 1, wx.ALL^wx.RIGHT^wx.LEFT|wx.EXPAND, 12)
        hbox.Add(self.control, 0, wx.ALL|wx.EXPAND, 12)
        vbox.Add(hbox, 0, wx.ALL|wx.EXPAND, 0)
        vbox.Add(self.response, 1, wx.ALL^wx.LEFT|wx.EXPAND, 12)
        self.SetSizer(vbox)
        self.Layout()

        #-- Events
        self.control.Bind(wx.EVT_BUTTON, self.onButtonClicked)

    def onButtonClicked(self, event):
        self.control.Disable()
        self.waitCursor = wx.BusyCursor()
        if self.engineCallback is not None:
            ret = self.engineCallback(str(self.request.GetValue()), self.onFinishCallback)

    def onFinishCallback(self, ret):
        wx.CallAfter(self.control.Enable)
        if ret is not None:
            wx.CallAfter(lambda: self.response.SetValue(ret))
        if hasattr(self,'waitCursor'):
            del self.waitCursor

    def updateProfile(self):
        if hasattr(self,'control'):
            self.update(None)

    def update(self, value):
        if self.isVisible():
            self.Show()
        else:
            self.Hide()

class LDRSection(SectionItem):
    def __init__(self, parent, name, engineCallback=None):
        """"""
        SectionItem.__init__(self, parent, name, engineCallback)

        #-- Elements
        self.LDR0 = wx.Button(self, label='LDR 0')
        self.LDR1 = wx.Button(self, label='LDR 1')

        self.response = wx.TextCtrl(self, size=(10,250), style=wx.TE_MULTILINE)

        #-- Layout
        vbox =wx.BoxSizer(wx.VERTICAL)
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.Add(self.LDR0, 0, wx.ALL|wx.EXPAND, 12)
        hbox.Add(self.LDR1, 0, wx.ALL|wx.EXPAND, 12)
        vbox.Add(hbox, 0, wx.ALL|wx.EXPAND, 0)
        vbox.Add(self.response, 1, wx.ALL^wx.LEFT|wx.EXPAND, 12)
        self.SetSizer(vbox)
        self.Layout()

        #-- Events
        self.LDR0.Bind(wx.EVT_BUTTON, self.onButton0Clicked)
        self.LDR1.Bind(wx.EVT_BUTTON, self.onButton1Clicked)


    def onButton0Clicked(self, event):
        self.LDR0.Disable()
        self.waitCursor = wx.BusyCursor()
        if self.engineCallback is not None:
            ret = self.engineCallback('M50 T0', self.onFinishCallback)

    def onButton1Clicked(self, event):
        self.LDR1.Disable()
        self.waitCursor = wx.BusyCursor()
        if self.engineCallback is not None:
            ret = self.engineCallback('M50 T1', self.onFinishCallback)

    def onFinishCallback(self, ret):
        wx.CallAfter(self.LDR0.Enable)
        wx.CallAfter(self.LDR1.Enable)

        if ret is not None:
            wx.CallAfter(lambda: self.response.SetValue(ret))
        if hasattr(self,'waitCursor'):
            del self.waitCursor

    def updateProfile(self):
        if hasattr(self,'control'):
            self.update(None)

    def update(self, value):
        if self.isVisible():
            self.Show()
        else:
            self.Hide()