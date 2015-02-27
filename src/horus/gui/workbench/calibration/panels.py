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

import wx._core
import numpy as np

from horus.gui.util.customPanels import ExpandablePanel, Slider, ComboBox, \
                                        CheckBox, ToggleButton, Button, TextBox

from horus.util import profile

from horus.engine.driver import Driver
from horus.engine import scan, calibration

class PatternSettingsPanel(ExpandablePanel):
    def __init__(self, parent):
        """"""
        ExpandablePanel.__init__(self, parent, _("Pattern Settings"))

        self.cameraIntrinsics = calibration.CameraIntrinsics.Instance()
        self.simpleLaserTriangulation = calibration.SimpleLaserTriangulation.Instance()
        self.laserTriangulation = calibration.LaserTriangulation.Instance()
        self.platformExtrinsics = calibration.PlatformExtrinsics.Instance()

        self.initialize()

    def initialize(self):
        self.clearSections()
        section = self.createSection('pattern_settings')
        section.addItem(TextBox, 'square_width', lambda v: self.updatePatternParameters())
        section.addItem(TextBox, 'pattern_rows', lambda v: self.updatePatternParameters())
        section.addItem(TextBox, 'pattern_columns', lambda v: self.updatePatternParameters())
        section.addItem(TextBox, 'pattern_distance', lambda v: self.updatePatternParameters(), tooltip=_('Distance between the upper edge of the chess row closer to the platform and the platform.'))

    def updatePatternParameters(self):
        self.cameraIntrinsics.setPatternParameters(profile.getProfileSettingInteger('pattern_rows'),
                                                   profile.getProfileSettingInteger('pattern_columns'),
                                                   profile.getProfileSettingInteger('square_width'),
                                                   profile.getProfileSettingFloat('pattern_distance'))

        self.simpleLaserTriangulation.setPatternParameters(profile.getProfileSettingInteger('pattern_rows'),
                                                           profile.getProfileSettingInteger('pattern_columns'),
                                                           profile.getProfileSettingInteger('square_width'),
                                                           profile.getProfileSettingFloat('pattern_distance'))

        self.laserTriangulation.setPatternParameters(profile.getProfileSettingInteger('pattern_rows'),
                                                     profile.getProfileSettingInteger('pattern_columns'),
                                                     profile.getProfileSettingInteger('square_width'),
                                                     profile.getProfileSettingFloat('pattern_distance'))
        self.platformExtrinsics.setPatternParameters(profile.getProfileSettingInteger('pattern_rows'),
                                                     profile.getProfileSettingInteger('pattern_columns'),
                                                     profile.getProfileSettingInteger('square_width'),
                                                     profile.getProfileSettingFloat('pattern_distance'))

class CameraSettingsPanel(ExpandablePanel):
    def __init__(self, parent):
        """"""
        ExpandablePanel.__init__(self, parent, _("Camera Settings"))

        self.driver = Driver.Instance()
        self.main = self.GetParent().GetParent().GetParent().GetParent()

        self.initialize()

    def initialize(self):
        self.clearSections()
        section = self.createSection('camera_settings')
        section.addItem(Slider, 'brightness_calibration', self.driver.camera.setBrightness)
        section.addItem(Slider, 'contrast_calibration', self.driver.camera.setContrast)
        section.addItem(Slider, 'saturation_calibration', self.driver.camera.setSaturation)
        section.addItem(Slider, 'exposure_calibration', self.driver.camera.setExposure)
        section.addItem(ComboBox, 'framerate_calibration', lambda v: self.driver.camera.setFrameRate(int(v)))
        section.addItem(ComboBox, 'resolution_calibration', lambda v: self.driver.camera.setResolution(int(v.split('x')[0]), int(v.split('x')[1])))
        section.addItem(CheckBox, 'use_distortion_calibration', self.driver.camera.setUseDistortion, tooltip=_("Use the distortion vector to remove the distortion caused by the camera from the image. This process slows the video feed from the camera."))

class LaserSettingsPanel(ExpandablePanel):
    def __init__(self, parent):
        """"""
        ExpandablePanel.__init__(self, parent, _("Laser Settings"))

        self.driver = Driver.Instance()
        self.laserTriangulation = calibration.LaserTriangulation.Instance()

        self.initialize()

    def initialize(self):
        self.clearSections()
        section = self.createSection('laser_settings')
        # section.addItem(Slider, 'laser_threshold_value', self.laserTriangulation.setThreshold, tooltip=_("Value that determines wich pixels belong to the laser and which to the rest of the image. A low value may cause noise errors on the final scanned model, and a high value may reduce its accuracy."))
        section.addItem(ToggleButton, 'left_button', (self.driver.board.setLeftLaserOn, self.driver.board.setLeftLaserOff))
        section.addItem(ToggleButton, 'right_button', (self.driver.board.setRightLaserOn, self.driver.board.setRightLaserOff))

## TODO: Use TextBoxArray

class CalibrationPanel(ExpandablePanel):

    def __init__(self, parent, titleText="Workbench", buttonStartCallback=None, description="_(Workbench description)"):
        ExpandablePanel.__init__(self, parent, titleText, hasUndo=False, hasRestore=False)

        self.buttonStartCallback = buttonStartCallback

        self.parametersBox = wx.BoxSizer(wx.VERTICAL)
        self.buttonsPanel = wx.Panel(self.content)
        self.SetToolTip(wx.ToolTip(description))


        self.buttonEdit = wx.ToggleButton(self.buttonsPanel, wx.NewId(), label=_("Edit"))
        self.buttonEdit.SetMinSize((0,-1))
        self.buttonDefault = wx.Button(self.buttonsPanel, wx.NewId(), label=_("Default"))
        self.buttonDefault.SetMinSize((0,-1))
        self.buttonStart = wx.Button(self.buttonsPanel, wx.NewId(), label=_("Start"))
        self.buttonStart.SetMinSize((0,-1))

        self.contentBox.Add(self.parametersBox, 1, wx.TOP|wx.BOTTOM|wx.EXPAND, 5)
        self.hbox = wx.BoxSizer(wx.HORIZONTAL)
        self.hbox.Add(self.buttonEdit, 2, wx.ALL|wx.EXPAND, 3)
        self.hbox.Add(self.buttonDefault, 2, wx.ALL|wx.EXPAND, 3)
        self.hbox.Add(self.buttonStart, 3, wx.ALL|wx.EXPAND, 3)
        self.buttonsPanel.SetSizer(self.hbox)
        self.contentBox.Add(self.buttonsPanel, 0, wx.ALL|wx.EXPAND, 3)

        self.buttonEdit.Bind(wx.EVT_TOGGLEBUTTON, self.onButtonEditPressed)
        self.buttonDefault.Bind(wx.EVT_BUTTON, self.onButtonDefaultPressed)
        self.buttonStart.Bind(wx.EVT_BUTTON, self.onButtonStartPressed)

        self.Layout()

    def onButtonEditPressed(self, event):
        pass

    def onButtonDefaultPressed(self, event):
        pass

    def onButtonStartPressed(self, event):
        if self.buttonStartCallback is not None:
            self.buttonStartCallback()


class CameraIntrinsicsPanel(CalibrationPanel):

    def __init__(self, parent, buttonStartCallback):
        CalibrationPanel.__init__(self, parent, titleText=_("Camera Intrinsics"), buttonStartCallback=buttonStartCallback,
                                  description=_("Determines the camera matrix and the distortion coefficients using Zhang2000 algorithm and pinhole camera model."))

        self.driver = Driver.Instance()
        self.pcg = scan.PointCloudGenerator.Instance()
        self.cameraIntrinsics = calibration.CameraIntrinsics.Instance()
        self.laserTriangulation = calibration.LaserTriangulation.Instance()
        self.platformExtrinsics = calibration.PlatformExtrinsics.Instance()

        cameraPanel = wx.Panel(self.content)
        distortionPanel = wx.Panel(self.content)

        cameraText = wx.StaticText(self.content, label=_("Camera matrix"))
        cameraText.SetFont((wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.NORMAL, wx.FONTWEIGHT_NORMAL)))

        self.cameraTexts = [[0.0 for j in range(3)] for i in range(3)]
        self.cameraValues = np.zeros((3,3))

        cameraBox = wx.BoxSizer(wx.VERTICAL)
        cameraPanel.SetSizer(cameraBox)
        for i in range(3):
            ibox = wx.BoxSizer(wx.HORIZONTAL)
            for j in range(3):
                jbox = wx.BoxSizer(wx.VERTICAL)
                self.cameraTexts[i][j] = wx.TextCtrl(cameraPanel, wx.ID_ANY, "")
                self.cameraTexts[i][j].SetMinSize((0,-1))
                self.cameraTexts[i][j].SetEditable(False)
                self.cameraTexts[i][j].Disable()
                jbox.Add(self.cameraTexts[i][j], 1, wx.ALL|wx.EXPAND, 2)
                ibox.Add(jbox, 1, wx.ALL|wx.EXPAND, 2)
            cameraBox.Add(ibox, 1, wx.ALL|wx.EXPAND, 2)

        distortionText = wx.StaticText(self.content, label=_("Distortion vector"))
        distortionText.SetFont((wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.NORMAL, wx.FONTWEIGHT_NORMAL)))

        self.distortionTexts = [0]*5
        self.distortionValues = np.zeros(5)

        distortionBox = wx.BoxSizer(wx.HORIZONTAL)
        distortionPanel.SetSizer(distortionBox)
        for i in range(5):
            ibox = wx.BoxSizer(wx.HORIZONTAL)
            self.distortionTexts[i] = wx.TextCtrl(distortionPanel, wx.ID_ANY, "")
            self.distortionTexts[i].SetMinSize((0,-1))
            self.distortionTexts[i].SetEditable(False)
            self.distortionTexts[i].Disable()
            ibox.Add(self.distortionTexts[i], 1, wx.ALL|wx.EXPAND, 2)
            distortionBox.Add(ibox, 1, wx.ALL|wx.EXPAND, 2)

        self.parametersBox.Add(cameraText, 0, wx.ALL|wx.EXPAND, 8)
       	self.parametersBox.Add(cameraPanel, 0, wx.ALL|wx.EXPAND, 2)
        self.parametersBox.Add(distortionText, 0, wx.ALL|wx.EXPAND, 8)
       	self.parametersBox.Add(distortionPanel, 0, wx.ALL|wx.EXPAND, 2)

        self.Layout()

    def onButtonEditPressed(self, event):
        enable = self.buttonEdit.GetValue()

        for i in range(3):
            for j in range(3):
                self.cameraTexts[i][j].SetEditable(enable)
                if enable:
                    self.cameraTexts[i][j].Enable()
                else:
                    self.cameraTexts[i][j].Disable()
                    self.cameraValues[i][j] = self.getValueFloat(self.cameraTexts[i][j].GetValue())

        for i in range(5):
            self.distortionTexts[i].SetEditable(enable)
            if enable:
                self.distortionTexts[i].Enable()
            else:
                self.distortionTexts[i].Disable()
                self.distortionValues[i] = self.getValueFloat(self.distortionTexts[i].GetValue())

        if enable:
            self.buttonEdit.SetLabel(_("OK"))
        else:
            self.buttonEdit.SetLabel(_("Edit"))

    def onButtonDefaultPressed(self, event):
        dlg = wx.MessageDialog(self, _("This will reset camera intrinsics profile settings to defaults.\nUnless you have saved your current profile, all settings will be lost! Do you really want to reset?"), _("Camera Intrinsics reset"), wx.YES_NO | wx.ICON_QUESTION)
        result = dlg.ShowModal() == wx.ID_YES
        dlg.Destroy()
        if result:
            profile.resetProfileSetting('camera_matrix')
            profile.resetProfileSetting('distortion_vector')
            self.updateProfile()

    def getParameters(self):
        return self.cameraValues, self.distortionValues

    def setParameters(self, params):
        self.cameraValues = params[0]
        self.distortionValues = params[1]
        self.updateAllControls()

    def getProfileSettings(self):
        self.cameraValues = profile.getProfileSettingNumpy('camera_matrix')
        self.distortionValues = profile.getProfileSettingNumpy('distortion_vector')

    def putProfileSettings(self):
        profile.putProfileSettingNumpy('camera_matrix', self.cameraValues)
        profile.putProfileSettingNumpy('distortion_vector', self.distortionValues)

    def updateAllControls(self):
        for i in range(3):
            for j in range(3):
                self.cameraValues[i][j] = round(self.cameraValues[i][j], 3)
                self.cameraTexts[i][j].SetValue(str(self.cameraValues[i][j]))
        for i in range(5):
            self.distortionValues[i] = round(self.distortionValues[i], 4)
            self.distortionTexts[i].SetValue(str(self.distortionValues[i]))

    def updateEngine(self):
        if hasattr(self, 'driver'):
            self.driver.camera.setIntrinsics(self.cameraValues, self.distortionValues)
        if hasattr(self, 'cameraIntrinsics'):
            self.cameraIntrinsics.setIntrinsics(self.cameraValues, self.distortionValues)
        if hasattr(self, 'laserTriangulation'):
            self.laserTriangulation.setIntrinsics(self.cameraValues, self.distortionValues)
        if hasattr(self, 'platformExtrinsics'):
            self.platformExtrinsics.setIntrinsics(self.cameraValues, self.distortionValues)
        if hasattr(self, 'pcg'):
            self.pcg.setCameraIntrinsics(self.cameraValues, self.distortionValues)

    def updateProfile(self):
        self.getProfileSettings()
        self.updateAllControls()
        self.updateEngine()

    def updateAllControlsToProfile(self):
        self.putProfileSettings()
        self.updateEngine()

    #TODO: move
    def getValueFloat(self, value): 
        try:
            return float(eval(value.replace(',', '.'), {}, {}))
        except:
            return 0.0


class LaserTriangulationPanel(CalibrationPanel):

    def __init__(self, parent, buttonStartCallback):
        CalibrationPanel.__init__(self, parent, titleText=_("Laser Triangulation"), buttonStartCallback=buttonStartCallback,
                                  description=_("Determines the planes of both line lasers: minimum distance and normal vector."))

        self.pcg = scan.PointCloudGenerator.Instance()

        distanceLeftPanel = wx.Panel(self.content)
        normalLeftPanel = wx.Panel(self.content)
        distanceRightPanel = wx.Panel(self.content)
        normalRightPanel = wx.Panel(self.content)

        laserLeftText = wx.StaticText(self.content, label=_("Left Laser Plane"))
        laserLeftText.SetFont((wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.NORMAL, wx.FONTWEIGHT_NORMAL)))

        self.distanceLeftValue = 0

        distanceLeftBox = wx.BoxSizer(wx.HORIZONTAL)
        distanceLeftPanel.SetSizer(distanceLeftBox)
        self.distanceLeftText = wx.TextCtrl(distanceLeftPanel, wx.ID_ANY, "")
        self.distanceLeftText.SetMinSize((0,-1))
        self.distanceLeftText.SetEditable(False)
        self.distanceLeftText.Disable()
        distanceLeftBox.Add(self.distanceLeftText, 1, wx.ALL|wx.EXPAND, 4)

        self.normalLeftTexts = [0]*3
        self.normalLeftValues = np.zeros(3)

        normalLeftBox = wx.BoxSizer(wx.HORIZONTAL)
        normalLeftPanel.SetSizer(normalLeftBox)
        for i in range(3):
            ibox = wx.BoxSizer(wx.HORIZONTAL)
            self.normalLeftTexts[i] = wx.TextCtrl(normalLeftPanel, wx.ID_ANY, "")
            self.normalLeftTexts[i].SetMinSize((0,-1))
            self.normalLeftTexts[i].SetEditable(False)
            self.normalLeftTexts[i].Disable()
            ibox.Add(self.normalLeftTexts[i], 1, wx.ALL|wx.EXPAND, 2)
            normalLeftBox.Add(ibox, 1, wx.ALL|wx.EXPAND, 2)

        laserRightText = wx.StaticText(self.content, label=_("Right Laser Plane"))
        laserRightText.SetFont((wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.NORMAL, wx.FONTWEIGHT_NORMAL)))

        self.distanceRightValue = 0

        distanceRightBox = wx.BoxSizer(wx.HORIZONTAL)
        distanceRightPanel.SetSizer(distanceRightBox)
        self.distanceRightText = wx.TextCtrl(distanceRightPanel, wx.ID_ANY, "")
        self.distanceRightText.SetMinSize((0,-1))
        self.distanceRightText.SetEditable(False)
        self.distanceRightText.Disable()
        distanceRightBox.Add(self.distanceRightText, 1, wx.ALL|wx.EXPAND, 4)

        self.normalRightTexts = [0]*3
        self.normalRightValues = np.zeros(3)

        normalRightBox = wx.BoxSizer(wx.HORIZONTAL)
        normalRightPanel.SetSizer(normalRightBox)
        for i in range(3):
            ibox = wx.BoxSizer(wx.HORIZONTAL)
            self.normalRightTexts[i] = wx.TextCtrl(normalRightPanel, wx.ID_ANY, "")
            self.normalRightTexts[i].SetMinSize((0,-1))
            self.normalRightTexts[i].SetEditable(False)
            self.normalRightTexts[i].Disable()
            ibox.Add(self.normalRightTexts[i], 1, wx.ALL|wx.EXPAND, 2)
            normalRightBox.Add(ibox, 1, wx.ALL|wx.EXPAND, 2)

        self.parametersBox.Add(laserLeftText, 0, wx.ALL|wx.EXPAND, 8)
        self.parametersBox.Add(distanceLeftPanel, 1, wx.ALL|wx.EXPAND, 2)
        self.parametersBox.Add(normalLeftPanel, 0, wx.ALL|wx.EXPAND, 2)
        self.parametersBox.Add(laserRightText, 0, wx.ALL|wx.EXPAND, 8)
        self.parametersBox.Add(distanceRightPanel, 1, wx.ALL|wx.EXPAND, 2)
        self.parametersBox.Add(normalRightPanel, 0, wx.ALL|wx.EXPAND, 2)

        self.Layout()

    def onButtonEditPressed(self, event):
        enable = self.buttonEdit.GetValue()

        self.distanceLeftText.SetEditable(enable)
        if enable:
            self.distanceLeftText.Enable()
        else:
            self.distanceLeftText.Disable()
            self.distanceLeftValue = self.getValueFloat(self.distanceLeftText.GetValue())

        for i in range(3):
            self.normalLeftTexts[i].SetEditable(enable)
            if enable:
                self.normalLeftTexts[i].Enable()
            else:
                self.normalLeftTexts[i].Disable()
                self.normalLeftValues[i] = self.getValueFloat(self.normalLeftTexts[i].GetValue())

        self.distanceRightText.SetEditable(enable)
        if enable:
            self.distanceRightText.Enable()
        else:
            self.distanceRightText.Disable()
            self.distanceRightValue = self.getValueFloat(self.distanceRightText.GetValue())

        for i in range(3):
            self.normalRightTexts[i].SetEditable(enable)
            if enable:
                self.normalRightTexts[i].Enable()
            else:
                self.normalRightTexts[i].Disable()
                self.normalRightValues[i] = self.getValueFloat(self.normalRightTexts[i].GetValue())

        if enable:
            self.buttonEdit.SetLabel(_("OK"))
        else:
            self.buttonEdit.SetLabel(_("Edit"))
            self.updateAllControlsToProfile()

    def onButtonDefaultPressed(self, event):
        dlg = wx.MessageDialog(self, _("This will reset laser triangulation profile settings to defaults.\nUnless you have saved your current profile, all settings will be lost! Do you really want to reset?"), _("Laser Triangulation reset"), wx.YES_NO | wx.ICON_QUESTION)
        result = dlg.ShowModal() == wx.ID_YES
        dlg.Destroy()
        if result:
            profile.resetProfileSetting('distance_left')
            profile.resetProfileSetting('normal_left')
            profile.resetProfileSetting('distance_right')
            profile.resetProfileSetting('normal_right')
            self.updateProfile()

    def getParameters(self):
        return self.distanceLeftValue, self.normalLeftValues, self.distanceLeftValue, self.normalRightValues

    def setParameters(self, params):
        self.distanceLeftValue = params[0]
        self.normalLeftValues = params[1]
        self.distanceRightValue = params[2]
        self.normalRightValues = params[3]
        self.updateAllControls()

    def getProfileSettings(self):
        self.distanceLeftValue = profile.getProfileSettingFloat('distance_left')
        self.normalLeftValues = profile.getProfileSettingNumpy('normal_left')
        self.distanceRightValue = profile.getProfileSettingFloat('distance_right')
        self.normalRightValues = profile.getProfileSettingNumpy('normal_right')

    def putProfileSettings(self):
        profile.putProfileSettingNumpy('distance_left', self.distanceLeftValue)
        profile.putProfileSettingNumpy('normal_left', self.normalLeftValues)
        profile.putProfileSettingNumpy('distance_right', self.distanceRightValue)
        profile.putProfileSettingNumpy('normal_right', self.normalRightValues)

    def updateAllControls(self):
        self.distanceLeftValue = round(self.distanceLeftValue, 6)
        self.distanceLeftText.SetValue(str(self.distanceLeftValue))

        for i in range(3):
            self.normalLeftValues[i] = round(self.normalLeftValues[i], 6)
            self.normalLeftTexts[i].SetValue(str(self.normalLeftValues[i]))

        self.distanceRightValue = round(self.distanceRightValue, 6)
        self.distanceRightText.SetValue(str(self.distanceRightValue))

        for i in range(3):
            self.normalRightValues[i] = round(self.normalRightValues[i], 6)
            self.normalRightTexts[i].SetValue(str(self.normalRightValues[i]))

    def updateEngine(self):
        if hasattr(self, 'pcg'):
            self.pcg.setLaserTriangulation(self.distanceLeftValue, self.normalLeftValues, self.distanceRightValue, self.normalRightValues)

    def updateProfile(self):
        self.getProfileSettings()
        self.updateAllControls()
        self.updateEngine()

    def updateAllControlsToProfile(self):
        self.putProfileSettings()
        self.updateEngine()

    #TODO: move
    def getValueFloat(self, value): 
        try:
            return float(eval(value.replace(',', '.'), {}, {}))
        except:
            return 0.0


class SimpleLaserTriangulationPanel(CalibrationPanel):

    def __init__(self, parent, buttonStartCallback):

        CalibrationPanel.__init__(self, parent, titleText=_("Laser Simple Triangulation Calibration"), buttonStartCallback=buttonStartCallback,
                                  description=_("Determines the depth of the intersection camera-laser considering the inclination of the lasers."))

        self.pcg = scan.PointCloudGenerator.Instance()

        coordinatesPanel = wx.Panel(self.content)
        originPanel = wx.Panel(self.content)
        normalPanel = wx.Panel(self.content)

        coordinatesText = wx.StaticText(self.content, label=_("Coordinates"))
        coordinatesText.SetFont((wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.NORMAL, wx.FONTWEIGHT_NORMAL)))

        self.coordinatesTexts = [[0 for j in range(2)] for i in range(2)]
        self.coordinatesValues = np.zeros((2,2))

        coordinatesBox = wx.BoxSizer(wx.VERTICAL)
        coordinatesPanel.SetSizer(coordinatesBox)
        for i in range(2):
            ibox = wx.BoxSizer(wx.HORIZONTAL)
            for j in range(2):
                jbox = wx.BoxSizer(wx.VERTICAL)
                self.coordinatesTexts[i][j] = wx.TextCtrl(coordinatesPanel, wx.ID_ANY, "")
                self.coordinatesTexts[i][j].SetMinSize((0,-1))
                self.coordinatesTexts[i][j].SetEditable(False)
                self.coordinatesTexts[i][j].Disable()
                jbox.Add(self.coordinatesTexts[i][j], 1, wx.ALL|wx.EXPAND, 2)
                ibox.Add(jbox, 1, wx.ALL|wx.EXPAND, 2)
            coordinatesBox.Add(ibox, 1, wx.ALL|wx.EXPAND, 2)

        originText = wx.StaticText(self.content, label=_("Origin"))
        originText.SetFont((wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.NORMAL, wx.FONTWEIGHT_NORMAL)))

        self.originTexts = [0]*3
        self.originValues = np.zeros(3)

        originBox = wx.BoxSizer(wx.HORIZONTAL)
        originPanel.SetSizer(originBox)
        for i in range(3):
            ibox = wx.BoxSizer(wx.HORIZONTAL)
            self.originTexts[i] = wx.TextCtrl(originPanel, wx.ID_ANY, "")
            self.originTexts[i].SetMinSize((0,-1))
            self.originTexts[i].SetEditable(False)
            self.originTexts[i].Disable()
            ibox.Add(self.originTexts[i], 1, wx.ALL|wx.EXPAND, 2)
            originBox.Add(ibox, 1, wx.ALL|wx.EXPAND, 2)

        normalText = wx.StaticText(self.content, label=_("Normal"))
        normalText.SetFont((wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.NORMAL, wx.FONTWEIGHT_NORMAL)))

        self.normalTexts = [0]*3
        self.normalValues = np.zeros(3)

        normalBox = wx.BoxSizer(wx.HORIZONTAL)
        normalPanel.SetSizer(normalBox)
        for i in range(3):
            ibox = wx.BoxSizer(wx.HORIZONTAL)
            self.normalTexts[i] = wx.TextCtrl(normalPanel, wx.ID_ANY, "")
            self.normalTexts[i].SetMinSize((0,-1))
            self.normalTexts[i].SetEditable(False)
            self.normalTexts[i].Disable()
            ibox.Add(self.normalTexts[i], 1, wx.ALL|wx.EXPAND, 2)
            normalBox.Add(ibox, 1, wx.ALL|wx.EXPAND, 2)

        self.parametersBox.Add(coordinatesText, 0, wx.ALL|wx.EXPAND, 8)
        self.parametersBox.Add(coordinatesPanel, 0, wx.ALL|wx.EXPAND, 2)
        self.parametersBox.Add(originText, 0, wx.ALL|wx.EXPAND, 8)
        self.parametersBox.Add(originPanel, 0, wx.ALL|wx.EXPAND, 2)
        self.parametersBox.Add(normalText, 0, wx.ALL|wx.EXPAND, 8)
        self.parametersBox.Add(normalPanel, 0, wx.ALL|wx.EXPAND, 2)

        self.Layout()

    def onButtonEditPressed(self, event):
        enable = self.buttonEdit.GetValue()
        for i in range(2):
            for j in range(2):
                self.coordinatesTexts[i][j].SetEditable(enable)
                if enable:
                    self.coordinatesTexts[i][j].Enable()
                else:
                    self.coordinatesTexts[i][j].Disable()
                    self.coordinatesValues[i][j] = self.getValueFloat(self.coordinatesTexts[i][j].GetValue())

        for i in range(3):
            self.originTexts[i].SetEditable(enable)
            if enable:
                self.originTexts[i].Enable()
            else:
                self.originTexts[i].Disable()
                self.originValues[i] = self.getValueFloat(self.originTexts[i].GetValue())

        for i in range(3):
            self.normalTexts[i].SetEditable(enable)
            if enable:
                self.normalTexts[i].Enable()
            else:
                self.normalTexts[i].Disable()
                self.normalValues[i] = self.getValueFloat(self.normalTexts[i].GetValue())

        if enable:
            self.buttonEdit.SetLabel(_("OK"))
        else:
            self.buttonEdit.SetLabel(_("Edit"))
            self.updateAllControlsToProfile()

    def onButtonDefaultPressed(self, event):
        dlg = wx.MessageDialog(self, _("This will reset simple laser triangulation profile settings to defaults.\nUnless you have saved your current profile, all settings will be lost! Do you really want to reset?"), _("Laser Triangulation reset"), wx.YES_NO | wx.ICON_QUESTION)
        result = dlg.ShowModal() == wx.ID_YES
        dlg.Destroy()
        if result:
            profile.resetProfileSetting('laser_coordinates')
            profile.resetProfileSetting('laser_origin')
            profile.resetProfileSetting('laser_normal')
            self.updateProfile()

    def getParameters(self):
        return self.coordinatesValues, self.originValues, self.normalValues

    def setParameters(self, params):
        self.coordinatesValues = params[0]
        self.originValues = params[1]
        self.normalValues = params[2]
        self.updateAllControls()

    def getProfileSettings(self):
        self.coordinatesValues = profile.getProfileSettingNumpy('laser_coordinates')
        self.originValues = profile.getProfileSettingNumpy('laser_origin')
        self.normalValues = profile.getProfileSettingNumpy('laser_normal')

    def putProfileSettings(self):
        profile.putProfileSettingNumpy('laser_coordinates', self.coordinatesValues)
        profile.putProfileSettingNumpy('laser_origin', self.originValues)
        profile.putProfileSettingNumpy('laser_normal', self.normalValues)

    def updateAllControls(self):
        for i in range(2):
            for j in range(2):
                self.coordinatesValues[i][j] = round(self.coordinatesValues[i][j], 3)
                self.coordinatesTexts[i][j].SetValue(str(self.coordinatesValues[i][j]))

        for i in range(3):
            self.originValues[i] = round(self.originValues[i], 4)
            self.originTexts[i].SetValue(str(self.originValues[i]))

        for i in range(3):
            self.normalValues[i] = round(self.normalValues[i], 6)
            self.normalTexts[i].SetValue(str(self.normalValues[i]))

    def updateEngine(self):
        if hasattr(self, 'pcg'):
            pass

    def updateProfile(self):
        self.getProfileSettings()
        self.updateAllControls()
        self.updateEngine()

    def updateAllControlsToProfile(self):
        self.putProfileSettings()
        self.updateEngine()

    #TODO: move
    def getValueFloat(self, value): 
        try:
            return float(eval(value.replace(',', '.'), {}, {}))
        except:
            return 0.0


class PlatformExtrinsicsPanel(CalibrationPanel):

    def __init__(self, parent, buttonStartCallback):
        CalibrationPanel.__init__(self, parent, titleText=_("Platform Extrinsics"), buttonStartCallback=buttonStartCallback,
                                  description=_("Determines the transformation matrix between the camera and the platform using a circular interpolation method."))

        self.pcg = scan.PointCloudGenerator.Instance()

        vbox = wx.BoxSizer(wx.VERTICAL)

        rotationPanel = wx.Panel(self.content)
        translationPanel = wx.Panel(self.content)

        rotationText = wx.StaticText(self.content, label=_("Rotation matrix"))
        rotationText.SetFont((wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.NORMAL, wx.FONTWEIGHT_NORMAL)))

        self.rotationTexts = [[0 for j in range(3)] for i in range(3)]
        self.rotationValues = np.zeros((3,3))

        rotationBox = wx.BoxSizer(wx.VERTICAL)
        rotationPanel.SetSizer(rotationBox)
        for i in range(3):
            ibox = wx.BoxSizer(wx.HORIZONTAL)
            for j in range(3):
                jbox = wx.BoxSizer(wx.VERTICAL)
                self.rotationTexts[i][j] = wx.TextCtrl(rotationPanel, wx.ID_ANY, "")
                self.rotationTexts[i][j].SetMinSize((0,-1))
                self.rotationTexts[i][j].SetEditable(False)
                self.rotationTexts[i][j].Disable()
                jbox.Add(self.rotationTexts[i][j], 1, wx.ALL|wx.EXPAND, 2)
                ibox.Add(jbox, 1, wx.ALL|wx.EXPAND, 2)
            rotationBox.Add(ibox, 1, wx.ALL|wx.EXPAND, 2)

        translationText = wx.StaticText(self.content, label=_("Translation vector"))
        translationText.SetFont((wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.NORMAL, wx.FONTWEIGHT_NORMAL)))

        self.translationTexts = [0]*3
        self.translationValues = np.zeros(3)

        translationBox = wx.BoxSizer(wx.HORIZONTAL)
        translationPanel.SetSizer(translationBox)
        for i in range(3):
            ibox = wx.BoxSizer(wx.HORIZONTAL)
            self.translationTexts[i] = wx.TextCtrl(translationPanel, wx.ID_ANY, "")
            self.translationTexts[i].SetMinSize((0,-1))
            self.translationTexts[i].SetEditable(False)
            self.translationTexts[i].Disable()
            ibox.Add(self.translationTexts[i], 1, wx.ALL|wx.EXPAND, 2)
            translationBox.Add(ibox, 1, wx.ALL|wx.EXPAND, 2)

        self.parametersBox.Add(rotationText, 0, wx.ALL|wx.EXPAND, 8)
        self.parametersBox.Add(rotationPanel, 0, wx.ALL|wx.EXPAND, 2)
        self.parametersBox.Add(translationText, 0, wx.ALL|wx.EXPAND, 8)
        self.parametersBox.Add(translationPanel, 0, wx.ALL|wx.EXPAND, 2)

        self.Layout

    def onButtonEditPressed(self, event):
        enable = self.buttonEdit.GetValue()
        for i in range(3):
            for j in range(3):
                self.rotationTexts[i][j].SetEditable(enable)
                if enable:
                    self.rotationTexts[i][j].Enable()
                else:
                    self.rotationTexts[i][j].Disable()
                    self.rotationValues[i][j] = self.getValueFloat(self.rotationTexts[i][j].GetValue())
        for i in range(3):
            self.translationTexts[i].SetEditable(enable)
            if enable:
                self.translationTexts[i].Enable()
            else:
                self.translationTexts[i].Disable()
                self.translationValues[i] = self.getValueFloat(self.translationTexts[i].GetValue())

        if enable:
            self.buttonEdit.SetLabel(_("OK"))
        else:
            self.buttonEdit.SetLabel(_("Edit"))
            self.updateAllControlsToProfile()

    def onButtonDefaultPressed(self, event):
        dlg = wx.MessageDialog(self, _("This will reset platform extrinsics profile settings to defaults.\nUnless you have saved your current profile, all settings will be lost! Do you really want to reset?"), _("Platform Extrinsics reset"), wx.YES_NO | wx.ICON_QUESTION)
        result = dlg.ShowModal() == wx.ID_YES
        dlg.Destroy()
        if result:
            profile.resetProfileSetting('rotation_matrix')
            profile.resetProfileSetting('translation_vector')
            self.updateProfile()

    def getParameters(self):
        return self.rotationValues, self.translationValues

    def setParameters(self, params):
        self.rotationValues = params[0]
        self.translationValues = params[1]
        self.updateAllControls()

    def getProfileSettings(self):
        self.rotationValues = profile.getProfileSettingNumpy('rotation_matrix')
        self.translationValues = profile.getProfileSettingNumpy('translation_vector')

    def putProfileSettings(self):
        profile.putProfileSettingNumpy('rotation_matrix', self.rotationValues)
        profile.putProfileSettingNumpy('translation_vector', self.translationValues)

    def updateAllControls(self):
        for i in range(3):
            for j in range(3):
                self.rotationValues[i][j] = round(self.rotationValues[i][j], 6)
                self.rotationTexts[i][j].SetValue(str(self.rotationValues[i][j]))

        for i in range(3):
            self.translationValues[i] = round(self.translationValues[i], 4)
            self.translationTexts[i].SetValue(str(self.translationValues[i]))

    def updateEngine(self):
        if hasattr(self, 'pcg'):
            self.pcg.setPlatformExtrinsics(self.rotationValues, self.translationValues)

    def updateProfile(self):
        self.getProfileSettings()
        self.updateAllControls()
        self.updateEngine()

    def updateAllControlsToProfile(self):
        self.putProfileSettings()
        self.updateEngine()

    #TODO: move
    def getValueFloat(self, value): 
        try:
            return float(eval(value.replace(',', '.'), {}, {}))
        except:
            return 0.0