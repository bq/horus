#!/usr/bin/python
# -*- coding: utf-8 -*-
#-----------------------------------------------------------------------#
#                                                                       #
# This file is part of the Horus Project                                #
#                                                                       #
# Copyright (C) 2014-2015 Mundo Reader S.L.                             #
#                                                                       #
# Date: October 2014 - February 2015                                    #
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

import time
import wx._core

from horus.gui.util.imageView import ImageView

from horus.gui.wizard.wizardPage import WizardPage

import horus.util.error as Error
from horus.util import profile, resources

from horus.engine.driver import Driver
from horus.engine import calibration


class ConnectionPage(WizardPage):
    def __init__(self, parent, buttonPrevCallback=None, buttonNextCallback=None):
        WizardPage.__init__(self, parent,
                            title=_("Connection"),
                            buttonPrevCallback=buttonPrevCallback,
                            buttonNextCallback=buttonNextCallback)

        self.parent = parent
        self.driver = Driver.Instance()
        self.cameraIntrinsics = calibration.CameraIntrinsics.Instance()
        self.autoCheck = calibration.SimpleLaserTriangulation.Instance()

        self.connectButton = wx.Button(self.panel, label=_("Connect"))
        self.settingsButton = wx.Button(self.panel, label=_("Edit settings"))


        self.patternLabel = wx.StaticText(self.panel, label=_("Put the pattern on the platform as shown in the picture and press \"Auto check\""))
        self.patternLabel.Wrap(400)
        self.imageView = ImageView(self.panel)
        self.imageView.setImage(wx.Image(resources.getPathForImage("pattern-position-right.jpg")))
        self.autoCheckButton = wx.Button(self.panel, label=_("Auto check"))
        self.gauge = wx.Gauge(self.panel, range=100, size=(-1, 30))
        self.resultLabel = wx.StaticText(self.panel, size=(-1, 30))

        self.connectButton.Enable()
        self.settingsButton.Enable()
        self.patternLabel.Disable()
        self.imageView.Disable()
        self.autoCheckButton.Disable()
        self.skipButton.Disable()
        self.nextButton.Disable()
        self.resultLabel.Hide()
        self.enableNext = False

        vbox = wx.BoxSizer(wx.VERTICAL)
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.Add(self.connectButton, 1, wx.ALL|wx.EXPAND, 5)
        hbox.Add(self.settingsButton, 1, wx.ALL|wx.EXPAND, 5)
        vbox.Add(hbox, 0, wx.ALL|wx.EXPAND, 2)
        vbox.Add(self.patternLabel, 0, wx.ALL|wx.CENTER, 5)
        vbox.Add(self.imageView, 1, wx.ALL|wx.EXPAND, 5)
        vbox.Add(self.resultLabel, 0, wx.ALL|wx.CENTER, 5)
        vbox.Add(self.gauge, 0, wx.ALL|wx.EXPAND, 5)
        vbox.Add(self.autoCheckButton, 0, wx.ALL|wx.EXPAND, 5)
        self.panel.SetSizer(vbox)

        self.Layout()

        self.connectButton.Bind(wx.EVT_BUTTON, self.onConnectButtonClicked)
        self.settingsButton.Bind(wx.EVT_BUTTON, self.onSettingsButtonClicked)
        self.autoCheckButton.Bind(wx.EVT_BUTTON, self.onAutoCheckButtonClicked)
        self.Bind(wx.EVT_SHOW, self.onShow)

        self.videoView.setMilliseconds(50)
        self.videoView.setCallback(self.getDetectChessboardFrame)
        self.updateStatus(self.driver.isConnected)

    def onShow(self, event):
        if event.GetShow():
            self.updateStatus(self.driver.isConnected)
        else:
            try:
                self.videoView.stop()
            except:
                pass

    def getFrame(self):
        return self.driver.camera.captureImage()

    def getDetectChessboardFrame(self):
        frame = self.getFrame()
        if frame is not None:
            retval, frame = self.cameraIntrinsics.detectChessboard(frame)
        return frame

    def onUnplugged(self):
        self.videoView.stop()
        self.autoCheck.cancel()
        self.afterMoveMotor()

    def onConnectButtonClicked(self, event):
        self.driver.setCallbacks(self.beforeConnect, lambda r: wx.CallAfter(self.afterConnect,r))
        self.driver.connect()

    def onSettingsButtonClicked(self, event):
        SettingsWindow(self)

    def beforeConnect(self):
        self.connectButton.Disable()
        self.prevButton.Disable()
        self.videoView.stop()
        self.driver.board.setUnplugCallback(None)
        self.driver.camera.setUnplugCallback(None)
        self.waitCursor = wx.BusyCursor()

    def afterConnect(self, response):
        ret, result = response

        if not ret:
            if result is Error.WrongFirmware:
                dlg = wx.MessageDialog(self, _("Board has a wrong firmware.\nPlease select your Board\nand press Upload Firmware"), _(result), wx.OK|wx.ICON_INFORMATION)
                dlg.ShowModal()
                dlg.Destroy()
                self.updateStatus(False)
                self.GetParent().parent.onPreferences(None)
            elif result is Error.BoardNotConnected:
                dlg = wx.MessageDialog(self, _("Board is not connected.\nPlease connect your board\nand select a valid Serial Name"), _(result), wx.OK|wx.ICON_INFORMATION)
                dlg.ShowModal()
                dlg.Destroy()
                self.updateStatus(False)
                self.GetParent().parent.onPreferences(None)
            elif result is Error.CameraNotConnected:
                dlg = wx.MessageDialog(self, _("Please plug your camera and try to connect again"), _(result), wx.OK|wx.ICON_ERROR)
                dlg.ShowModal()
                dlg.Destroy()
            elif result is Error.WrongCamera:
                dlg = wx.MessageDialog(self, _("You probably have selected a wrong camera.\nPlease select other Camera Id"), _(result), wx.OK|wx.ICON_INFORMATION)
                dlg.ShowModal()
                dlg.Destroy()
                self.updateStatus(False)
                self.GetParent().parent.onPreferences(None)
            elif result is Error.InvalidVideo:
                dlg = wx.MessageDialog(self, _("Unplug and plug your camera USB cable and try to connect again."), _(result), wx.OK|wx.ICON_ERROR)
                dlg.ShowModal()
                dlg.Destroy()

        self.updateStatus(self.driver.isConnected)
        self.prevButton.Enable()
        del self.waitCursor

    def onAutoCheckButtonClicked(self, event):
        if profile.getProfileSettingBool('adjust_laser'):
            profile.putProfileSetting('adjust_laser', False)
            dlg = wx.MessageDialog(self, _("It is recomended to adjust line lasers vertically.\nYou need to use the allen wrench.\nDo you want to adjust it now?"), _("Manual laser adjustment"), wx.YES_NO | wx.ICON_QUESTION)
            result = dlg.ShowModal() == wx.ID_YES
            dlg.Destroy()
            if result:
                self.driver.board.setLeftLaserOn()
                self.driver.board.setRightLaserOn()
        else:
            self.beforeAutoCheck()

            #-- Perform auto check
            self.autoCheck.setCallbacks(None,
                                        lambda p: wx.CallAfter(self.progressAutoCheck,p),
                                        lambda r: wx.CallAfter(self.afterAutoCheck,r))
            self.autoCheck.start()

    def beforeAutoCheck(self):
        self.autoCheckButton.Disable()
        self.prevButton.Disable()
        self.skipButton.Disable()
        self.nextButton.Disable()
        self.enableNext = False
        self.gauge.SetValue(0)
        self.resultLabel.Hide()
        self.gauge.Show()
        self.waitCursor = wx.BusyCursor()
        self.Layout()

    def progressAutoCheck(self, progress):
        self.gauge.SetValue(0.9*progress)

    def afterAutoCheck(self, response):
        ret, result = response

        if ret:
            self.resultLabel.SetLabel(_("All OK. Please press next to continue"))
        else:
            self.resultLabel.SetLabel(_("Please check motor direction and pattern position and try again"))

        if ret:
            self.skipButton.Disable()
            self.nextButton.Enable()
        else:
            self.skipButton.Enable()
            self.nextButton.Disable()

        self.videoView.setMilliseconds(20)
        self.videoView.setCallback(self.getFrame)

        self.driver.board.setSpeedMotor(150)
        self.driver.board.setRelativePosition(-90)
        self.driver.board.enableMotor()
        self.driver.board.moveMotor(nonblocking=True, callback=(lambda r: wx.CallAfter(self.afterMoveMotor)))

    def afterMoveMotor(self):
        self.videoView.setMilliseconds(50)
        self.videoView.setCallback(self.getDetectChessboardFrame)
        self.gauge.SetValue(100)
        self.enableNext = True
        self.resultLabel.Show()
        self.autoCheckButton.Enable()
        self.prevButton.Enable()
        self.driver.board.disableMotor()
        self.gauge.Hide()
        if hasattr(self, 'waitCursor'):
            del self.waitCursor
        self.panel.Fit()
        self.panel.Layout()
        self.Layout()

    def updateStatus(self, status):
        if status:
            self.driver.board.setUnplugCallback(lambda: wx.CallAfter(self.parent.onBoardUnplugged))
            self.driver.camera.setUnplugCallback(lambda: wx.CallAfter(self.parent.onCameraUnplugged))
            #if profile.getPreference('workbench') != 'Calibration workbench':
            profile.putPreference('workbench', 'Calibration workbench')
            self.GetParent().parent.workbenchUpdate(False)
            self.videoView.play()
            self.connectButton.Disable()
            self.autoCheckButton.Enable()
            self.patternLabel.Enable()
            self.imageView.Enable()
            self.skipButton.Enable()
            self.enableNext = True
            self.driver.board.setLeftLaserOff()
            self.driver.board.setRightLaserOff()
        else:
            self.videoView.stop()
            self.gauge.SetValue(0)
            self.gauge.Show()
            self.resultLabel.Hide()
            self.resultLabel.SetLabel("")
            self.connectButton.Enable()
            self.skipButton.Disable()
            self.nextButton.Disable()
            self.enableNext = False
            self.autoCheckButton.Disable()
        self.Layout()

class SettingsWindow(wx.Dialog):

    def __init__(self, parent):
        super(SettingsWindow, self).__init__(parent, title=_('Settings'), size=(420,-1), style=wx.DEFAULT_FRAME_STYLE^wx.RESIZE_BORDER)

        self.driver = Driver.Instance()
        self.cameraIntrinsics = calibration.CameraIntrinsics.Instance()
        self.simpleLaserTriangulation = calibration.SimpleLaserTriangulation.Instance()
        self.laserTriangulation = calibration.LaserTriangulation.Instance()
        self.platformExtrinsics = calibration.PlatformExtrinsics.Instance()

        #-- Elements
        _choices = []
        choices = profile.getProfileSettingObject('luminosity').getType()
        for i in choices:
            _choices.append(_(i))
        self.initLuminosity = profile.getProfileSetting('luminosity')
        self.luminosityDict = dict(zip(_choices, choices))
        self.luminosityText = wx.StaticText(self, label=_('Luminosity'))
        self.luminosityText.SetToolTip(wx.ToolTip(_('Change the luminosity until colored lines appear over the chess pattern in the video')))
        self.luminosityComboBox = wx.ComboBox(self, wx.ID_ANY,
                                            value=_(self.initLuminosity),
                                            choices=_choices,
                                            style=wx.CB_READONLY)
        invert = profile.getProfileSettingBool('invert_motor')
        self.invertMotorCheckBox = wx.CheckBox(self, label=_("Invert the motor direction"))
        self.invertMotorCheckBox.SetValue(invert)
        tooltip = _("Minimum distance between the origin of the pattern (bottom-left corner) and the pattern's base surface")
        self.image = wx.Image(resources.getPathForImage("pattern-distance.jpg"), wx.BITMAP_TYPE_ANY)
        
        self.patternDistance = float(profile.getProfileSetting('pattern_distance'))
        self.patternImage = wx.StaticBitmap(self, wx.ID_ANY, wx.BitmapFromImage(self.image))
        self.patternImage.SetToolTip(wx.ToolTip(tooltip))
        self.patternLabel = wx.StaticText(self, label=_('Pattern distance (mm)'))
        self.patternLabel.SetToolTip(wx.ToolTip(tooltip))
        self.patternTextbox = wx.TextCtrl(self, value = str(profile.getProfileSettingFloat('pattern_distance')))
        self.okButton = wx.Button(self, label=_('OK'))
        self.cancelButton = wx.Button(self, label=_('Cancel'))
        
        #-- Events
        self.luminosityComboBox.Bind(wx.EVT_COMBOBOX, self.onLuminosityComboBoxChanged)
        self.invertMotorCheckBox.Bind(wx.EVT_CHECKBOX, self.onInvertMotor)
        self.patternTextbox.Bind(wx.EVT_TEXT, self.onTextBoxChanged)
        self.cancelButton.Bind(wx.EVT_BUTTON, self.onClose)
        self.okButton.Bind(wx.EVT_BUTTON, self.onOk)
        self.Bind(wx.EVT_CLOSE, self.onClose)

        #-- Layout
        vbox = wx.BoxSizer(wx.VERTICAL)
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.Add(self.luminosityText, 0, wx.ALL, 7)
        hbox.Add(self.luminosityComboBox, 1, wx.ALL, 3)
        vbox.Add(hbox, 0, wx.ALL^wx.BOTTOM|wx.EXPAND, 7)
        vbox.Add(wx.StaticLine(self), 0, wx.ALL^wx.BOTTOM|wx.EXPAND, 10)
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.Add(self.invertMotorCheckBox, 0, wx.ALL, 10)
        vbox.Add(hbox)
        vbox.Add(wx.StaticLine(self), 0, wx.ALL^wx.BOTTOM^wx.TOP|wx.EXPAND, 10)
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.Add(self.patternLabel, 0, wx.ALL, 7)
        hbox.Add(self.patternTextbox, 1, wx.ALL, 3)
        vbox.Add(hbox, 0, wx.ALL^wx.BOTTOM|wx.EXPAND, 10)
        vbox.Add(self.patternImage, 0, wx.ALL|wx.CENTER, 10)
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.Add(self.cancelButton, 1, wx.ALL, 3)
        hbox.Add(self.okButton, 1, wx.ALL, 3)
        vbox.Add(hbox, 0, wx.ALL|wx.EXPAND, 10)
        self.SetSizer(vbox)
        self.Center()
        self.Fit()

        self.ShowModal()

    def onTextBoxChanged(self, event):
        try:
            value = float(self.patternTextbox.GetValue())
            if value >= 0:
                self.patternDistance = value
        except:
            pass

    def setPatternDistance(self, patternDistance):
        profile.putProfileSetting('pattern_distance', patternDistance)

        patternRows = profile.getProfileSettingInteger('pattern_rows')
        patternColumns = profile.getProfileSettingInteger('pattern_columns')
        squareWidth = profile.getProfileSettingInteger('square_width')

        self.cameraIntrinsics.setPatternParameters(patternRows, patternColumns, squareWidth, patternDistance)
        self.simpleLaserTriangulation.setPatternParameters(patternRows, patternColumns, squareWidth, patternDistance)
        self.laserTriangulation.setPatternParameters(patternRows, patternColumns, squareWidth, patternDistance)
        self.platformExtrinsics.setPatternParameters(patternRows, patternColumns, squareWidth, patternDistance)

    def setLuminosity(self, luminosity):
        profile.putProfileSetting('luminosity', luminosity)
        
        if luminosity =='Low':
            luminosity = 32
        elif luminosity =='Medium':
            luminosity = 16
        elif luminosity =='High':
            luminosity = 8
        profile.putProfileSetting('exposure_control', luminosity)
        profile.putProfileSetting('exposure_calibration', luminosity)

        self.driver.camera.setExposure(luminosity)

    def onLuminosityComboBoxChanged(self, event):
        value = self.luminosityDict[event.GetEventObject().GetValue()]
        self.setLuminosity(value)

    def onInvertMotor(self, event):
        invert = self.invertMotorCheckBox.GetValue()
        profile.putProfileSetting('invert_motor', invert)
        self.driver.board.setInvertMotor(invert)

    def onOk(self, event):
        self.setPatternDistance(self.patternDistance)
        self.EndModal(wx.ID_OK)
        self.Destroy()

    def onClose(self, event):
        self.setLuminosity(self.initLuminosity)
        self.EndModal(wx.ID_OK)
        self.Destroy()