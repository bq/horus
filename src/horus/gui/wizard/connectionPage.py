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
        self.settingsButton = wx.Button(self.panel, label=_("Edit preferences"))


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
        sa = PreferencesWindow(self)


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
                dlg = wx.MessageDialog(self, _("Board has a wrong firmware.\nPlease select your Board\nand press Upload Firmware"), Error.str(result), wx.OK|wx.ICON_INFORMATION)
                dlg.ShowModal()
                dlg.Destroy()
                self.updateStatus(False)
                self.GetParent().parent.onPreferences(None)
            elif result is Error.BoardNotConnected:
                dlg = wx.MessageDialog(self, _("Board is not connected.\nPlease connect your board\nand select a valid Serial Name"), Error.str(result), wx.OK|wx.ICON_INFORMATION)
                dlg.ShowModal()
                dlg.Destroy()
                self.updateStatus(False)
                self.GetParent().parent.onPreferences(None)
            elif result is Error.CameraNotConnected:
                dlg = wx.MessageDialog(self, _("Please plug your camera and try to connect again"), Error.str(result), wx.OK|wx.ICON_ERROR)
                dlg.ShowModal()
                dlg.Destroy()
            elif result is Error.WrongCamera:
                dlg = wx.MessageDialog(self, _("You probably have selected a wrong camera.\nPlease select other Camera Id"), Error.str(result), wx.OK|wx.ICON_INFORMATION)
                dlg.ShowModal()
                dlg.Destroy()
                self.updateStatus(False)
                self.GetParent().parent.onPreferences(None)
            elif result is Error.InvalidVideo:
                dlg = wx.MessageDialog(self, _("Unplug and plug your camera USB cable and try to connect again."), Error.str(result), wx.OK|wx.ICON_ERROR)
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
            self.resultLabel.SetLabel(_("Error in Auto check. Please try again"))

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
            #if profile.getPreference('workbench') != 'calibration':
            profile.putPreference('workbench', 'calibration')
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



class PreferencesWindow(wx.Dialog):

    def __init__(self, parent):
        super(PreferencesWindow, self).__init__(parent, size=(300,330), style=wx.DEFAULT_FRAME_STYLE^wx.RESIZE_BORDER)

        self.parent = parent
        self.driver = Driver.Instance()
        self.cameraIntrinsics = calibration.CameraIntrinsics.Instance()
        self.simpleLaserTriangulation = calibration.SimpleLaserTriangulation.Instance()
        self.laserTriangulation = calibration.LaserTriangulation.Instance()
        self.platformExtrinsics = calibration.PlatformExtrinsics.Instance()

        #-- Elements
        self.text = wx.StaticText(self, label=_('Pattern distance (mm)'))
        self.textbox = wx.TextCtrl(self, value = str(profile.getProfileSettingFloat('pattern_distance')))
        self.okButton = wx.Button(self, label=_("OK"))
        self.text.SetToolTip(wx.ToolTip(_('Distance between the upper edge of the chess row closer to the platform and the platform.')))

        luminosity=profile.getProfileSettingObject('luminosity').getType()
        self.luminosityText = wx.StaticText(self, label=_('Luminosity'))
        self.luminosityText.SetToolTip(wx.ToolTip(_('Change the luminosity until coloured lines appear over the chess pattern in the video.')))
        self.luminosityComboBox = wx.ComboBox(self, wx.ID_ANY,
                                            value=profile.getProfileSetting('luminosity'),
                                            choices=[_(luminosity[0]), _(luminosity[1]), _(luminosity[2])],
                                            style=wx.CB_READONLY)

        img = wx.Image(resources.getPathForImage("pattern_distance.jpg"), wx.BITMAP_TYPE_ANY)
        img = img.Scale(280, 160, wx.IMAGE_QUALITY_HIGH)
        self.pattern_image=wx.StaticBitmap(self, wx.ID_ANY, wx.BitmapFromImage(img))


        #-- Layout
        vbox = wx.BoxSizer(wx.VERTICAL)
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.Add(self.luminosityText, 0, wx.ALL|wx.EXPAND, 12)
        hbox.Add(self.luminosityComboBox, -1, wx.ALL|wx.EXPAND, 12)
        vbox.Add(hbox, 0, wx.ALL|wx.EXPAND,-1)
        hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        hbox1.Add(self.pattern_image, 0, wx.ALL, 10)
        vbox.Add(hbox1, 0, wx.ALL, -1)
        hbox2 = wx.BoxSizer(wx.HORIZONTAL)
        hbox2.Add(self.text, 0, wx.ALL|wx.EXPAND, 12)
        hbox2.Add(self.textbox, 0, wx.ALL|wx.EXPAND, 12)
        vbox.Add(hbox2, 0, wx.ALL|wx.EXPAND,-1)
        hbox3 = wx.BoxSizer(wx.HORIZONTAL)
        hbox3.Add(self.okButton, 0, wx.ALL|wx.EXPAND, 12)
        vbox.Add(hbox3, 0, wx.ALIGN_CENTER_HORIZONTAL,-1)
        self.SetSizer(vbox)



        self.textbox.Bind(wx.EVT_TEXT, self.onTextBoxChanged)
        self.luminosityComboBox.Bind(wx.EVT_COMBOBOX, self.onLuminosityComboBoxChanged)
        self.Bind(wx.EVT_CLOSE, self.onClose)
        self.okButton.Bind(wx.EVT_BUTTON, self.onClose)

        self.Centre()
        self.ShowModal()



    def onTextBoxChanged(self, event):
        profile.putProfileSetting('pattern_distance',self.textbox.GetValue())

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

    def onClose(self, event):
        self.Destroy()

    def onLuminosityComboBoxChanged(self, event):
        value = event.GetEventObject().GetValue()
        profile.putProfileSetting('luminosity', value)
        if value ==_("Low"):
            value = 32
        elif value ==_("Medium"):
            value = 16
        elif value ==_("High"):
            value = 8
        profile.putProfileSetting('exposure_control', value)
        profile.putProfileSetting('exposure_calibration', value)
        profile.putProfileSetting('exposure_scanning', value)
        self.driver.camera.setExposure(value)
