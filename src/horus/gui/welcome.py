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

import os
import wx._core

from horus.util import profile, resources

from horus.gui.wizard.main import Wizard
from horus.gui.util.imageView import ImageView

class WelcomeWindow(wx.Dialog):

    def __init__(self, parent):
        super(WelcomeWindow, self).__init__(parent, size=(640+120,480+40), style=wx.DEFAULT_FRAME_STYLE^wx.RESIZE_BORDER)

        self.parent = parent

        self.lastFiles = eval(profile.getPreference('last_files'))

        header = Header(self)
        content = Content(self)
        checkBoxShow = wx.CheckBox(self, label=_("Don't show this dialog again"), style=wx.ALIGN_LEFT)
        checkBoxShow.SetValue(not profile.getPreferenceBool('show_welcome'))

        checkBoxShow.Bind(wx.EVT_CHECKBOX, self.onCheckBoxChanged)
        self.Bind(wx.EVT_CLOSE, self.onClose)

        #-- Layout
        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(header, 2, wx.ALL|wx.EXPAND, 1)
        vbox.Add(content, 3, wx.ALL|wx.EXPAND^wx.BOTTOM, 20)
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.Add((0,0), 1, wx.ALL|wx.EXPAND, 0)
        hbox.Add(checkBoxShow, 0, wx.ALL, 0)
        vbox.Add(hbox, 0, wx.ALL|wx.EXPAND, 15)
        self.SetSizer(vbox)

        self.Centre()
        self.ShowModal()

    def onCheckBoxChanged(self, event):
        profile.putPreference('show_welcome', not event.Checked())

    def onClose(self, event):
        self.EndModal(wx.ID_OK)
        self.Destroy()


class Header(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)

        logo = ImageView(self)
        logo.setImage(wx.Image(resources.getPathForImage("logo.png")))
        titleText = wx.StaticText(self, label=_("3D Scanning for everyone"))
        titleText.SetFont((wx.Font(13, wx.FONTFAMILY_DEFAULT, wx.NORMAL, wx.FONTWEIGHT_NORMAL)))
        separator = wx.StaticLine(self, -1, style=wx.LI_HORIZONTAL)

        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(logo, 10, wx.ALL^wx.BOTTOM|wx.EXPAND, 30)
        vbox.Add(titleText, 0, wx.TOP|wx.CENTER, 20)
        vbox.Add((0,0), 1, wx.ALL|wx.EXPAND, 0)
        vbox.Add(separator, 0, wx.ALL|wx.EXPAND, 10)
        self.SetSizer(vbox)
        self.Layout()


class CreateNew(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)

        titleText = wx.StaticText(self, label=_("Create new"))
        titleText.SetFont((wx.Font(wx.SystemSettings.GetFont(wx.SYS_ANSI_VAR_FONT).GetPointSize(), wx.FONTFAMILY_DEFAULT, wx.NORMAL, wx.FONTWEIGHT_NORMAL)))

        wizardButton = wx.Button(self, label=_("Wizard mode (step by step)"))
        scanButton = wx.Button(self, label=_("Scan using recent settings"))
        advancedControlButton = wx.Button(self, label=_("Advanced Control"))
        advancedCalibrationButton = wx.Button(self, label=_("Advanced Calibration"))

        wizardButton.Bind(wx.EVT_BUTTON, self.onWizard)
        scanButton.Bind(wx.EVT_BUTTON, self.onScan)
        advancedControlButton.Bind(wx.EVT_BUTTON, self.onAdvancedControl)
        advancedCalibrationButton.Bind(wx.EVT_BUTTON, self.onAdvancedCalibration)

        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(titleText, 0, wx.BOTTOM|wx.CENTER, 10)
        vbox.Add(wizardButton, 1, wx.ALL|wx.EXPAND, 5)
        vbox.Add(scanButton, 1, wx.ALL|wx.EXPAND, 5)
        vbox.Add(advancedControlButton, 1, wx.ALL|wx.EXPAND, 5)
        vbox.Add(advancedCalibrationButton, 1, wx.ALL|wx.EXPAND, 5)

        self.SetSizer(vbox)
        self.Layout()

    def onWizard(self, event):
        self.GetParent().GetParent().Hide()
        self.GetParent().GetParent().parent.controlWorkbench.videoView.stop()
        self.GetParent().GetParent().parent.calibrationWorkbench.videoView.stop()
        self.GetParent().GetParent().parent.scanningWorkbench.videoView.stop()
        self.GetParent().GetParent().parent.controlWorkbench.Disable()
        self.GetParent().GetParent().parent.calibrationWorkbench.Disable()
        self.GetParent().GetParent().parent.scanningWorkbench.Disable()
        wizard = Wizard(self.GetParent().GetParent().parent)
        self.GetParent().GetParent().parent.controlWorkbench.Enable()
        self.GetParent().GetParent().parent.calibrationWorkbench.Enable()
        self.GetParent().GetParent().parent.scanningWorkbench.Enable()
        self.GetParent().GetParent().Close()

    def onScan(self, event):
        profile.putPreference('workbench', 'Scanning workbench')
        self.GetParent().GetParent().parent.workbenchUpdate()
        self.GetParent().GetParent().Close()

    def onAdvancedControl(self, event):
        profile.putPreference('workbench', 'Control workbench')
        self.GetParent().GetParent().parent.workbenchUpdate()
        self.GetParent().GetParent().Close()

    def onAdvancedCalibration(self, event):
        profile.putPreference('workbench', 'Calibration workbench')
        self.GetParent().GetParent().parent.workbenchUpdate()
        self.GetParent().GetParent().Close()


class OpenRecent(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)

        titleText = wx.StaticText(self, label=_("Open recent file"))
        titleText.SetFont((wx.Font(wx.SystemSettings.GetFont(wx.SYS_ANSI_VAR_FONT).GetPointSize(), wx.FONTFAMILY_DEFAULT, wx.NORMAL, wx.FONTWEIGHT_NORMAL)))

        lastFiles = eval(profile.getPreference('last_files'))
        lastFiles.reverse()

        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(titleText, 0, wx.BOTTOM|wx.CENTER, 10)

        for path in lastFiles:
            button = wx.Button(self, label=os.path.basename(path), name=path)
            button.Bind(wx.EVT_BUTTON, self.onButtonPressed)
            vbox.Add(button, 0, wx.ALL|wx.EXPAND, 5)

        self.SetSizer(vbox)
        self.Layout()

    def onButtonPressed(self, event):
        button = event.GetEventObject()
        profile.putPreference('workbench', 'Scanning workbench')
        self.GetParent().GetParent().parent.workbenchUpdate()
        self.GetParent().GetParent().parent.appendLastFile(button.GetName())
        self.GetParent().GetParent().parent.scanningWorkbench.sceneView.loadFile(button.GetName())
        self.GetParent().GetParent().Close()


class Content(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)

        createNew = CreateNew(self)
        openRecent = OpenRecent(self)

        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.Add(createNew, 1, wx.ALL|wx.EXPAND, 20)
        hbox.Add(wx.StaticLine(self, style=wx.LI_VERTICAL), 0, wx.LEFT|wx.RIGHT|wx.EXPAND, 20)
        hbox.Add(openRecent, 1, wx.ALL|wx.EXPAND, 20)

        self.SetSizer(hbox)
        self.Layout()