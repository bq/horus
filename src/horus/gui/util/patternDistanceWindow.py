#!/usr/bin/python
# -*- coding: utf-8 -*-
#-----------------------------------------------------------------------#
#                                                                       #
# This file is part of the Horus Project                                #
#                                                                       #
# Copyright (C) 2014-2015 Mundo Reader S.L.                             #
#                                                                       #
# Date: February 2015                                                   #
# Author: Irene Sanz Nieto <irene.sanz@bq.com>                          #
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

__author__ = "Irene Sanz Nieto <irene.sanz@bq.com>"
__license__ = "GNU General Public License v2 http://www.gnu.org/licenses/gpl.html"

import wx._core
from horus.util import profile, resources

from horus.engine import calibration


class PatternDistanceWindow(wx.Dialog):

    def __init__(self, parent):
        super(PatternDistanceWindow, self).__init__(parent, title=_('Pattern distance'), size=(420,-1), style=wx.DEFAULT_FRAME_STYLE^wx.RESIZE_BORDER)

        self.value = float(profile.getProfileSetting('pattern_distance'))
        self.cameraIntrinsics = calibration.CameraIntrinsics.Instance()
        self.simpleLaserTriangulation = calibration.SimpleLaserTriangulation.Instance()
        self.laserTriangulation = calibration.LaserTriangulation.Instance()
        self.platformExtrinsics = calibration.PlatformExtrinsics.Instance()

        #-- Elements
        self.description = wx.StaticText(self, label=_('Pattern distance value must be a number higher than 0. Please, change it in the textbox below.'))
        self.description.Wrap(400)
        tooltip = _("Minimum distance between the origin of the pattern (bottom-left corner) and the pattern's base surface")
        self.image = wx.Image(resources.getPathForImage("pattern-distance.jpg"), wx.BITMAP_TYPE_ANY)
        self.patternImage = wx.StaticBitmap(self, wx.ID_ANY, wx.BitmapFromImage(self.image))
        self.patternImage.SetToolTip(wx.ToolTip(tooltip))
        self.patternLabel = wx.StaticText(self, label=_('Pattern distance (mm)'))
        self.patternLabel.SetToolTip(wx.ToolTip(tooltip))
        self.patternTextbox = wx.TextCtrl(self, value = str(profile.getProfileSettingFloat('pattern_distance')))
        self.okButton = wx.Button(self, label=_('OK'))
        self.cancelButton = wx.Button(self, label=_('Cancel'))
        
        #-- Events
        self.patternTextbox.Bind(wx.EVT_TEXT, self.onTextBoxChanged)
        self.cancelButton.Bind(wx.EVT_BUTTON, self.onClose)
        self.okButton.Bind(wx.EVT_BUTTON, self.onOk)
        self.Bind(wx.EVT_CLOSE, self.onClose)

        #-- Layout
        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(self.description, 0, wx.ALL|wx.CENTER, 10)
        vbox.Add(self.patternImage, 0, wx.ALL|wx.CENTER, 10)
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.Add(self.patternLabel, 0, wx.ALL, 7)
        hbox.Add(self.patternTextbox, 0, wx.ALL, 3)
        hbox.Add(self.okButton, 0, wx.ALL, 3)
        hbox.Add(self.cancelButton, 0, wx.ALL, 3)
        vbox.Add(hbox, 0, wx.ALL|wx.CENTER, 10)
        self.SetSizer(vbox)
        self.Center()
        self.Fit()

        self.ShowModal()

    def onTextBoxChanged(self, event):
        try:
            value = float(self.patternTextbox.GetValue())
            if value >= 0:
                self.value = value
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

    def onOk(self, event):
        self.setPatternDistance(self.value)
        self.Destroy()

    def onClose(self, event):
        self.Destroy()