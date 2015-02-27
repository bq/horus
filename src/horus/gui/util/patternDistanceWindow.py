#!/usr/bin/python
# -*- coding: utf-8 -*-
#-----------------------------------------------------------------------#
#                                                                       #
# This file is part of the Horus Project                                #
#                                                                       #
# Copyright (C) 2014-2015 Mundo Reader S.L.                             #
#                                                                       #
# Date: October 2014                                                    #
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
        super(PatternDistanceWindow, self).__init__(parent, size=(300,350), style=wx.DEFAULT_FRAME_STYLE^wx.RESIZE_BORDER)

        self.parent = parent
        self.cameraIntrinsics = calibration.CameraIntrinsics.Instance()
        self.simpleLaserTriangulation = calibration.SimpleLaserTriangulation.Instance()
        self.laserTriangulation = calibration.LaserTriangulation.Instance()
        self.platformExtrinsics = calibration.PlatformExtrinsics.Instance()

        #-- Elements
        self.text = wx.StaticText(self, label=_('Pattern distance value must be a number higher than 0. Please, change it in the textbox below.'))
        self.text.Wrap(280)
        # self.pattern_image = wx.Image(resources.getPathForImage("pattern_distance.jpg"))
        img = wx.Image(resources.getPathForImage("pattern_distance.jpg"), wx.BITMAP_TYPE_ANY)
        img = img.Scale(280, 160, wx.IMAGE_QUALITY_HIGH)
        self.pattern_image=wx.StaticBitmap(self, wx.ID_ANY, wx.BitmapFromImage(img))


        self.pattern_text = wx.StaticText(self, label=_('Pattern distance (mm)'))
        self.pattern_text.SetToolTip(wx.ToolTip(_('Distance between the upper edge of the chess row closer to the platform and the platform.')))

        self.pattern_textbox = wx.TextCtrl(self, value = str(profile.getProfileSettingFloat('pattern_distance')))
        self.okButton = wx.Button(self, label=_("OK"))
        self.text.SetToolTip(wx.ToolTip(_('Distance between the upper edge of the chess row closer to the platform and the platform.')))

        #-- Layout
        vbox = wx.BoxSizer(wx.VERTICAL)
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.Add(self.text, 0, wx.ALL|wx.EXPAND, 12)
        vbox.Add(hbox, 0, wx.ALL|wx.EXPAND,-1)
        hbox2 = wx.BoxSizer(wx.HORIZONTAL)
        hbox2.Add(self.pattern_image, 0, wx.ALL, 10)
        vbox.Add(hbox2, 0, wx.ALL, -1)
        hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        hbox1.Add(self.pattern_text, 0, wx.ALL|wx.EXPAND, 12)
        hbox1.Add(self.pattern_textbox, 0, wx.ALL|wx.EXPAND, 12)
        vbox.Add(hbox1, 0, wx.ALL|wx.EXPAND,-1)
        hbox3 = wx.BoxSizer(wx.HORIZONTAL)
        hbox3.Add(self.okButton, 0, wx.ALL|wx.EXPAND, 12)
        vbox.Add(hbox3, 0, wx.ALIGN_CENTER_HORIZONTAL,-1)
        self.SetSizer(vbox)

        self.pattern_textbox.Bind(wx.EVT_TEXT, self.onTextBoxChanged)
        self.Bind(wx.EVT_CLOSE, self.onClose)
        self.okButton.Bind(wx.EVT_BUTTON, self.onClose)
        self.okButton.Disable()

        self.Centre()
        self.ShowModal()

    def onTextBoxChanged(self, event):
        try:
            value=float(self.pattern_textbox.GetValue())
            if value > 0:
                profile.putProfileSetting('pattern_distance',self.pattern_textbox.GetValue())

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
                self.okButton.Enable()

        except:
            self.okButton.Disable()

    def onClose(self, event):
        self.Destroy()