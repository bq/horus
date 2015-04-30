#!/usr/bin/python
# -*- coding: utf-8 -*-
#-----------------------------------------------------------------------#
#                                                                       #
# This file is part of the Horus Project                                #
#                                                                       #
# Copyright (C) 2014-2015 Mundo Reader S.L.                             #
#                                                                       #
# Date: March, June, November 2014                                      #
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

import gc
import os
import cv2
import glob
import time
import struct
import platform
import wx._core
import webbrowser

if platform.system() == "Darwin":
    from horus.engine import uvc

from horus.util import profile, resources, meshLoader

from horus.gui.workbench.control.main import ControlWorkbench
from horus.gui.workbench.scanning.main import ScanningWorkbench
from horus.gui.workbench.calibration.main import CalibrationWorkbench
from horus.gui.preferences import PreferencesDialog
from horus.gui.welcome import WelcomeWindow
from horus.gui.wizard.main import *

from horus.engine.driver import Driver
from horus.engine import scan, calibration

VERSION = '0.1.1'

class MainWindow(wx.Frame):

    size = (640+337,480+150)

    def __init__(self):
        super(MainWindow, self).__init__(None, title=_("Horus " + VERSION + " - Beta"), size=self.size)

        self.SetMinSize((600, 450))

        ###-- Initialize Engine
        self.driver = Driver.Instance()
        self.simpleScan = scan.SimpleScan.Instance()
        self.textureScan = scan.TextureScan.Instance()
        self.pcg = scan.PointCloudGenerator.Instance()
        self.cameraIntrinsics = calibration.CameraIntrinsics.Instance()
        self.simpleLaserTriangulation = calibration.SimpleLaserTriangulation.Instance()
        self.laserTriangulation = calibration.LaserTriangulation.Instance()
        self.platformExtrinsics = calibration.PlatformExtrinsics.Instance()

        #-- Serial Name initialization
        serialList = self.serialList()
        currentSerial = profile.getProfileSetting('serial_name')
        if len(serialList) > 0:
            if currentSerial not in serialList:
                profile.putProfileSetting('serial_name', serialList[0])

        #-- Video Id initialization
        videoList = self.videoList()
        currentVideoId = profile.getProfileSetting('camera_id')
        if len(videoList) > 0:
            if currentVideoId not in videoList:
                profile.putProfileSetting('camera_id', videoList[0])

        self.lastFiles = eval(profile.getPreference('last_files'))

        print ">>> Horus " + VERSION + " <<<"

        ###-- Initialize GUI

        ##-- Set Icon
        icon = wx.Icon(resources.getPathForImage("horus.ico"), wx.BITMAP_TYPE_ICO)
        self.SetIcon(icon)

        ##-- Status Bar
        #self.CreateStatusBar()

        ##-- Menu Bar
        self.menuBar = wx.MenuBar()

        #--  Menu File
        self.menuFile = wx.Menu()
        self.menuLaunchWizard = self.menuFile.Append(wx.NewId(), _("Launch Wizard"))
        self.menuFile.AppendSeparator()
        self.menuLoadModel = self.menuFile.Append(wx.NewId(), _("Load Model"))
        self.menuSaveModel = self.menuFile.Append(wx.NewId(), _("Save Model"))
        self.menuClearModel = self.menuFile.Append(wx.NewId(), _("Clear Model"))
        self.menuFile.AppendSeparator()
        self.menuOpenProfile = self.menuFile.Append(wx.NewId(), _("Open Profile"), _("Opens Profile .ini"))
        self.menuSaveProfile = self.menuFile.Append(wx.NewId(), _("Save Profile"))
        self.menuResetProfile = self.menuFile.Append(wx.NewId(), _("Reset Profile"))
        self.menuFile.AppendSeparator()
        self.menuExit = self.menuFile.Append(wx.ID_EXIT, _("Exit"))
        self.menuBar.Append(self.menuFile, _("File"))

        #-- Menu Edit
        self.menuEdit = wx.Menu()
        # self.menuBasicMode = self.menuEdit.AppendRadioItem(wx.NewId(), _("Basic Mode"))
        # self.menuAdvancedMode = self.menuEdit.AppendRadioItem(wx.NewId(), _("Advanced Mode"))
        # self.menuEdit.AppendSeparator()
        self.menuPreferences = self.menuEdit.Append(wx.NewId(), _("Preferences"))
        self.menuBar.Append(self.menuEdit, _("Edit"))

        #-- Menu View
        self.menuView = wx.Menu()
        self.menuControl = wx.Menu()
        self.menuControlPanel = self.menuControl.AppendCheckItem(wx.NewId(), _("Panel"))
        self.menuControlVideo = self.menuControl.AppendCheckItem(wx.NewId(), _("Video"))
        self.menuView.AppendMenu(wx.NewId(), _("Control"), self.menuControl)
        self.menuCalibration = wx.Menu()
        self.menuCalibrationPanel = self.menuCalibration.AppendCheckItem(wx.NewId(), _("Panel"))
        self.menuCalibrationVideo = self.menuCalibration.AppendCheckItem(wx.NewId(), _("Video"))
        self.menuView.AppendMenu(wx.NewId(), _("Calibration"), self.menuCalibration)
        self.menuScanning = wx.Menu()
        self.menuScanningPanel = self.menuScanning.AppendCheckItem(wx.NewId(), _("Panel"))
        self.menuScanningVideo = self.menuScanning.AppendCheckItem(wx.NewId(), _("Video"))
        self.menuScanningScene = self.menuScanning.AppendCheckItem(wx.NewId(), _("Scene"))
        self.menuView.AppendMenu(wx.NewId(), _("Scanning"), self.menuScanning)
        self.menuBar.Append(self.menuView, _("View"))

        #-- Menu Help
        self.menuHelp = wx.Menu()
        self.menuWelcome = self.menuHelp.Append(wx.ID_ANY, _("Welcome"))
        self.menuWiki = self.menuHelp.Append(wx.ID_ANY, _("Wiki"))
        self.menuSources = self.menuHelp.Append(wx.ID_ANY, _("Sources"))
        self.menuIssues = self.menuHelp.Append(wx.ID_ANY, _("Issues"))
        self.menuForum = self.menuHelp.Append(wx.ID_ANY, _("Forum"))
        self.menuAbout = self.menuHelp.Append(wx.ID_ABOUT, _("About"))
        self.menuBar.Append(self.menuHelp, _("Help"))

        self.SetMenuBar(self.menuBar)

        ##-- Create Workbenchs
        self.controlWorkbench = ControlWorkbench(self)
        self.scanningWorkbench = ScanningWorkbench(self)
        self.calibrationWorkbench = CalibrationWorkbench(self)

        _choices = []
        choices = profile.getProfileSettingObject('workbench').getType()
        for i in choices:
            _choices.append(_(i))
        self.workbenchDict = dict(zip(_choices, choices))

        for workbench in [self.controlWorkbench, self.calibrationWorkbench, self.scanningWorkbench]:
            workbench.combo.Clear()
            for i in choices:
                workbench.combo.Append(_(i))

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.controlWorkbench, 1, wx.ALL|wx.EXPAND)
        sizer.Add(self.calibrationWorkbench, 1, wx.ALL|wx.EXPAND)
        sizer.Add(self.scanningWorkbench, 1, wx.ALL|wx.EXPAND)
        self.SetSizer(sizer)

        ##-- Events
        self.Bind(wx.EVT_MENU, self.onLaunchWizard, self.menuLaunchWizard)
        self.Bind(wx.EVT_MENU, self.onLoadModel, self.menuLoadModel)
        self.Bind(wx.EVT_MENU, self.onSaveModel, self.menuSaveModel)
        self.Bind(wx.EVT_MENU, self.onClearModel, self.menuClearModel)
        self.Bind(wx.EVT_MENU, self.onOpenProfile, self.menuOpenProfile)
        self.Bind(wx.EVT_MENU, self.onSaveProfile, self.menuSaveProfile)
        self.Bind(wx.EVT_MENU, self.onResetProfile, self.menuResetProfile)
        self.Bind(wx.EVT_MENU, self.onExit, self.menuExit)

        # self.Bind(wx.EVT_MENU, self.onModeChanged, self.menuBasicMode)
        # self.Bind(wx.EVT_MENU, self.onModeChanged, self.menuAdvancedMode)
        self.Bind(wx.EVT_MENU, self.onPreferences, self.menuPreferences)

        self.Bind(wx.EVT_MENU, self.onControlPanelClicked, self.menuControlPanel)
        self.Bind(wx.EVT_MENU, self.onControlVideoClicked, self.menuControlVideo)
        self.Bind(wx.EVT_MENU, self.onCalibrationPanelClicked, self.menuCalibrationPanel)
        self.Bind(wx.EVT_MENU, self.onCalibrationVideoClicked, self.menuCalibrationVideo)
        self.Bind(wx.EVT_MENU, self.onScanningPanelClicked, self.menuScanningPanel)
        self.Bind(wx.EVT_MENU, self.onScanningVideoSceneClicked, self.menuScanningVideo)
        self.Bind(wx.EVT_MENU, self.onScanningVideoSceneClicked, self.menuScanningScene)

        self.Bind(wx.EVT_MENU, self.onAbout, self.menuAbout)
        self.Bind(wx.EVT_MENU, self.onWelcome, self.menuWelcome)
        self.Bind(wx.EVT_MENU, lambda e: webbrowser.open('https://github.com/bq/horus/wiki'), self.menuWiki)
        self.Bind(wx.EVT_MENU, lambda e: webbrowser.open('https://github.com/bq/horus'), self.menuSources)
        self.Bind(wx.EVT_MENU, lambda e: webbrowser.open('https://github.com/bq/horus/issues'), self.menuIssues)
        self.Bind(wx.EVT_MENU, lambda e: webbrowser.open('https://groups.google.com/forum/?hl=es#!forum/ciclop-3d-scanner'), self.menuForum)

        self.Bind(wx.EVT_COMBOBOX, self.onComboBoxWorkbenchSelected, self.controlWorkbench.combo)
        self.Bind(wx.EVT_COMBOBOX, self.onComboBoxWorkbenchSelected, self.calibrationWorkbench.combo)
        self.Bind(wx.EVT_COMBOBOX, self.onComboBoxWorkbenchSelected, self.scanningWorkbench.combo)

        self.Bind(wx.EVT_CLOSE, self.onClose)

        self.updateProfileToAllControls()

        x, y, w, h = wx.Display(0).GetGeometry()
        ws, hs = self.size

        self.SetPosition((x+(w-ws)/2., y+(h-hs)/2.))

        #self.Center()
        self.Show()

    def onLaunchWizard(self, event):
        self.controlWorkbench.videoView.stop()
        self.calibrationWorkbench.videoView.stop()
        self.calibrationWorkbench.cameraIntrinsicsMainPage.videoView.stop()
        self.calibrationWorkbench.laserTriangulationMainPage.videoView.stop()
        self.calibrationWorkbench.platformExtrinsicsMainPage.videoView.stop()
        self.scanningWorkbench.videoView.stop()
        self.controlWorkbench.Disable()
        self.calibrationWorkbench.Disable()
        self.scanningWorkbench.Disable()
        wizard = Wizard(self)
        self.controlWorkbench.Enable()
        self.calibrationWorkbench.Enable()
        self.scanningWorkbench.Enable()

    def onLoadModel(self, event):
        lastFile = os.path.split(profile.getPreference('last_file'))[0]
        dlg = wx.FileDialog(self, _("Open 3D model"), lastFile, style=wx.FD_OPEN|wx.FD_FILE_MUST_EXIST)
        wildcardList = ';'.join(map(lambda s: '*' + s, meshLoader.loadSupportedExtensions()))
        wildcardFilter = "All (%s)|%s;%s" % (wildcardList, wildcardList, wildcardList.upper())
        wildcardList = ';'.join(map(lambda s: '*' + s, meshLoader.loadSupportedExtensions()))
        wildcardFilter += "|Mesh files (%s)|%s;%s" % (wildcardList, wildcardList, wildcardList.upper())
        dlg.SetWildcard(wildcardFilter)
        if dlg.ShowModal() == wx.ID_OK:
            filename = dlg.GetPath()
            if filename is not None:
                self.scanningWorkbench.sceneView.loadFile(filename)
                self.appendLastFile(filename)
        dlg.Destroy()

    def onSaveModel(self, event):
        if self.scanningWorkbench.sceneView._object is None:
            return
        dlg = wx.FileDialog(self, _("Save 3D model"), os.path.split(profile.getPreference('last_file'))[0], style=wx.FD_SAVE|wx.FD_OVERWRITE_PROMPT)
        fileExtensions = meshLoader.saveSupportedExtensions()
        wildcardList = ';'.join(map(lambda s: '*' + s, fileExtensions))
        wildcardFilter = "Mesh files (%s)|%s;%s" % (wildcardList, wildcardList, wildcardList.upper())
        dlg.SetWildcard(wildcardFilter)
        if dlg.ShowModal() == wx.ID_OK:
            filename = dlg.GetPath()
            if not filename.endswith('.ply'):
                if platform.system() == 'Linux': #hack for linux, as for some reason the .ply is not appended.
                    filename += '.ply'
            meshLoader.saveMesh(filename, self.scanningWorkbench.sceneView._object)
            self.appendLastFile(filename)
        dlg.Destroy()

    def onClearModel(self, event):
        if self.scanningWorkbench.sceneView._object is not None:
            dlg = wx.MessageDialog(self, _("Your current model will be erased.\nDo you really want to do it?"), _("Clear Point Cloud"), wx.YES_NO | wx.ICON_QUESTION)
            result = dlg.ShowModal() == wx.ID_YES
            dlg.Destroy()
            if result:
                self.scanningWorkbench.sceneView._clearScene()

    def onOpenProfile(self, event):
        """ """
        dlg=wx.FileDialog(self, _("Select profile file to load"), os.path.split(profile.getPreference('last_profile'))[0], style=wx.FD_OPEN|wx.FD_FILE_MUST_EXIST)
        dlg.SetWildcard("ini files (*.ini)|*.ini")
        if dlg.ShowModal() == wx.ID_OK:
            profileFile = dlg.GetPath()
            profile.loadProfile(profileFile)
            self.updateProfileToAllControls()
        dlg.Destroy()

    def onSaveProfile(self, event):
        """ """
        dlg=wx.FileDialog(self, _("Select profile file to save"), os.path.split(profile.getPreference('last_profile'))[0], style=wx.FD_SAVE)
        dlg.SetWildcard("ini files (*.ini)|*.ini")
        if dlg.ShowModal() == wx.ID_OK:
            profileFile = dlg.GetPath()
            if not profileFile.endswith('.ini'):
                if platform.system() == 'Linux': #hack for linux, as for some reason the .ini is not appended.
                    profileFile += '.ini'
            profile.saveProfile(profileFile)
        dlg.Destroy()

    def onResetProfile(self, event):
        """ """
        dlg = wx.MessageDialog(self, _("This will reset all profile settings to defaults.\nUnless you have saved your current profile, all settings will be lost!\nDo you really want to reset?"), _("Profile reset"), wx.YES_NO | wx.ICON_QUESTION)
        result = dlg.ShowModal() == wx.ID_YES
        dlg.Destroy()
        if result:
            profile.resetProfile()
            self.updateProfileToAllControls()

    def onExit(self, event):
        self.Close(True)

    def onClose(self, event):
        try:
            if self.simpleScan.run or self.textureScan.run:
                self.simpleScan.stop()
                self.textureScan.stop()
                time.sleep(0.5)
            self.controlWorkbench.videoView.stop()
            self.calibrationWorkbench.videoView.stop()
            self.calibrationWorkbench.cameraIntrinsicsMainPage.videoView.stop()
            self.calibrationWorkbench.laserTriangulationMainPage.videoView.stop()
            self.calibrationWorkbench.platformExtrinsicsMainPage.videoView.stop()
            self.scanningWorkbench.videoView.stop()
            self.driver.board.setUnplugCallback(None)
            self.driver.camera.setUnplugCallback(None)
            self.driver.disconnect()
        except:
            pass
        event.Skip()

    def appendLastFile(self, lastFile):
        if lastFile in self.lastFiles:
            self.lastFiles.remove(lastFile)
        self.lastFiles.append(lastFile)
        if len(self.lastFiles) > 4:
            self.lastFiles = self.lastFiles[1:5]
        profile.putPreference('last_file', lastFile)
        profile.putPreference('last_files', self.lastFiles)

    def onModeChanged(self, event):
        # profile.putPreference('basic_mode', self.menuBasicMode.IsChecked())
        self.controlWorkbench.updateProfileToAllControls()
        self.calibrationWorkbench.updateProfileToAllControls()
        self.scanningWorkbench.updateProfileToAllControls()
        self.Layout()

    def onPreferences(self, event):
        if os.name == 'nt':
            self.simpleScan.stop()
            self.textureScan.stop()
            self.laserTriangulation.cancel()
            self.platformExtrinsics.cancel()
            self.controlWorkbench.videoView.stop()
            self.calibrationWorkbench.videoView.stop()
            self.calibrationWorkbench.cameraIntrinsicsMainPage.videoView.stop()
            self.calibrationWorkbench.laserTriangulationMainPage.videoView.stop()
            self.calibrationWorkbench.platformExtrinsicsMainPage.videoView.stop()
            self.scanningWorkbench.videoView.stop()
            self.driver.board.setUnplugCallback(None)
            self.driver.camera.setUnplugCallback(None)
            self.controlWorkbench.updateStatus(False)
            self.calibrationWorkbench.updateStatus(False)
            self.scanningWorkbench.updateStatus(False)
            self.driver.disconnect()
            waitCursor = wx.BusyCursor()

        prefDialog = PreferencesDialog(self)
        prefDialog.ShowModal()
        wx.CallAfter(prefDialog.Show)

        self.updateDriverProfile()
        self.controlWorkbench.updateCallbacks()
        self.calibrationWorkbench.updateCallbacks()
        self.scanningWorkbench.updateCallbacks()

    def onMenuViewClicked(self, key, checked, panel):
        profile.putPreference(key, checked)
        if checked:
            panel.Show()
        else:
            panel.Hide()
        panel.GetParent().Layout()
        panel.Layout()
        self.Layout()

    def onControlPanelClicked(self, event):
        self.onMenuViewClicked('view_control_panel',
                                self.menuControlPanel.IsChecked(),
                                self.controlWorkbench.scrollPanel)

    def onControlVideoClicked(self, event):
        self.onMenuViewClicked('view_control_video',
                                self.menuControlVideo.IsChecked(),
                                self.controlWorkbench.videoView)

    def onCalibrationPanelClicked(self, event):
        self.onMenuViewClicked('view_calibration_panel',
                                self.menuCalibrationPanel.IsChecked(),
                                self.calibrationWorkbench.scrollPanel)

    def onCalibrationVideoClicked(self, event):
        self.onMenuViewClicked('view_calibration_video',
                                self.menuCalibrationVideo.IsChecked(),
                                self.calibrationWorkbench.videoView)

    def onScanningPanelClicked(self, event):
        self.onMenuViewClicked('view_scanning_panel',
                                self.menuScanningPanel.IsChecked(),
                                self.scanningWorkbench.scrollPanel)

    def onScanningVideoSceneClicked(self, event):
        checkedVideo = self.menuScanningVideo.IsChecked()
        checkedScene = self.menuScanningScene.IsChecked()
        profile.putPreference('view_scanning_video', checkedVideo)
        profile.putPreference('view_scanning_scene', checkedScene)
        self.scanningWorkbench.splitterWindow.Unsplit()
        if checkedVideo:
            self.scanningWorkbench.videoView.Show()
            self.scanningWorkbench.splitterWindow.SplitVertically(self.scanningWorkbench.videoView, self.scanningWorkbench.scenePanel)
            if checkedScene:
                self.scanningWorkbench.scenePanel.Show()
            else:
                self.scanningWorkbench.scenePanel.Hide()
                self.scanningWorkbench.splitterWindow.Unsplit()
        else:
            self.scanningWorkbench.videoView.Hide()
            if checkedScene:
                self.scanningWorkbench.scenePanel.Show()
                self.scanningWorkbench.splitterWindow.SplitVertically(self.scanningWorkbench.scenePanel, self.scanningWorkbench.videoView)
                self.scanningWorkbench.splitterWindow.Unsplit()
            else:
                self.scanningWorkbench.scenePanel.Hide()
                self.scanningWorkbench.splitterWindow.Unsplit()

        self.scanningWorkbench.splitterWindow.Layout()
        self.scanningWorkbench.Layout()
        self.Layout()

    def onComboBoxWorkbenchSelected(self, event):
        """ """
        if _(profile.getPreference('workbench')) != event.GetEventObject().GetValue():
            profile.putPreference('workbench', self.workbenchDict[event.GetEventObject().GetValue()])
            profile.saveProfile(os.path.join(profile.getBasePath(), 'current-profile.ini'))
            self.workbenchUpdate()

    def onAbout(self, event):
        """ """
        info = wx.AboutDialogInfo()
        icon = wx.Icon(resources.getPathForImage("horus.ico"), wx.BITMAP_TYPE_ICO)
        info.SetIcon(icon)
        info.SetName(u'Horus')
        info.SetVersion(VERSION)
        techDescription = ''
        if os.path.isfile(resources.getPathForVersion()):
            with open(resources.getPathForVersion(), 'r') as f:
              techDescription = '\n' + f.read().replace('\n','')
        info.SetDescription(_('Horus is an Open Source 3D Scanner manager') + techDescription)
        info.SetCopyright(u'(C) 2014-2015 Mundo Reader S.L.')
        info.SetWebSite(u'http://www.bq.com')
        info.SetLicence("""Horus is free software; you can redistribute
it and/or modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 of the License,
or (at your option) any later version.

Horus is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU General Public License for more details. You should have
received a copy of the GNU General Public License along with File Hunter;
if not, write to the Free Software Foundation, Inc., 59 Temple Place,
Suite 330, Boston, MA  02111-1307  USA""")
        info.AddDeveloper(u'Jesús Arroyo, Irene Sanz, Jorge Robles')
        info.AddDocWriter(u'Jesús Arroyo, Ángel Larrañaga')
        info.AddArtist(u'Jesús Arroyo, Nestor Toribio')
        info.AddTranslator(u'Jesús Arroyo, Irene Sanz, Alexandre Galode, Natasha da Silva')

        wx.AboutBox(info)

    def onWelcome(self, event):
        """ """
        welcome = WelcomeWindow(self)

    def onBoardUnplugged(self):
        self._onDeviceUnplugged(_("Board unplugged"), _("Board has been unplugged. Please, plug it in and press connect"))

    def onCameraUnplugged(self):
        self._onDeviceUnplugged(_("Camera unplugged"), _("Camera has been unplugged. Please, plug it in and press connect"))

    def _onDeviceUnplugged(self, title="", description=""):
        self.simpleScan.stop()
        self.textureScan.stop()
        self.laserTriangulation.cancel()
        self.platformExtrinsics.cancel()
        self.controlWorkbench.updateStatus(False)
        self.calibrationWorkbench.updateStatus(False)
        self.scanningWorkbench.updateStatus(False)
        self.driver.disconnect()
        dlg = wx.MessageDialog(self, description, title, wx.OK|wx.ICON_ERROR)
        dlg.ShowModal()
        dlg.Destroy()

    def updateProfileToAllControls(self):
        """ """
        # if profile.getPreferenceBool('basic_mode'):
        #     self.menuBasicMode.Check(True)
        # else:
        #     self.menuAdvancedMode.Check(True)

        if profile.getPreferenceBool('view_control_panel'):
            self.controlWorkbench.scrollPanel.Show()
            self.menuControlPanel.Check(True)
        else:
            self.controlWorkbench.scrollPanel.Hide()
            self.menuControlPanel.Check(False)

        if profile.getPreferenceBool('view_control_video'):
            self.controlWorkbench.videoView.Show()
            self.menuControlVideo.Check(True)
        else:
            self.controlWorkbench.videoView.Hide()
            self.menuControlVideo.Check(False)

        if profile.getPreferenceBool('view_calibration_panel'):
            self.calibrationWorkbench.scrollPanel.Show()
            self.menuCalibrationPanel.Check(True)
        else:
            self.calibrationWorkbench.scrollPanel.Hide()
            self.menuCalibrationPanel.Check(False)

        if profile.getPreferenceBool('view_calibration_video'):
            self.calibrationWorkbench.videoView.Show()
            self.menuCalibrationVideo.Check(True)
        else:
            self.calibrationWorkbench.videoView.Hide()
            self.menuCalibrationVideo.Check(False)

        if profile.getPreferenceBool('view_scanning_panel'):
            self.scanningWorkbench.scrollPanel.Show()
            self.menuScanningPanel.Check(True)
        else:
            self.scanningWorkbench.scrollPanel.Hide()
            self.menuScanningPanel.Check(False)

        checkedVideo = profile.getPreferenceBool('view_scanning_video')
        checkedScene = profile.getPreferenceBool('view_scanning_scene')

        self.menuScanningVideo.Check(checkedVideo)
        self.menuScanningScene.Check(checkedScene)

        self.scanningWorkbench.splitterWindow.Unsplit()
        if checkedVideo:
            self.scanningWorkbench.videoView.Show()
            self.scanningWorkbench.splitterWindow.SplitVertically(self.scanningWorkbench.videoView, self.scanningWorkbench.scenePanel)
            if checkedScene:
                self.scanningWorkbench.scenePanel.Show()
            else:
                self.scanningWorkbench.scenePanel.Hide()
                self.scanningWorkbench.splitterWindow.Unsplit()
        else:
            self.scanningWorkbench.videoView.Hide()
            if checkedScene:
                self.scanningWorkbench.scenePanel.Show()
                self.scanningWorkbench.splitterWindow.SplitVertically(self.scanningWorkbench.scenePanel, self.scanningWorkbench.videoView)
                self.scanningWorkbench.splitterWindow.Unsplit()
            else:
                self.scanningWorkbench.scenePanel.Hide()
                self.scanningWorkbench.splitterWindow.Unsplit()

        self.updateDriverProfile()
        self.updatePCGProfile()
        self.updateCalibrationProfile()

        self.workbenchUpdate()
        self.Layout()

    def updateDriverProfile(self):
        self.driver.camera.setCameraId(int(profile.getProfileSetting('camera_id')[-1:]))
        self.driver.board.setSerialName(profile.getProfileSetting('serial_name'))
        self.driver.board.setBaudRate(profile.getProfileSettingInteger('baud_rate'))
        self.driver.board.setInvertMotor(profile.getProfileSettingBool('invert_motor'))

    def updatePCGProfile(self):
            self.pcg.resetTheta()
            self.pcg.setViewROI(profile.getProfileSettingBool('view_roi'))
            self.pcg.setROIDiameter(profile.getProfileSettingInteger('roi_diameter'))
            self.pcg.setROIHeight(profile.getProfileSettingInteger('roi_height'))
            self.pcg.setDegrees(profile.getProfileSettingFloat('step_degrees_scanning'))
            resolution = profile.getProfileSetting('resolution_scanning')
            self.pcg.setResolution(int(resolution.split('x')[1]), int(resolution.split('x')[0]))
            useLaser = profile.getProfileSetting('use_laser')
            self.pcg.setUseLaser(useLaser == 'Left' or useLaser == 'Both',
                                 useLaser == 'Right' or useLaser == 'Both')
            self.pcg.setCameraIntrinsics(profile.getProfileSettingNumpy('camera_matrix'),
                                         profile.getProfileSettingNumpy('distortion_vector'))
            self.pcg.setLaserTriangulation(profile.getProfileSettingNumpy('distance_left'),
                                           profile.getProfileSettingNumpy('normal_left'),
                                           profile.getProfileSettingNumpy('distance_right'),
                                           profile.getProfileSettingNumpy('normal_right'))
            self.pcg.setPlatformExtrinsics(profile.getProfileSettingNumpy('rotation_matrix'),
                                           profile.getProfileSettingNumpy('translation_vector'))

            scanType = profile.getProfileSetting('scan_type')
            if scanType == 'Simple Scan':
                self.scanningWorkbench.currentScan = self.simpleScan
                self.driver.camera.setExposure(profile.getProfileSettingInteger('laser_exposure_scanning'))
            elif scanType == 'Texture Scan':
                self.scanningWorkbench.currentScan = self.textureScan
                self.driver.camera.setExposure(profile.getProfileSettingInteger('color_exposure_scanning'))

            self.simpleScan.setFastScan(profile.getProfileSettingBool('fast_scan'))
            self.simpleScan.setSpeedMotor(profile.getProfileSettingInteger('feed_rate_scanning'))
            self.simpleScan.setAccelerationMotor(profile.getProfileSettingInteger('acceleration_scanning'))
            self.simpleScan.setImageType(profile.getProfileSetting('img_type'))
            self.simpleScan.setUseThreshold(profile.getProfileSettingBool('use_cr_threshold'))
            self.simpleScan.setThresholdValue(profile.getProfileSettingInteger('cr_threshold_value'))
            self.simpleScan.setColor(struct.unpack('BBB',profile.getProfileSetting('point_cloud_color').decode('hex')))

            self.textureScan.setFastScan(profile.getProfileSettingBool('fast_scan'))
            self.textureScan.setSpeedMotor(profile.getProfileSettingInteger('feed_rate_scanning'))
            self.textureScan.setAccelerationMotor(profile.getProfileSettingInteger('acceleration_scanning'))
            self.textureScan.setImageType(profile.getProfileSetting('img_type'))
            self.textureScan.setUseOpen(profile.getProfileSettingBool('use_open'))
            self.textureScan.setOpenValue(profile.getProfileSettingInteger('open_value'))
            self.textureScan.setUseThreshold(profile.getProfileSettingBool('use_threshold'))
            self.textureScan.setThresholdValue(profile.getProfileSettingInteger('threshold_value'))

    def updateCalibrationProfile(self):
        self.driver.camera.setIntrinsics(profile.getProfileSettingNumpy('camera_matrix'),
                                         profile.getProfileSettingNumpy('distortion_vector'))

        self.cameraIntrinsics.setIntrinsics(profile.getProfileSettingNumpy('camera_matrix'),
                                            profile.getProfileSettingNumpy('distortion_vector'))
        self.cameraIntrinsics.setPatternParameters(profile.getProfileSettingInteger('pattern_rows'),
                                                   profile.getProfileSettingInteger('pattern_columns'),
                                                   profile.getProfileSettingInteger('square_width'),
                                                   profile.getProfileSettingFloat('pattern_distance'))
        self.cameraIntrinsics.setUseDistortion(profile.getProfileSettingInteger('use_distortion_calibration'))

        self.simpleLaserTriangulation.setIntrinsics(profile.getProfileSettingNumpy('camera_matrix'),
                                                    profile.getProfileSettingNumpy('distortion_vector'))
        self.simpleLaserTriangulation.setPatternParameters(profile.getProfileSettingInteger('pattern_rows'),
                                                           profile.getProfileSettingInteger('pattern_columns'),
                                                           profile.getProfileSettingInteger('square_width'),
                                                           profile.getProfileSettingFloat('pattern_distance'))
        self.simpleLaserTriangulation.setUseDistortion(profile.getProfileSettingInteger('use_distortion_calibration'))

        self.laserTriangulation.setIntrinsics(profile.getProfileSettingNumpy('camera_matrix'),
                                              profile.getProfileSettingNumpy('distortion_vector'))
        self.laserTriangulation.setPatternParameters(profile.getProfileSettingInteger('pattern_rows'),
                                                     profile.getProfileSettingInteger('pattern_columns'),
                                                     profile.getProfileSettingInteger('square_width'),
                                                     profile.getProfileSettingFloat('pattern_distance'))
        self.laserTriangulation.setUseDistortion(profile.getProfileSettingInteger('use_distortion_calibration'))

        self.platformExtrinsics.setExtrinsicsStep(profile.getProfileSettingFloat('extrinsics_step'))
        self.platformExtrinsics.setIntrinsics(profile.getProfileSettingNumpy('camera_matrix'),
                                              profile.getProfileSettingNumpy('distortion_vector'))
        self.platformExtrinsics.setPatternParameters(profile.getProfileSettingInteger('pattern_rows'),
                                                     profile.getProfileSettingInteger('pattern_columns'),
                                                     profile.getProfileSettingInteger('square_width'),
                                                     profile.getProfileSettingFloat('pattern_distance'))
        self.platformExtrinsics.setUseDistortion(profile.getProfileSettingInteger('use_distortion_calibration'))

    def workbenchUpdate(self, layout=True):
        currentWorkbench = profile.getPreference('workbench')

        wb = {'Control workbench'     : self.controlWorkbench,
              'Calibration workbench' : self.calibrationWorkbench,
              'Scanning workbench'    : self.scanningWorkbench}

        waitCursor = wx.BusyCursor()

        self.menuFile.Enable(self.menuLoadModel.GetId(), currentWorkbench == 'Scanning workbench')
        self.menuFile.Enable(self.menuSaveModel.GetId(), currentWorkbench == 'Scanning workbench')
        self.menuFile.Enable(self.menuClearModel.GetId(), currentWorkbench == 'Scanning workbench')

        wb[currentWorkbench].updateProfileToAllControls()
        wb[currentWorkbench].combo.SetValue(_(currentWorkbench))

        if layout:
            for key in wb:
                if wb[key] is not None:
                    if key == currentWorkbench:
                        wb[key].Hide()
                        wb[key].Show()
                    else:
                        wb[key].Hide()

            self.Layout()

        del waitCursor

        gc.collect()

    ##-- TODO: move to util

    def serialList(self):
        baselist=[]
        if os.name=="nt":
            import _winreg
            try:
                key=_winreg.OpenKey(_winreg.HKEY_LOCAL_MACHINE,"HARDWARE\\DEVICEMAP\\SERIALCOMM")
                i=0
                while True:
                    try:
                        values = _winreg.EnumValue(key, i)
                    except:
                        return baselist
                    if 'USBSER' in values[0] or 'VCP' in values[0]:
                        baselist.append(values[1])
                    i+=1
            except:
                return baselist
        else:
            for device in ['/dev/ttyACM*', '/dev/ttyUSB*',  "/dev/tty.usb*", "/dev/tty.wchusb*", "/dev/cu.*", "/dev/rfcomm*"]:
                baselist = baselist + glob.glob(device)
        return baselist

    def baudRateList(self):
        baselist=['9600', '14400', '19200', '38400', '57600', '115200']
        return baselist

    def countCameras(self):
        for i in xrange(5):
            cap = cv2.VideoCapture(i)
            res = not cap.isOpened()
            cap.release()
            if res:
                return i
        return 5

    def videoList(self):
        baselist=[]
        if os.name == 'nt':
            count = self.countCameras()
            for i in xrange(count):
                baselist.append(str(i))
        elif os.sys.platform == 'darwin':
            for device in uvc.Camera_List():
                baselist.append(str(device.src_id))
        else:
            for device in ['/dev/video*']:
                baselist = baselist + glob.glob(device)
        return baselist

    ##-- END TODO