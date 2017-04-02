# -*- coding: utf-8 -*-
# This file is part of the Horus Project

__author__ = 'Jesús Arroyo Torrens <jesus.arroyo@bq.com>'
__copyright__ = 'Copyright (C) 2014-2016 Mundo Reader S.L.'
__license__ = 'GNU General Public License v2 http://www.gnu.org/licenses/gpl2.html'

import wx._core

from horus.util import profile, system as sys

from horus.gui.engine import driver

from horus.gui.wizard.connection_page import ConnectionPage
from horus.gui.wizard.calibration_page import CalibrationPage
from horus.gui.wizard.scanning_page import ScanningPage


class Wizard(wx.Dialog):

    def __init__(self, parent):
        if sys.is_windows():
            size = (760, 560)
        else:
            size = (760, 520)
        super(Wizard, self).__init__(parent, title="", size=size)

        self.parent = parent

        self.connection_page = ConnectionPage(
            self,
            button_prev_callback=self.on_connection_page_prev_clicked,
            button_next_callback=self.on_connection_page_next_clicked)
        self.calibration_page = CalibrationPage(
            self,
            button_prev_callback=self.on_calibration_page_prev_clicked,
            button_next_callback=self.on_calibration_page_next_clicked)
        self.scanning_page = ScanningPage(
            self,
            button_prev_callback=self.on_scanning_page_prev_clicked,
            button_next_callback=self.on_scanning_page_next_clicked)

        pages = [self.connection_page, self.calibration_page, self.scanning_page]

        self.connection_page.intialize(pages)
        self.calibration_page.intialize(pages)
        self.scanning_page.intialize(pages)

        self.connection_page.Show()
        self.calibration_page.Hide()
        self.scanning_page.Hide()

        driver.board.set_unplug_callback(lambda: wx.CallAfter(self.on_board_unplugged))
        driver.camera.set_unplug_callback(lambda: wx.CallAfter(self.on_camera_unplugged))

        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.Add(self.connection_page, 1, wx.ALL | wx.EXPAND, 0)
        hbox.Add(self.calibration_page, 1, wx.ALL | wx.EXPAND, 0)
        hbox.Add(self.scanning_page, 1, wx.ALL | wx.EXPAND, 0)

        self.SetSizer(hbox)

        self.Bind(wx.EVT_CLOSE, lambda e: self.on_exit())

        self.Centre()
        self.ShowModal()

    def on_board_unplugged(self):
        self.on_unplugged()
        self.parent.on_board_unplugged()
        self.connection_page.update_status(False)
        self.calibration_page.update_status(False)

    def on_camera_unplugged(self):
        self.on_unplugged()
        self.parent.on_camera_unplugged()
        self.connection_page.update_status(False)
        self.calibration_page.update_status(False)

    def on_unplugged(self):
        if hasattr(self.connection_page, 'wait_cursor'):
            del self.connection_page.wait_cursor
        if hasattr(self.calibration_page, 'wait_cursor'):
            del self.calibration_page.wait_cursor
        self.connection_page.on_unplugged()
        self.calibration_page.on_unplugged()
        self.connection_page.Show()
        self.calibration_page.Hide()
        self.scanning_page.Hide()

    def on_exit(self, message=True):
        result = True
        if message:
            dlg = wx.MessageDialog(
                self, _("Do you really want to exit?"),
                _("Exit wizard"), wx.YES_NO | wx.ICON_INFORMATION)
            result = dlg.ShowModal() == wx.ID_YES
            dlg.Destroy()
        if result:
            driver.board.lasers_off()
            self.connection_page.video_view.stop()
            self.calibration_page.video_view.stop()
            self.scanning_page.video_view.stop()
            self.parent.toolbar.update_status(driver.is_connected)
            self.parent.update_profile_to_all_controls()
            self.EndModal(wx.ID_OK)
            self.Destroy()

    def on_connection_page_prev_clicked(self):
        self.on_exit()

    def on_calibration_page_prev_clicked(self):
        if sys.is_windows():
            self.connection_page.Show()
            self.calibration_page.Hide()
        else:
            self.calibration_page.Hide()
            self.connection_page.Show()
        self.Layout()

    def on_scanning_page_prev_clicked(self):
        if sys.is_windows():
            self.calibration_page.Show()
            self.scanning_page.Hide()
        else:
            self.scanning_page.Hide()
            self.calibration_page.Show()
        self.Layout()

    def on_connection_page_next_clicked(self):
        if sys.is_windows():
            self.calibration_page.Show()
            self.connection_page.Hide()
        else:
            self.connection_page.Hide()
            self.calibration_page.Show()
        self.Layout()

    def on_calibration_page_next_clicked(self):
        if sys.is_windows():
            self.scanning_page.Show()
            self.calibration_page.Hide()
        else:
            self.calibration_page.Hide()
            self.scanning_page.Show()
        self.Layout()

    def on_scanning_page_next_clicked(self):
        driver.board.lasers_off()
        profile.settings.save_settings()
        dlg = wx.MessageDialog(
            self,
            _("You have completed the wizard.\nPress Play button to start scanning."),
            _("Ready to scan!"), wx.OK | wx.ICON_INFORMATION)
        dlg.ShowModal()
        dlg.Destroy()
        self.connection_page.video_view.stop()
        self.calibration_page.video_view.stop()
        self.scanning_page.video_view.stop()
        self.parent.toolbar.update_status(driver.is_connected)
        self.parent.update_profile_to_all_controls()
        self.parent.workbench['scanning'].update_controls()
        profile.settings['workbench'] = 'scanning'
        workbench = self.parent.workbench[profile.settings['workbench']].name
        self.parent.update_workbench(workbench)
        self.EndModal(wx.ID_OK)
        self.Destroy()
