# -*- coding: utf-8 -*-
# This file is part of the Horus Project

__author__ = 'Jesús Arroyo Torrens <jesus.arroyo@bq.com>'
__copyright__ = 'Copyright (C) 2014-2016 Mundo Reader S.L.'
__license__ = 'GNU General Public License v2 http://www.gnu.org/licenses/gpl2.html'

import wx._core

from horus.gui.engine import scanner_autocheck, image_capture
from horus.engine.calibration.autocheck import PatternNotDetected, WrongMotorDirection, \
    LaserNotDetected

from horus.gui.workbench.calibration.pages.video_page import VideoPage


class ScannerAutocheckPages(wx.Panel):

    def __init__(self, parent, start_callback=None, exit_callback=None):
        wx.Panel.__init__(self, parent)  # , style=wx.RAISED_BORDER)

        self.start_callback = start_callback
        self.exit_callback = exit_callback

        # Elements
        self.video_page = VideoPage(self, title=_('Scanner autocheck'),
                                    start_callback=self.on_start,
                                    cancel_callback=self.on_cancel)

        # Layout
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.Add(self.video_page, 1, wx.ALL | wx.EXPAND, 0)
        self.SetSizer(hbox)

        self._initialize()
        self.Layout()

    def _initialize(self):
        self.video_page.initialize()

    def play(self):
        self.video_page.play()

    def stop(self):
        self.video_page.stop()

    def reset(self):
        self.video_page.reset()

    def before_calibration(self):
        if self.start_callback is not None:
            self.start_callback()
        self.video_page.right_button.Disable()
        if not hasattr(self, 'wait_cursor'):
            self.wait_cursor = wx.BusyCursor()

    def progress_calibration(self, progress):
        self.video_page.gauge.SetValue(progress)

    def after_calibration(self, response):
        ret, result = response

        # Flush video
        image_capture.capture_pattern()
        image_capture.capture_pattern()

        if ret:
            dlg = wx.MessageDialog(
                self, _("Scanner configured correctly"),
                _("Success"), wx.OK | wx.ICON_INFORMATION)
            dlg.ShowModal()
            dlg.Destroy()
        else:
            if isinstance(result, PatternNotDetected):
                dlg = wx.MessageDialog(
                    self, _("Please, put the pattern on the platform. "
                            "Also you can set up the calibration's capture "
                            "settings in the \"Adjustment workbench\" "
                            "until the pattern is detected correctly"),
                    _(result), wx.OK | wx.ICON_ERROR)
                dlg.ShowModal()
                dlg.Destroy()
            elif isinstance(result, WrongMotorDirection):
                dlg = wx.MessageDialog(
                    self, _(
                        "Please, select \"Invert the motor direction\" in the preferences"),
                    _(result), wx.OK | wx.ICON_ERROR)
                dlg.ShowModal()
                dlg.Destroy()
                self.GetParent().GetParent().launch_preferences(basic=True)
            elif isinstance(result, LaserNotDetected):
                dlg = wx.MessageDialog(
                    self, _("Please, check the lasers connection. "
                            "Also you can set up the calibration's capture and "
                            "segmentation settings in the \"Adjustment workbench\" "
                            "until the lasers are detected correctly"),
                    _(result), wx.OK | wx.ICON_ERROR)
                dlg.ShowModal()
                dlg.Destroy()
        self._initialize()
        self.video_page.right_button.Enable()
        if hasattr(self, 'wait_cursor'):
            del self.wait_cursor
        if self.exit_callback is not None:
            self.exit_callback()

    def on_start(self):
        scanner_autocheck.set_callbacks(lambda: wx.CallAfter(self.before_calibration),
                                        lambda p: wx.CallAfter(self.progress_calibration, p),
                                        lambda r: wx.CallAfter(self.after_calibration, r))
        scanner_autocheck.start()

    def on_cancel(self):
        scanner_autocheck.cancel()
        self._initialize()
        self.video_page.right_button.Enable()
        if hasattr(self, 'wait_cursor'):
            del self.wait_cursor
        if self.exit_callback is not None:
            self.exit_callback()
