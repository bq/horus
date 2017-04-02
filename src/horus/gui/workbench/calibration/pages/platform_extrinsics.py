# -*- coding: utf-8 -*-
# This file is part of the Horus Project

__author__ = 'Jesús Arroyo Torrens <jesus.arroyo@bq.com>'
__copyright__ = 'Copyright (C) 2014-2016 Mundo Reader S.L.'
__license__ = 'GNU General Public License v2 http://www.gnu.org/licenses/gpl2.html'

import wx._core
import numpy as np

from horus.util import profile

from horus.gui.engine import pattern, calibration_data, platform_extrinsics
from horus.gui.util.pattern_distance_window import PatternDistanceWindow
from horus.engine.calibration.platform_extrinsics import PlatformExtrinsicsError

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg

from horus.gui.workbench.calibration.pages.page import Page
from horus.gui.workbench.calibration.pages.video_page import VideoPage


class PlatformExtrinsicsPages(wx.Panel):

    def __init__(self, parent, start_callback=None, exit_callback=None):
        wx.Panel.__init__(self, parent)  # , style=wx.RAISED_BORDER)

        self.start_callback = start_callback
        self.exit_callback = exit_callback

        self.video_page = VideoPage(self, title=_('Platform extrinsics'),
                                    start_callback=self.on_start, cancel_callback=self.on_exit)
        self.result_page = ResultPage(self, exit_callback=self.on_exit)

        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.Add(self.video_page, 1, wx.ALL | wx.EXPAND, 0)
        hbox.Add(self.result_page, 1, wx.ALL | wx.EXPAND, 0)
        self.SetSizer(hbox)
        self.Layout()

        self._initialize()

    def _initialize(self):
        self.video_page.initialize()
        self.result_page.Hide()
        self.video_page.Show()
        self.video_page.play()
        self.video_page.right_button.Enable()
        self.GetParent().Layout()
        self.Layout()

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
        if ret:
            self.video_page.Hide()
            self.video_page.stop()
            self.result_page.Show()
            self.Layout()
        else:
            self.on_exit()
        self.result_page.process_calibration(response)
        if hasattr(self, 'wait_cursor'):
            del self.wait_cursor

    def on_start(self):
        if profile.settings['pattern_origin_distance'] == 0.0:
            PatternDistanceWindow(self)
        else:
            platform_extrinsics.set_callbacks(lambda: wx.CallAfter(self.before_calibration),
                                              lambda p: wx.CallAfter(self.progress_calibration, p),
                                              lambda r: wx.CallAfter(self.after_calibration, r))
            platform_extrinsics.start()

    def on_exit(self):
        platform_extrinsics.cancel()
        self._initialize()
        if self.exit_callback is not None:
            self.exit_callback()


class ResultPage(Page):

    def __init__(self, parent, exit_callback=None):
        Page.__init__(self, parent,
                      title=_('Platform extrinsics result'),
                      desc='.',
                      left=_('Reject'),
                      right=_('Accept'),
                      button_left_callback=self.on_reject,
                      button_right_callback=self.on_accept)

        self.result = None
        self.exit_callback = exit_callback

        # 3D Plot Panel
        self.plot_panel = PlatformExtrinsics3DPlot(self.panel)

        # Layout
        self.panel_box.Add(self.plot_panel, 2, wx.ALL | wx.EXPAND, 3)

    def on_reject(self):
        platform_extrinsics.cancel()
        if self.exit_callback is not None:
            self.exit_callback()
        self.plot_panel.clear()

    def on_accept(self):
        platform_extrinsics.accept()
        R, t = self.result
        profile.settings['rotation_matrix'] = R
        profile.settings['translation_vector'] = t
        profile.settings['platform_extrinsics_hash'] = calibration_data.md5_hash()
        if self.exit_callback is not None:
            self.exit_callback()
        self.plot_panel.clear()

    def process_calibration(self, response):
        ret, result = response

        if ret:
            R = result[0]
            t = result[1]
            self.result = (R, t)
            np.set_printoptions(formatter={'float': '{:g}'.format})
            text = ' R: {0}  t: {1}'.format(
                   np.round(R, 2), np.round(t, 4)).replace('\n', '')
            np.set_printoptions()
            self.desc_text.SetLabel(text)
            self.plot_panel.clear()
            self.plot_panel.add(result)
            self.plot_panel.Show()
            self.Layout()
            dlg = wx.MessageDialog(
                self, _("Platform calibrated correctly"),
                _("Success"), wx.OK | wx.ICON_INFORMATION)
            dlg.ShowModal()
            dlg.Destroy()
            self.Layout()
        else:
            if isinstance(result, PlatformExtrinsicsError):
                dlg = wx.MessageDialog(
                    self, _("Platform extrinsics calibration has failed. "
                            "Please check the pattern and try again. "
                            "Also you can set up the calibration's capture settings "
                            "in the \"Adjustment workbench\" until the pattern "
                            "is detected correctly"),
                    _(result), wx.OK | wx.ICON_ERROR)
                dlg.ShowModal()
                dlg.Destroy()


class PlatformExtrinsics3DPlot(wx.Panel):

    def __init__(self, parent):
        wx.Panel.__init__(self, parent)

        self.initialize()

    def initialize(self):
        fig = Figure(facecolor=(0.7490196, 0.7490196, 0.7490196, 1), tight_layout=True)
        self.canvas = FigureCanvasWxAgg(self, -1, fig)
        self.canvas.SetExtraStyle(wx.EXPAND)
        self.ax = fig.gca(projection='3d', axisbg=(0.7490196, 0.7490196, 0.7490196, 1))

        self.Bind(wx.EVT_SIZE, self.on_size)
        self.Layout()

    def on_size(self, event):
        self.canvas.SetClientSize(self.GetClientSize())
        self.canvas.draw()
        self.Layout()

    def add(self, args):
        R, t, center, point, normal, [x, y, z], circle = args

        # plot the surface, data, and synthetic circle
        self.ax.scatter(x, z, y, c='b', marker='o')
        # self.ax.scatter(center[0], center[2], center[1], c='b', marker='o')
        self.ax.plot(circle[0], circle[2], circle[1], c='r')

        d = pattern.origin_distance

        self.ax.plot([t[0], t[0] + 50 * R[0][0]], [t[2], t[2] + 50 * R[2][0]],
                     [t[1], t[1] + 50 * R[1][0]], linewidth=2.0, color='red')
        self.ax.plot([t[0], t[0] + 50 * R[0][1]], [t[2], t[2] + 50 * R[2][1]],
                     [t[1], t[1] + 50 * R[1][1]], linewidth=2.0, color='green')
        self.ax.plot([t[0], t[0] + d * R[0][2]], [t[2], t[2] + d * R[2][2]],
                     [t[1], t[1] + d * R[1][2]], linewidth=2.0, color='blue')

        self.ax.plot([0, 50], [0, 0], [0, 0], linewidth=2.0, color='red')
        self.ax.plot([0, 0], [0, 0], [0, 50], linewidth=2.0, color='green')
        self.ax.plot([0, 0], [0, 50], [0, 0], linewidth=2.0, color='blue')

        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Z')
        self.ax.set_zlabel('Y')

        self.ax.set_xlim(-150, 150)
        self.ax.set_ylim(0, 400)
        self.ax.set_zlim(-150, 150)

        self.ax.invert_xaxis()
        self.ax.invert_yaxis()
        self.ax.invert_zaxis()

        self.canvas.draw()
        self.Layout()

    def clear(self):
        self.ax.cla()
