# -*- coding: utf-8 -*-
# This file is part of the Horus Project

__author__ = 'Jesús Arroyo Torrens <jesus.arroyo@bq.com>'
__copyright__ = 'Copyright (C) 2014-2016 Mundo Reader S.L.'
__license__ = 'GNU General Public License v2 http://www.gnu.org/licenses/gpl2.html'

import wx._core
import numpy as np

from horus.util import profile

from horus.gui.engine import calibration_data, laser_triangulation
from horus.engine.calibration.laser_triangulation import LaserTriangulationError

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg

from horus.gui.workbench.calibration.pages.page import Page
from horus.gui.workbench.calibration.pages.video_page import VideoPage


class LaserTriangulationPages(wx.Panel):

    def __init__(self, parent, start_callback=None, exit_callback=None):
        wx.Panel.__init__(self, parent)  # , style=wx.RAISED_BORDER)

        self.start_callback = start_callback
        self.exit_callback = exit_callback

        self.video_page = VideoPage(self, title=_('Laser triangulation'),
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
        laser_triangulation.set_callbacks(lambda: wx.CallAfter(self.before_calibration),
                                          lambda p: wx.CallAfter(self.progress_calibration, p),
                                          lambda r: wx.CallAfter(self.after_calibration, r))
        laser_triangulation.start()

    def on_exit(self):
        laser_triangulation.cancel()
        self._initialize()
        if self.exit_callback is not None:
            self.exit_callback()


class ResultPage(Page):

    def __init__(self, parent, exit_callback=None):
        Page.__init__(self, parent,
                      title=_('Laser triangulation result'),
                      desc='.',
                      left=_('Reject'),
                      right=_('Accept'),
                      button_left_callback=self.on_reject,
                      button_right_callback=self.on_accept)

        self.result = None
        self.exit_callback = exit_callback

        # 3D Plot Panel
        self.plot_panel = LaserTriangulation3DPlot(self.panel)

        # Layout
        self.panel_box.Add(self.plot_panel, 2, wx.ALL | wx.EXPAND, 3)

    def on_reject(self):
        laser_triangulation.cancel()
        if self.exit_callback is not None:
            self.exit_callback()
        self.plot_panel.clear()

    def on_accept(self):
        laser_triangulation.accept()
        dL, nL, dR, nR = self.result
        profile.settings['distance_left'] = dL
        profile.settings['normal_left'] = nL
        profile.settings['distance_right'] = dR
        profile.settings['normal_right'] = nR
        profile.settings['laser_triangulation_hash'] = calibration_data.md5_hash()
        if self.exit_callback is not None:
            self.exit_callback()
        self.plot_panel.clear()

    def process_calibration(self, response):
        ret, result = response

        if ret:
            dL = result[0][0]
            nL = result[0][1]
            stdL = result[0][2]
            dR = result[1][0]
            nR = result[1][1]
            stdR = result[1][2]
            self.result = (dL, nL, dR, nR)
            np.set_printoptions(formatter={'float': '{:g}'.format})
            text = ' L: {0} {1}  R: {2} {3}'.format(
                   round(dL, 3), np.round(nL, 3),
                   round(dR, 3), np.round(nR, 3))
            np.set_printoptions()
            self.desc_text.SetLabel(text)
            self.plot_panel.clear()
            self.plot_panel.add((dL, nL, stdL, dR, nR, stdR))
            self.plot_panel.Show()
            self.Layout()
            dlg = wx.MessageDialog(
                self, _("Lasers calibrated correctly"),
                _("Success"), wx.OK | wx.ICON_INFORMATION)
            dlg.ShowModal()
            dlg.Destroy()
            self.Layout()
        else:
            if isinstance(result, LaserTriangulationError):
                dlg = wx.MessageDialog(
                    self, _("Laser triangulation calibration has failed. "
                            "Please check the pattern and the lasers and try again. "
                            "Also you can set up the calibration's settings "
                            "in the \"Adjustment workbench\" until the pattern "
                            "and the lasers are detected correctly"),
                    _(result), wx.OK | wx.ICON_ERROR)
                dlg.ShowModal()
                dlg.Destroy()


class LaserTriangulation3DPlot(wx.Panel):

    def __init__(self, parent):
        wx.Panel.__init__(self, parent)

        self.initialize()

    def initialize(self):
        fig = Figure(facecolor=(0.7490196, 0.7490196, 0.7490196, 1), tight_layout=True)
        self.canvas = FigureCanvasWxAgg(self, -1, fig)
        self.canvas.SetExtraStyle(wx.EXPAND)
        self.ax = fig.gca(projection='3d', axisbg=(0.7490196, 0.7490196, 0.7490196, 1))

        self.Bind(wx.EVT_SIZE, self.onSize)
        self.Layout()

    def onSize(self, event):
        self.canvas.SetClientSize(self.GetClientSize())
        self.canvas.draw()
        self.Layout()

    def add(self, args):
        dL, nL, stdL, dR, nR, stdR = args

        rL = np.cross(np.array([0, 0, 1]), nL)
        sL = np.cross(rL, nL)
        RL = np.array([rL, sL, nL])

        rR = np.cross(np.array([0, 0, 1]), nR)
        sR = np.cross(rR, nR)
        RR = np.array([rR, sR, nR])

        self.addPlane(RL, dL * nL)
        self.addPlane(RR, dR * nR)

        self.ax.plot([0, 50], [0, 0], [0, 0], linewidth=2.0, color='red')
        self.ax.plot([0, 0], [0, 0], [0, 50], linewidth=2.0, color='green')
        self.ax.plot([0, 0], [0, 50], [0, 0], linewidth=2.0, color='blue')

        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Z')
        self.ax.set_zlabel('Y')

        self.ax.text(-100, 0, 0, str(round(stdL, 5)), fontsize=15)
        self.ax.text(100, 0, 0, str(round(stdR, 5)), fontsize=15)

        self.ax.set_xlim(-150, 150)
        self.ax.set_ylim(0, 400)
        self.ax.set_zlim(-150, 150)

        self.ax.invert_xaxis()
        self.ax.invert_yaxis()
        self.ax.invert_zaxis()

        self.canvas.draw()
        self.Layout()

    def addPlane(self, R, t):
        w = 200
        h = 300

        p = np.array([[-w / 2, -h / 2, 0], [-w / 2, h / 2, 0],
                      [w / 2, h / 2, 0], [w / 2, -h / 2, 0], [-w / 2, -h / 2, 0]])
        n = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]])

        self.ax.plot([0, t[0]], [0, t[2]], [0, t[1]], linewidth=2.0, color='yellow')

        points = np.dot(R.T, p.T) + np.array([t, t, t, t, t]).T
        normals = np.dot(R.T, n.T)

        X = np.array([points[0], normals[0]])
        Y = np.array([points[1], normals[1]])
        Z = np.array([points[2], normals[2]])

        self.ax.plot_surface(X, Z, Y, linewidth=0, color=(1, 0, 0, 0.8))

        self.canvas.draw()

    def clear(self):
        self.ax.cla()
