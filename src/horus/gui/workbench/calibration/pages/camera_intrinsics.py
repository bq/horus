# -*- coding: utf-8 -*-
# This file is part of the Horus Project

__author__ = 'Jesús Arroyo Torrens <jesus.arroyo@bq.com>'
__copyright__ = 'Copyright (C) 2014-2016 Mundo Reader S.L.'
__license__ = 'GNU General Public License v2 http://www.gnu.org/licenses/gpl2.html'

import cv2
import random
import wx._core
import numpy as np

from horus.util import profile

from horus.gui.engine import camera_intrinsics, pattern

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg

from horus.gui.workbench.calibration.pages.page import Page
from horus.gui.workbench.calibration.pages.capture_page import CapturePage


class CameraIntrinsicsPages(wx.Panel):

    def __init__(self, parent, start_callback=None, exit_callback=None):
        wx.Panel.__init__(self, parent)

        self.start_callback = start_callback
        self.exit_callback = exit_callback

        # Elements
        self.capture_page = CapturePage(self, start_callback=self.on_start)
        self.result_page = ResultPage(self, exit_callback=self.on_exit)
        self.capture_page.initialize()

        # Layout
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.Add(self.capture_page, 1, wx.ALL | wx.EXPAND, 0)
        hbox.Add(self.result_page, 1, wx.ALL | wx.EXPAND, 0)
        self.SetSizer(hbox)
        self.Layout()

        self._initialize()

    def _initialize(self):
        self.capture_page.initialize()
        self.capture_page.SetFocus()
        self.capture_page.Show()
        self.result_page.Hide()
        self.capture_page.left_button.Enable()

    def play(self):
        self.capture_page.play()

    def stop(self):
        self.capture_page.stop()

    def reset(self):
        self.capture_page.reset()

    def before_calibration(self):
        if self.start_callback is not None:
            self.start_callback()
        self.capture_page.left_button.Disable()
        if not hasattr(self, 'wait_cursor'):
            self.wait_cursor = wx.BusyCursor()

    def after_calibration(self, response):
        self.capture_page.Hide()
        self.result_page.Show()
        self.Layout()
        self.result_page.process_calibration(response)
        if hasattr(self, 'wait_cursor'):
            del self.wait_cursor

    def on_start(self):
        camera_intrinsics.set_callbacks(lambda: wx.CallAfter(self.before_calibration),
                                        None,
                                        lambda r: wx.CallAfter(self.after_calibration, r))
        camera_intrinsics.start()

    def on_exit(self):
        self._initialize()
        if self.exit_callback is not None:
            self.exit_callback()


class ResultPage(Page):

    def __init__(self, parent, exit_callback=None):
        Page.__init__(self, parent,
                      title=_('Camera intrinsics result'),
                      desc='.',
                      left=_('Reject'),
                      right=_('Accept'),
                      button_left_callback=self.on_reject,
                      button_right_callback=self.on_accept)

        self.result = None
        self.exit_callback = exit_callback

        # 3D Plot Panel
        self.plot_panel = CameraIntrinsics3DPlot(self.panel)

        # Layout
        self.panel_box.Add(self.plot_panel, 2, wx.ALL | wx.EXPAND, 3)

    def on_reject(self):
        camera_intrinsics.cancel()
        if self.exit_callback is not None:
            self.exit_callback()

    def on_accept(self):
        camera_intrinsics.accept()
        mtx, dist = self.result
        profile.settings['camera_matrix'] = mtx
        profile.settings['distortion_vector'] = dist
        if self.exit_callback is not None:
            self.exit_callback()

    def process_calibration(self, response):
        self.plot_panel.Hide()
        self.plot_panel.clear()
        ret, result = response

        if ret:
            error, mtx, dist, rvecs, tvecs = result
            text = ' fx: {0}  fy: {1}  cx: {2}  cy: {3}  dist: {4}'.format(
                round(mtx[0][0], 3), round(mtx[1][1], 3),
                round(mtx[0][2], 3), round(mtx[1][2], 3),
                np.round(dist, 2))
            self.result = (mtx, dist)
            self.desc_text.SetLabel(text)
            self.plot_panel.add(error, rvecs, tvecs)
            self.plot_panel.Show()
            self.Layout()
        else:
            if isinstance(result, CameraIntrinsicsError):
                dlg = wx.MessageDialog(
                    self, _("Camera intrinsics calibration has failed. Please try again"),
                    _(result), wx.OK | wx.ICON_ERROR)
                dlg.ShowModal()
                dlg.Destroy()


class CameraIntrinsics3DPlot(wx.Panel):

    def __init__(self, parent):
        wx.Panel.__init__(self, parent)

        self.initialize()

    def initialize(self):
        self.fig = Figure(facecolor=(0.7490196, 0.7490196, 0.7490196, 1), tight_layout=True)
        self.canvas = FigureCanvasWxAgg(self, -1, self.fig)
        self.canvas.SetExtraStyle(wx.EXPAND)

        self.ax = self.fig.gca(projection='3d', facecolor=(0.7490196, 0.7490196, 0.7490196, 1))

        self.print_canvas()

        self.Bind(wx.EVT_SIZE, self.on_size)
        self.Layout()

    def on_size(self, event):
        self.canvas.SetClientSize(self.GetClientSize())
        self.Layout()
        event.Skip()

    def print_canvas(self):
        self.ax.plot([0, 50], [0, 0], [0, 0], linewidth=2.0, color='red')
        self.ax.plot([0, 0], [0, 0], [0, 50], linewidth=2.0, color='green')
        self.ax.plot([0, 0], [0, 50], [0, 0], linewidth=2.0, color='blue')

        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Z')
        self.ax.set_zlabel('Y')
        self.ax.set_xlim(-150, 150)
        self.ax.set_ylim(0, 500)
        self.ax.set_zlim(-150, 150)
        self.ax.invert_xaxis()
        self.ax.invert_yaxis()
        self.ax.invert_zaxis()

    def add(self, error, rvecs, tvecs):
        w = pattern.columns * pattern.square_width
        h = pattern.rows * pattern.square_width

        p = np.array([[0, 0, 0], [w, 0, 0], [w, h, 0], [0, h, 0], [0, 0, 0]])
        n = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]])

        c = np.array([[30, 0, 0], [0, 30, 0], [0, 0, -30]])

        self.ax.text(-100, 200, 0, str(round(error, 5)), fontsize=15)

        for ind, transvector in enumerate(rvecs):

            R = cv2.Rodrigues(transvector)[0]
            t = tvecs[ind]

            points = (np.dot(R, p.T) + np.array([t, t, t, t, t]).T)[0]
            normals = np.dot(R, n.T)

            X = np.array([points[0], normals[0]])
            Y = np.array([points[1], normals[1]])
            Z = np.array([points[2], normals[2]])

            coords = (np.dot(R, c.T) + np.array([t, t, t]).T)[0]

            CX = coords[0]
            CY = coords[1]
            CZ = coords[2]

            color = (random.random(), random.random(), random.random(), 0.8)

            self.ax.plot_surface(X, Z, Y, linewidth=0, color=color)

            self.ax.plot([t[0][0], CX[0]], [t[2][0], CZ[0]],
                         [t[1][0], CY[0]], linewidth=1.0, color='green')
            self.ax.plot([t[0][0], CX[1]], [t[2][0], CZ[1]],
                         [t[1][0], CY[1]], linewidth=1.0, color='red')
            self.ax.plot([t[0][0], CX[2]], [t[2][0], CZ[2]],
                         [t[1][0], CY[2]], linewidth=1.0, color='blue')
            self.canvas.draw()

        self.Layout()

    def clear(self):
        self.ax.cla()
        self.print_canvas()
