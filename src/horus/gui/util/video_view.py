# -*- coding: utf-8 -*-
# This file is part of the Horus Project

__author__ = 'Jesús Arroyo Torrens <jesus.arroyo@bq.com>'
__copyright__ = 'Copyright (C) 2014-2016 Mundo Reader S.L.'
__license__ = 'GNU General Public License v2 http://www.gnu.org/licenses/gpl2.html'

import wx._core

from threading import Timer
from horus.gui.util.image_view import ImageView


class VideoView(ImageView):

    def __init__(self, parent, callback=None, size=(-1, -1), wxtimer=True):
        ImageView.__init__(self, parent, size=size, black=True)

        self.callback = callback

        self.wxtimer = wxtimer
        self.playing = False

        if self.wxtimer:
            self.timer = wx.Timer(self)
            self.Bind(wx.EVT_TIMER, self.on_timer, self.timer)

    def on_timer(self, event):
        try:
            if self.wxtimer:
                self.timer.Stop()
            else:
                self.timer.cancel()
            if self.playing:
                if self.callback is not None:
                    frame = self.callback()
                    if frame is not None:
                        if self.playing:
                            if self.wxtimer:
                                self.set_frame(frame)
                            else:
                                wx.CallAfter(self.set_frame, frame)
                    self._start()
        except Exception as e:
            print(e.message)

    def set_callback(self, callback):
        self.callback = callback

    def play(self, flush=True):
        if not self.playing:
            self.playing = True
            if self.callback is not None:
                # Flush video
                if flush:
                    self.callback()
                    self.callback()
                frame = self.callback()
                self.set_frame(frame)
            self._start()

    def _start(self):
        if self.wxtimer:
            self.timer.Start(milliseconds=100)
        else:
            self.timer = Timer(0.1, self.on_timer, (None,))
            self.timer.start()

    def stop(self):
        if self.playing:
            self.playing = False
            if self.wxtimer:
                self.timer.Stop()
            else:
                self.timer.cancel()
                self.timer.join()

    def reset(self):
        self.hide = True
        self.set_default_image()
