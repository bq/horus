# -*- coding: utf-8 -*-
# This file is part of the Horus Project

__author__ = 'Jesús Arroyo Torrens <jesus.arroyo@bq.com>'
__copyright__ = 'Copyright (C) 2014-2016 Mundo Reader S.L.'
__license__ = 'GNU General Public License v2 http://www.gnu.org/licenses/gpl2.html'

import os
import wx._core

from horus.util import profile, resources

from horus.gui.wizard.main import Wizard
from horus.gui.util.image_view import ImageView


class WelcomeDialog(wx.Dialog):

    def __init__(self, parent):
        wx.Dialog.__init__(self, parent, size=(640 + 120, 480 + 40),
                           style=wx.DEFAULT_FRAME_STYLE ^ wx.RESIZE_BORDER)

        self.parent = parent
        self.last_files = profile.settings['last_files']

        # Elements
        header = Header(self)
        content = Content(self)
        check_box_show = wx.CheckBox(
            self, label=_("Don't show this dialog again"), style=wx.ALIGN_LEFT)
        check_box_show.SetValue(not profile.settings['show_welcome'])

        # Layout
        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(header, 2, wx.ALL | wx.EXPAND, 1)
        vbox.Add(content, 3, wx.ALL | wx.EXPAND ^ wx.BOTTOM, 20)
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.Add((0, 0), 1, wx.ALL | wx.EXPAND, 0)
        hbox.Add(check_box_show, 0, wx.ALL, 0)
        vbox.Add(hbox, 0, wx.ALL | wx.EXPAND, 15)
        self.SetSizer(vbox)
        self.Centre()

        # Events
        check_box_show.Bind(wx.EVT_CHECKBOX, self.on_check_box_changed)
        self.Bind(wx.EVT_CLOSE, self.on_close)

        self.ShowModal()

    def on_check_box_changed(self, event):
        profile.settings['show_welcome'] = not event.Checked()

    def on_close(self, event):
        self.EndModal(wx.ID_OK)
        self.Destroy()


class Header(wx.Panel):

    def __init__(self, parent):
        wx.Panel.__init__(self, parent)

        # Elements
        logo = ImageView(self)
        logo.set_image(wx.Image(resources.get_path_for_image("logo.png")))
        title_text = wx.StaticText(self, label=_("3D scanning for everyone"))
        title_font = title_text.GetFont()
        title_font.SetPointSize(13)
        title_text.SetFont(title_font)
        separator = wx.StaticLine(self, -1, style=wx.LI_HORIZONTAL)

        # Layout
        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(logo, 10, wx.ALL ^ wx.BOTTOM | wx.EXPAND, 30)
        vbox.Add(title_text, 0, wx.TOP | wx.CENTER, 20)
        vbox.Add((0, 0), 1, wx.ALL | wx.EXPAND, 0)
        vbox.Add(separator, 0, wx.ALL | wx.EXPAND, 10)
        self.SetSizer(vbox)
        self.Layout()


class CreateNew(wx.Panel):

    def __init__(self, parent):
        wx.Panel.__init__(self, parent)

        # Elements
        title_text = wx.StaticText(self, label=_("Create new"))
        wizard_button = wx.Button(self, label=_("Wizard mode (step by step)"))
        scan_button = wx.Button(self, label=_("Scan using recent settings"))
        # advanced_control_button = wx.Button(self, label=_("Advanced control"))
        advanced_adjustment_button = wx.Button(self, label=_("Advanced adjustment"))
        advanced_calibration_button = wx.Button(self, label=_("Advanced calibration"))

        # Layout
        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(title_text, 0, wx.BOTTOM | wx.CENTER, 10)
        vbox.Add(wizard_button, 1, wx.ALL | wx.EXPAND, 5)
        vbox.Add(scan_button, 1, wx.ALL | wx.EXPAND, 5)
        # vbox.Add(advanced_control_button, 1, wx.TOP | wx.BOTTOM | wx.EXPAND, 5)
        vbox.Add(advanced_adjustment_button, 1, wx.ALL | wx.EXPAND, 5)
        vbox.Add(advanced_calibration_button, 1, wx.ALL | wx.EXPAND, 5)
        self.SetSizer(vbox)
        self.Layout()

        # Events
        wizard_button.Bind(wx.EVT_BUTTON, self.on_wizard)
        scan_button.Bind(wx.EVT_BUTTON, self.on_scan)
        # advanced_control_button.Bind(wx.EVT_BUTTON, self.on_advanced_control)
        advanced_adjustment_button.Bind(wx.EVT_BUTTON, self.on_advanced_adjustment)
        advanced_calibration_button.Bind(wx.EVT_BUTTON, self.on_advanced_calibration)

    def on_wizard(self, event):
        parent = self.GetParent().GetParent()
        parent.Hide()
        Wizard(parent.parent)

    def on_scan(self, event):
        profile.settings['workbench'] = 'scanning'
        parent = self.GetParent().GetParent()
        workbench = parent.parent.workbench[profile.settings['workbench']].name
        parent.parent.update_workbench(workbench)
        parent.Close()

    def on_advanced_control(self, event):
        profile.settings['workbench'] = 'control'
        parent = self.GetParent().GetParent()
        workbench = parent.parent.workbench[profile.settings['workbench']].name
        parent.parent.update_workbench(workbench)
        parent.Close()

    def on_advanced_adjustment(self, event):
        profile.settings['workbench'] = 'adjustment'
        parent = self.GetParent().GetParent()
        workbench = parent.parent.workbench[profile.settings['workbench']].name
        parent.parent.update_workbench(workbench)
        parent.Close()

    def on_advanced_calibration(self, event):
        profile.settings['workbench'] = 'calibration'
        parent = self.GetParent().GetParent()
        workbench = parent.parent.workbench[profile.settings['workbench']].name
        parent.parent.update_workbench(workbench)
        parent.Close()


class OpenRecent(wx.Panel):

    def __init__(self, parent):
        wx.Panel.__init__(self, parent)

        # Elements
        title_text = wx.StaticText(self, label=_("Open recent file"))

        lastFiles = profile.settings['last_files']
        lastFiles.reverse()

        # Layout
        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(title_text, 0, wx.BOTTOM | wx.CENTER, 10)

        for path in lastFiles:
            button = wx.Button(self, label=os.path.basename(path), name=path)
            button.Bind(wx.EVT_BUTTON, self.on_button_pressed)
            vbox.Add(button, 0, wx.ALL | wx.EXPAND, 5)

        self.SetSizer(vbox)
        self.Layout()

    def on_button_pressed(self, event):
        button = event.GetEventObject()
        profile.settings['workbench'] = 'scanning'
        parent = self.GetParent().GetParent()
        workbench = parent.parent.workbench[profile.settings['workbench']].name
        parent.parent.update_workbench(workbench)
        parent.parent.append_last_file(button.GetName())
        parent.parent.workbench['scanning'].scene_view.load_file(button.GetName())
        parent.Close()


class Content(wx.Panel):

    def __init__(self, parent):
        wx.Panel.__init__(self, parent)

        # Elements
        create_new = CreateNew(self)
        open_recent = OpenRecent(self)

        # Layout
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.Add(create_new, 1, wx.ALL | wx.EXPAND, 10)
        hbox.Add(wx.StaticLine(self, style=wx.LI_VERTICAL), 0, wx.LEFT | wx.RIGHT | wx.EXPAND, 20)
        hbox.Add(open_recent, 1, wx.ALL | wx.EXPAND, 10)
        self.SetSizer(hbox)
        self.Layout()
