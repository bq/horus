# -*- coding: utf-8 -*-
# This file is part of the Horus Project

__author__ = 'Jesús Arroyo Torrens <jesus.arroyo@bq.com>'
__copyright__ = 'Copyright (C) 2014-2016 Mundo Reader S.L.'
__license__ = 'GNU General Public License v2 http://www.gnu.org/licenses/gpl2.html'

import gc
import os
import time
import wx._core
import datetime
import webbrowser
from collections import OrderedDict

from horus import __version__, __datetime__, __commit__
from horus.gui.engine import driver, image_capture, ciclop_scan, scanner_autocheck, \
    laser_triangulation, platform_extrinsics

from horus.gui.welcome import WelcomeDialog
from horus.gui.util.preferences import PreferencesDialog
# from horus.gui.util.machine_settings import MachineSettingsDialog  # add in future version

from horus.gui.workbench.toolbar import ToolbarConnection
from horus.gui.workbench.control.main import ControlWorkbench
from horus.gui.workbench.adjustment.main import AdjustmentWorkbench
from horus.gui.workbench.calibration.main import CalibrationWorkbench
from horus.gui.workbench.scanning.main import ScanningWorkbench

from horus.gui.wizard.main import Wizard
from horus.gui.util.version_window import VersionWindow

from horus.util import profile, resources, mesh_loader, version, system as sys

import logging
logger = logging.getLogger(__name__)

__title__ = "Horus " + __version__


class MainWindow(wx.Frame):

    def __init__(self):
        wx.Frame.__init__(self, None, title=__title__, size=(980, 623))

        logger.info("Start application " + __title__)

        # Initialize driver
        self.initialize_driver()
        self.last_files = profile.settings['last_files']

        # Initialize GUI
        self.load_menu()
        self.load_workbenches()
        self.update_profile_to_all_controls()

        ws, hs = self.GetSize()
        x, y, w, h = wx.Display(0).GetGeometry()
        self.SetMinSize((600, 450))
        self.SetPosition((x + (w - ws) / 2., y + (h - hs) / 2.))
        self.SetIcon(wx.Icon(resources.get_path_for_image("horus.ico"), wx.BITMAP_TYPE_ICO))

        self.Bind(wx.EVT_CLOSE, self.on_close)

    def __del__(self):
        logger.info("Finish application " + __title__)

    def load_workbenches(self):
        self.toolbar = ToolbarConnection(self, self.on_connect, self.on_disconnect)
        self.workbench = OrderedDict()
        self.workbench['control'] = ControlWorkbench(self)
        self.workbench['adjustment'] = AdjustmentWorkbench(self)
        self.workbench['calibration'] = CalibrationWorkbench(self)
        self.workbench['scanning'] = ScanningWorkbench(self, self.toolbar.toolbar_scan)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.toolbar, 0, wx.ALL | wx.EXPAND)
        self.Bind(wx.EVT_COMBOBOX, self.on_combo_box_selected, self.toolbar.combo)

        for workbench in self.workbench.values():
            self.toolbar.combo.Append(workbench.name)
            sizer.Add(workbench, 1, wx.ALL | wx.EXPAND)
        name = self.workbench[profile.settings['workbench']].name
        self.update_workbench(name)
        self.SetSizer(sizer)

        self.workbench['scanning'].scene_view.set_point_size(profile.settings['point_size'])

    def load_menu(self):
        self.menu_bar = wx.MenuBar()

        # Menu File
        self.menu_file = wx.Menu()
        self.menu_launch_wizard = self.menu_file.Append(wx.NewId(), _("Launch wizard"))
        self.menu_file.AppendSeparator()
        self.menu_load_model = self.menu_file.Append(wx.NewId(), _("Open model"))
        self.menu_save_model = self.menu_file.Append(wx.NewId(), _("Save model"))
        self.menu_clear_model = self.menu_file.Append(wx.NewId(), _("Clear model"))
        self.menu_file.AppendSeparator()
        self.menu_open_profile = self.menu_file.Append(
            wx.NewId(), _("Open profile"), _("Open profile .json"))
        self.menu_save_profile = self.menu_file.Append(
            wx.NewId(), _("Save profile"), _("Save profile .json"))
        self.menu_reset_profile = self.menu_file.Append(
            wx.NewId(), _("Reset profile"), _("Reset default values"))
        self.menu_file.AppendSeparator()
        self.menu_open_calibration_profile = self.menu_file.Append(
            wx.NewId(), _("Open calibration"), _("Open calibration .json"))
        self.menu_save_calibration_profile = self.menu_file.Append(
            wx.NewId(), _("Save calibration"), _("Save calibration .json"))
        self.menu_reset_calibration_profile = self.menu_file.Append(
            wx.NewId(), _("Reset calibration"), _("Reset calibration default values"))
        self.menu_file.AppendSeparator()
        self.menu_export_log = self.menu_file.Append(
            wx.NewId(), _("Export log"), _("Export log file"))
        self.menu_clear_log = self.menu_file.Append(
            wx.NewId(), _("Clear log"), _("Clear log file"))
        self.menu_file.AppendSeparator()
        self.menu_exit = self.menu_file.Append(wx.ID_EXIT, _("Exit"))
        self.menu_bar.Append(self.menu_file, _("File"))

        # Menu Edit
        self.menu_edit = wx.Menu()
        self.menu_preferences = self.menu_edit.Append(wx.NewId(), _("Preferences"))
        # self.menu_machine_settings = self.menu_edit.Append(wx.NewId(), _("Machine settings"))
        self.menu_bar.Append(self.menu_edit, _("Edit"))

        # Menu View
        self.menu_view = wx.Menu()
        self.menu_control = wx.Menu()
        self.menu_scanning = wx.Menu()
        self.menu_scanning_panel = self.menu_scanning.AppendCheckItem(wx.NewId(), _("Panel"))
        self.menu_scanning_video = self.menu_scanning.AppendCheckItem(wx.NewId(), _("Video"))
        self.menu_scanning_scene = self.menu_scanning.AppendCheckItem(wx.NewId(), _("Scene"))
        self.menu_view.AppendMenu(wx.NewId(), _("Scanning"), self.menu_scanning)
        self.menu_mode_advanced = self.menu_view.AppendCheckItem(wx.NewId(), _("Advanced mode"))
        self.menu_bar.Append(self.menu_view, _("View"))

        # Menu Help
        self.menu_help = wx.Menu()
        self.menu_welcome = self.menu_help.Append(wx.ID_ANY, _("Welcome"))
        if profile.settings['check_for_updates']:
            self.menu_updates = self.menu_help.Append(wx.ID_ANY, _("Updates"))
        self.menu_sources = self.menu_help.Append(wx.ID_ANY, _("Sources"))
        self.menu_about = self.menu_help.Append(wx.ID_ABOUT, _("About"))
        self.menu_bar.Append(self.menu_help, _("Help"))

        self.SetMenuBar(self.menu_bar)

        # Events
        self.Bind(wx.EVT_MENU, self.on_launch_wizard, self.menu_launch_wizard)
        self.Bind(wx.EVT_MENU, self.on_load_model, self.menu_load_model)
        self.Bind(wx.EVT_MENU, self.on_save_model, self.menu_save_model)
        self.Bind(wx.EVT_MENU, self.on_clear_model, self.menu_clear_model)
        self.Bind(wx.EVT_MENU, lambda e: self.on_open_profile("profile_settings"),
                  self.menu_open_profile)
        self.Bind(wx.EVT_MENU, lambda e: self.on_save_profile("profile_settings"),
                  self.menu_save_profile)
        self.Bind(wx.EVT_MENU, lambda e: self.on_reset_profile("profile_settings"),
                  self.menu_reset_profile)
        self.Bind(wx.EVT_MENU, lambda e: self.on_open_profile("calibration_settings"),
                  self.menu_open_calibration_profile)
        self.Bind(wx.EVT_MENU, lambda e: self.on_save_profile("calibration_settings"),
                  self.menu_save_calibration_profile)
        self.Bind(wx.EVT_MENU, lambda e: self.on_reset_profile("calibration_settings"),
                  self.menu_reset_calibration_profile)
        self.Bind(wx.EVT_MENU, self.on_export_log, self.menu_export_log)
        self.Bind(wx.EVT_MENU, self.on_clear_log, self.menu_clear_log)
        self.Bind(wx.EVT_MENU, self.on_exit, self.menu_exit)

        self.Bind(wx.EVT_MENU, self.on_preferences, self.menu_preferences)
        # self.Bind(wx.EVT_MENU, self.on_machine_settings, self.menu_machine_settings)

        self.Bind(wx.EVT_MENU, self.on_scanning_panel_clicked, self.menu_scanning_panel)
        self.Bind(wx.EVT_MENU, self.on_scanning_video_scene_clicked, self.menu_scanning_video)
        self.Bind(wx.EVT_MENU, self.on_scanning_video_scene_clicked, self.menu_scanning_scene)
        self.Bind(wx.EVT_MENU, self.on_mode_advanced_clicked, self.menu_mode_advanced)

        self.Bind(wx.EVT_MENU, self.on_about, self.menu_about)
        self.Bind(wx.EVT_MENU, self.on_welcome, self.menu_welcome)
        if profile.settings['check_for_updates']:
            self.Bind(wx.EVT_MENU, self.on_updates, self.menu_updates)
        self.Bind(wx.EVT_MENU, lambda e: webbrowser.open(
            'https://github.com/LibreScanner/horus'), self.menu_sources)

    def on_launch_wizard(self, event):
        self.workbench[profile.settings['workbench']].on_close()
        Wizard(self)

    def on_load_model(self, event):
        last_file = os.path.split(profile.settings['last_file'])[0]
        dlg = wx.FileDialog(
            self, _("Open 3D model"), last_file, style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
        wildcard_list = ';'.join(map(lambda s: '*' + s, mesh_loader.load_supported_extensions()))
        wildcard_filter = "All (%s)|%s;%s" % (wildcard_list, wildcard_list, wildcard_list.upper())
        wildcard_list = ';'.join(map(lambda s: '*' + s, mesh_loader.load_supported_extensions()))
        wildcard_filter += "|Mesh files (%s)|%s;%s" % (wildcard_list, wildcard_list,
                                                       wildcard_list.upper())
        dlg.SetWildcard(wildcard_filter)
        if dlg.ShowModal() == wx.ID_OK:
            filename = dlg.GetPath()
            if filename is not None:
                self.workbench['scanning'].scene_view.load_file(filename)
                self.append_last_file(filename)
        dlg.Destroy()

    def on_save_model(self, event):
        if self.workbench['scanning'].scene_view._object is None or \
           not self.workbench['scanning'].scene_view._object._is_point_cloud:
            return
        dlg = wx.FileDialog(self, _("Save 3D model"), os.path.split(
            profile.settings['last_file'])[0], style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
        file_extensions = mesh_loader.save_supported_extensions()
        wildcard_list = ';'.join(map(lambda s: '*' + s, file_extensions))
        wildcard_filter = "Mesh files (%s)|%s;%s" % (wildcard_list, wildcard_list,
                                                     wildcard_list.upper())
        dlg.SetWildcard(wildcard_filter)
        if dlg.ShowModal() == wx.ID_OK:
            filename = dlg.GetPath()
            if not filename.endswith('.ply'):
                if sys.is_linux():  # hack for linux, as for some reason the .ply is not appended.
                    filename += '.ply'
            mesh_loader.save_mesh(filename, self.workbench['scanning'].scene_view._object)
            self.append_last_file(filename)
        dlg.Destroy()

    def on_clear_model(self, event):
        if self.workbench['scanning'].scene_view._object is not None:
            dlg = wx.MessageDialog(
                self,
                _("Your current model will be deleted.\nAre you sure you want to delete it?"),
                _("Clear point cloud"), wx.YES_NO | wx.ICON_QUESTION)
            result = dlg.ShowModal() == wx.ID_YES
            dlg.Destroy()
            if result:
                self.workbench['scanning'].scene_view._clear_scene()

    def on_open_profile(self, category):
        dlg = wx.FileDialog(self, _("Select profile file to load"), profile.get_base_path(),
                            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
        dlg.SetWildcard("JSON files (*.json)|*.json")
        if dlg.ShowModal() == wx.ID_OK:
            profile_file = dlg.GetPath()
            profile.settings.load_settings(profile_file, categories=[category])
            self.update_profile_to_all_controls()
        dlg.Destroy()

    def on_save_profile(self, category):
        dlg = wx.FileDialog(self, _("Select profile file to save"), profile.get_base_path(),
                            category.replace('_settings', ''), style=wx.FD_SAVE)
        dlg.SetWildcard("JSON files (*.json)|*.json")
        if dlg.ShowModal() == wx.ID_OK:
            profile_file = dlg.GetPath()
            if not profile_file.endswith('.json'):
                if sys.is_linux():  # hack for linux, as for some reason the .json is not appended.
                    profile_file += '.json'
            profile.settings.save_settings(profile_file, categories=[category])
        dlg.Destroy()

    def on_reset_profile(self, category):
        dlg = wx.MessageDialog(
            self,
            _("This will reset all profile settings to defaults. "
              "Unless you have saved your current profile, all settings will be lost!\n"
              "Do you really want to reset?"),
            _("Profile reset"), wx.YES_NO | wx.ICON_QUESTION)
        result = dlg.ShowModal() == wx.ID_YES
        dlg.Destroy()
        if result:
            profile.settings.reset_to_default(categories=[category])
            self.update_profile_to_all_controls()

    def on_clear_log(self, event):
        dlg = wx.MessageDialog(
            self,
            _("Your current log file will be deleted.\n"
              "Are you sure you want to delete it?"),
            _("Clear log file"), wx.YES_NO | wx.ICON_QUESTION)
        result = dlg.ShowModal() == wx.ID_YES
        dlg.Destroy()
        if result:
            with open('horus.log', 'w'):
                pass
            date_format = '%Y-%m-%d %H:%M:%S'
            current_log_date = datetime.datetime.now()
            profile.settings['last_clear_log_date'] = str(current_log_date.strftime(date_format))

    def on_export_log(self, event):
        dlg = wx.FileDialog(self, _("Select log file to save"),
                            profile.get_base_path(), style=wx.FD_SAVE)
        dlg.SetWildcard("Log files (*.log)|*.log")
        if dlg.ShowModal() == wx.ID_OK:
            log_file = dlg.GetPath()
            if not log_file.endswith('.log'):
                if sys.is_linux():  # hack for linux, as for some reason the .log is not appended.
                    log_file += '.log'

            with open(log_file, 'w') as _file:
                with open('horus.log', 'r') as _log:
                    _file.write(_log.read())
            log_file
        dlg.Destroy()

    def on_exit(self, event):
        self.Close(True)

    def on_close(self, event):
        try:
            if ciclop_scan.is_scanning:
                ciclop_scan.stop()
                time.sleep(0.5)
            driver.board.set_unplug_callback(None)
            driver.camera.set_unplug_callback(None)
            driver.disconnect()
            for workbench in self.workbench:
                workbench.on_disconnect()
        except:
            pass
        self.Show(False)
        self.Destroy()

    def enable_gui(self, status):
        if status:
            self.toolbar.toolbar.Enable()
            self.toolbar.combo.Enable()
            for i in xrange(self.menu_bar.GetMenuCount()):
                self.menu_bar.EnableTop(i, True)
        else:
            self.toolbar.toolbar.Disable()
            self.toolbar.combo.Disable()
            for i in xrange(self.menu_bar.GetMenuCount()):
                self.menu_bar.EnableTop(i, False)

    def append_last_file(self, last_file):
        if last_file in self.last_files:
            self.last_files.remove(last_file)
        self.last_files.append(last_file)
        if len(self.last_files) > 4:
            self.last_files = self.last_files[1:5]
        profile.settings['last_file'] = last_file
        profile.settings['last_files'] = self.last_files

    def on_preferences(self, event):
        self.launch_preferences()

    def launch_preferences(self, basic=False):
        preferences = PreferencesDialog(basic=basic)
        preferences.ShowModal()

    """def on_machine_settings(self, event):
        machine_settings = MachineSettingsDialog(self)
        ret = machine_settings.ShowModal()

        if ret == wx.ID_OK:
            try:  # TODO: Fix this. If not in the Scanning workbench, _draw_machine() fails.
                self.workbench['scanning'].scene_view._draw_machine()
            except:
                pass
            profile.settings.save_settings(categories=["machine_settings"])
            self.workbench['scanning'].controls.panels["point_cloud_roi"].update_profile()"""

    def on_menu_view_clicked(self, key, checked, panel):
        profile.settings[key] = checked
        if checked:
            panel.Show()
        else:
            panel.Hide()
        panel.GetParent().Layout()
        panel.Layout()
        self.Layout()

    def on_scanning_panel_clicked(self, event):
        self.on_menu_view_clicked('view_scanning_panel',
                                  self.menu_scanning_panel.IsChecked(),
                                  self.workbench['scanning'].scroll_panel)

    def on_scanning_video_scene_clicked(self, event):
        checked_video = self.menu_scanning_video.IsChecked()
        checked_scene = self.menu_scanning_scene.IsChecked()
        profile.settings['view_scanning_video'] = checked_video
        profile.settings['view_scanning_scene'] = checked_scene
        self.workbench['scanning'].pages_collection['view_page'].Unsplit()
        if checked_video:
            self.workbench['scanning'].video_view.Show()
            self.workbench['scanning'].pages_collection['view_page'].SplitVertically(
                self.workbench['scanning'].video_view, self.workbench['scanning'].scene_panel)
            if checked_scene:
                self.workbench['scanning'].scene_panel.Show()
            else:
                self.workbench['scanning'].scene_panel.Hide()
                self.workbench['scanning'].pages_collection['view_page'].Unsplit()
        else:
            self.workbench['scanning'].video_view.Hide()
            if checked_scene:
                self.workbench['scanning'].scene_panel.Show()
                self.workbench['scanning'].pages_collection['view_page'].SplitVertically(
                    self.workbench['scanning'].scene_panel, self.workbench['scanning'].video_view)
                self.workbench['scanning'].pages_collection['view_page'].Unsplit()
            else:
                self.workbench['scanning'].scene_panel.Hide()
                self.workbench['scanning'].pages_collection['view_page'].Unsplit()

        self.workbench['scanning'].pages_collection['view_page'].Layout()
        self.workbench['scanning'].Layout()
        self.Layout()

    def on_mode_advanced_clicked(self, event):
        checked = self.menu_mode_advanced.IsChecked()
        profile.settings['view_mode_advanced'] = checked
        if checked:
            self.workbench['calibration'].panels_collection.expandable_panels[
                'video_settings'].Show()
            self.workbench['calibration'].panels_collection.expandable_panels[
                'camera_intrinsics'].Show()
        else:
            self.workbench['calibration'].panels_collection.expandable_panels[
                'video_settings'].Hide()
            self.workbench['calibration'].panels_collection.expandable_panels[
                'camera_intrinsics'].Hide()

            if profile.settings['current_panel_calibration'] == 'video_settings' or \
               profile.settings['current_panel_calibration'] == 'camera_intrinsics':
                self.workbench['calibration'].on_pattern_settings_selected()
                self.workbench['calibration'].panels_collection.expandable_panels[
                    profile.settings['current_panel_calibration']].on_title_clicked(None)
        self.workbench['calibration'].Layout()
        self.Layout()

    def on_connect(self):
        for workbench in self.workbench.values():
            workbench.enable_content()
        self.workbench[profile.settings['workbench']].on_connect()

    def on_disconnect(self):
        for workbench in self.workbench.values():
            workbench.on_disconnect()

    def on_combo_box_selected(self, event):
        self.update_workbench(event.GetEventObject().GetValue())

    def update_workbench(self, name):
        self.wait_cursor = wx.BusyCursor()
        self.toolbar.combo.SetValue(name)
        if sys.is_windows():
            for key, wb in self.workbench.iteritems():
                if wb.name == name:
                    wb.Show()
                    profile.settings['workbench'] = key
            for key, wb in self.workbench.iteritems():
                if wb.name != name:
                    wb.Hide()
        else:
            for key, wb in self.workbench.iteritems():
                if wb.name != name:
                    wb.Hide()
            for key, wb in self.workbench.iteritems():
                if wb.name == name:
                    wb.Show()
                    profile.settings['workbench'] = key
        is_scan = profile.settings['workbench'] == 'scanning'
        self.menu_file.Enable(self.menu_load_model.GetId(), is_scan)
        self.menu_file.Enable(self.menu_save_model.GetId(), is_scan)
        self.menu_file.Enable(self.menu_clear_model.GetId(), is_scan)
        self.toolbar.scanning_mode(is_scan)
        profile.settings.save_settings()
        self.Layout()

        del self.wait_cursor
        gc.collect()

    def on_about(self, event):
        info = wx.AboutDialogInfo()
        icon = wx.Icon(resources.get_path_for_image("horus.ico"), wx.BITMAP_TYPE_ICO)
        info.SetIcon(icon)
        info.SetName(u'Horus')
        info.SetVersion(__version__)
        tech_description = _('Horus is an Open Source 3D Scanner manager')
        tech_description += '\nVersion: ' + __version__
        tech_description += '\nDatetime: ' + __datetime__
        tech_description += '\nCommit: ' + __commit__
        info.SetDescription(tech_description)
        info.SetCopyright(u'(C) 2014-2016 Mundo Reader S.L.')
        info.SetWebSite(u'http://www.bq.com')
        info.SetLicence("Horus is free software; you can redistribute it and/or modify it\n"
                        "under the terms of the GNU General Public License as published by\n"
                        "the Free Software Foundation; either version 2 of the License,\n"
                        "or (at your option) any later version.\n"
                        "Horus is distributed in the hope that it will be useful,\n"
                        "but WITHOUT ANY WARRANTY; without even the implied warranty of\n"
                        "MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.\n"
                        "See the GNU General Public License for more details. You should have\n"
                        "received a copy of the GNU General Public License along with\n"
                        "File Hunter; if not, write to the Free Software Foundation,\n"
                        "Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA")
        info.AddDeveloper(u'Jesús Arroyo, Irene Sanz, Jorge Robles')
        info.AddDocWriter(u'Jesús Arroyo, Ángel Larrañaga')
        info.AddArtist(u'Nestor Toribio')
        info.AddTranslator(u'Jesús Arroyo, Irene Sanz, Alexandre Galode, Natasha da Silva, '
                           'Camille Montgolfier, Markus Hoedl, Andrea Fantini, Maria Albuquerque, '
                           'Meike Schirmeister')
        wx.AboutBox(info)

    def on_welcome(self, event):
        WelcomeDialog(self)

    def on_updates(self, event):
        if profile.settings['check_for_updates']:
            if version.check_for_updates():
                VersionWindow(self)
            else:
                dlg = wx.MessageDialog(self, _("You are running the latest version of Horus!"), _(
                    "Updated!"), wx.OK | wx.ICON_INFORMATION)
                dlg.ShowModal()
                dlg.Destroy()

    def on_board_unplugged(self):
        self._on_device_unplugged(
            _("Scanner unplugged"),
            _("Scanner has been unplugged. Please, plug it in and press connect"))
        """self._on_device_unplugged(
            _("Board unplugged"),
            _("Board has been unplugged. Please, plug it in and press connect"))"""

    def on_camera_unplugged(self):
        self._on_device_unplugged(
            _("Scanner unplugged"),
            _("Scanner has been unplugged. Please, plug it in and press connect"))
        """self._on_device_unplugged(
            _("Camera unplugged"),
            _("Camera has been unplugged. Please, plug it in and press connect"))"""

    def _on_device_unplugged(self, title="", description=""):
        ciclop_scan.stop()
        scanner_autocheck.cancel()
        laser_triangulation.cancel()
        platform_extrinsics.cancel()
        self.toolbar.update_status(False)
        driver.disconnect()
        dlg = wx.MessageDialog(self, description, title, wx.OK | wx.ICON_ERROR)
        dlg.ShowModal()
        dlg.Destroy()

    def update_profile_to_all_controls(self):
        for _, w in self.workbench.iteritems():
            w.update_controls()
        self.workbench[profile.settings['workbench']].update_controls()

        # Scanning workbench layout
        if profile.settings['view_scanning_panel']:
            self.workbench['scanning'].scroll_panel.Show()
            self.menu_scanning_panel.Check(True)
        else:
            self.workbench['scanning'].scroll_panel.Hide()
            self.menu_scanning_panel.Check(False)

        checked_video = profile.settings['view_scanning_video']
        checked_scene = profile.settings['view_scanning_scene']

        self.menu_scanning_video.Check(checked_video)
        self.menu_scanning_scene.Check(checked_scene)

        checked = profile.settings['view_mode_advanced']

        self.menu_mode_advanced.Check(checked)

        self.workbench['scanning'].pages_collection['view_page'].Unsplit()
        if checked_video:
            self.workbench['scanning'].video_view.Show()
            self.workbench['scanning'].pages_collection['view_page'].SplitVertically(
                self.workbench['scanning'].video_view, self.workbench['scanning'].scene_panel)
            if checked_scene:
                self.workbench['scanning'].scene_panel.Show()
            else:
                self.workbench['scanning'].scene_panel.Hide()
                self.workbench['scanning'].pages_collection['view_page'].Unsplit()
        else:
            self.workbench['scanning'].video_view.Hide()
            if checked_scene:
                self.workbench['scanning'].scene_panel.Show()
                self.workbench['scanning'].pages_collection['view_page'].SplitVertically(
                    self.workbench['scanning'].scene_panel, self.workbench['scanning'].video_view)
                self.workbench['scanning'].pages_collection['view_page'].Unsplit()
            else:
                self.workbench['scanning'].scene_panel.Hide()
                self.workbench['scanning'].pages_collection['view_page'].Unsplit()

    def initialize_driver(self):
        # Serial name
        serial_list = driver.board.get_serial_list()
        current_serial = profile.settings['serial_name']
        if len(serial_list) > 0:
            if current_serial not in serial_list:
                profile.settings['serial_name'] = serial_list[0]
        # Video id
        video_list = driver.camera.get_video_list()
        current_video_id = profile.settings['camera_id']
        if len(video_list) > 0:
            if current_video_id not in video_list:
                profile.settings['camera_id'] = unicode(video_list[0])

        if len(profile.settings['camera_id']):
            driver.camera.camera_id = int(profile.settings['camera_id'][-1:])

        driver.board.serial_name = profile.settings['serial_name']
        driver.board.baud_rate = profile.settings['baud_rate']
        driver.board.motor_invert(profile.settings['invert_motor'])
        platform_extrinsics.set_estimated_size(profile.settings['estimated_size'])

        flush_setting = 'flush_'
        flush_stream_setting = 'flush_stream_'
        if sys.is_linux():
            flush_setting += 'linux'
            flush_stream_setting += 'linux'
        elif sys.is_darwin():
            flush_setting += 'darwin'
            flush_stream_setting += 'darwin'
        elif sys.is_windows():
            flush_setting += 'windows'
            flush_stream_setting += 'windows'

        texture, laser, pattern = profile.settings[flush_setting]
        image_capture.set_flush_values(texture, laser, pattern)
        texture, laser, pattern = profile.settings[flush_stream_setting]
        image_capture.set_flush_stream_values(texture, laser, pattern)
