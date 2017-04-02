# -*- coding: utf-8 -*-
# This file is part of the Horus Project

__author__ = 'Jesús Arroyo Torrens <jesus.arroyo@bq.com>'
__copyright__ = 'Copyright (C) 2014-2016 Mundo Reader S.L.'
__license__ = 'GNU General Public License v2 http://www.gnu.org/licenses/gpl2.html'

import wx._core

from horus.util import profile, resources, system

from horus.gui.engine import driver, scanner_autocheck, image_capture, image_detection
from horus.gui.util.image_view import ImageView
from horus.gui.wizard.wizard_page import WizardPage

from horus.engine.driver.board import WrongFirmware, BoardNotConnected, OldFirmware
from horus.engine.driver.camera import WrongCamera, CameraNotConnected, InvalidVideo, \
    WrongDriver
from horus.engine.calibration.autocheck import PatternNotDetected, \
    WrongMotorDirection, LaserNotDetected


class ConnectionPage(WizardPage):

    def __init__(self, parent, button_prev_callback=None, button_next_callback=None):
        WizardPage.__init__(self, parent,
                            title=_("Connection"),
                            button_prev_callback=button_prev_callback,
                            button_next_callback=button_next_callback)

        self.parent = parent

        self.connect_button = wx.Button(self.panel, label=_("Connect"))
        self.preferences_button = wx.Button(self.panel, label=_("Preferences"))

        self.pattern_label = wx.StaticText(self.panel, label=_(
            "Put the pattern on the platform as shown in the picture and press \"Auto check\""))
        self.pattern_label.Wrap(400)
        self.image_view = ImageView(self.panel, quality=wx.IMAGE_QUALITY_HIGH)
        self.image_view.set_image(wx.Image(resources.get_path_for_image("pattern-position.png")))
        self.auto_check_button = wx.Button(self.panel, label=_("Auto check"))
        self.gauge = wx.Gauge(self.panel, range=100, size=(-1, 30))
        self.result_label = wx.StaticText(self.panel, size=(-1, 30))

        self.auto_check_button.Disable()
        self.skip_button.Disable()
        self.next_button.Disable()
        self.result_label.Hide()
        self.enable_next = False

        vbox = wx.BoxSizer(wx.VERTICAL)
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.Add(self.connect_button, 1, wx.ALL | wx.EXPAND, 5)
        hbox.Add(self.preferences_button, 1, wx.ALL | wx.EXPAND, 5)
        vbox.Add(hbox, 0, wx.ALL | wx.EXPAND, 2)
        vbox.Add(self.pattern_label, 0, wx.ALL | wx.CENTER, 5)
        vbox.Add(self.image_view, 1, wx.ALL | wx.EXPAND, 5)
        vbox.Add(self.result_label, 0, wx.ALL | wx.CENTER, 5)
        vbox.Add(self.gauge, 0, wx.ALL | wx.EXPAND, 5)
        vbox.Add(self.auto_check_button, 0, wx.ALL | wx.EXPAND, 5)
        self.panel.SetSizer(vbox)

        self.Layout()

        self.connect_button.Bind(wx.EVT_BUTTON, self.on_connect_button_clicked)
        self.preferences_button.Bind(wx.EVT_BUTTON, self.on_preferences_button_clicked)
        self.auto_check_button.Bind(wx.EVT_BUTTON, self.on_auto_check_button_clicked)
        self.Bind(wx.EVT_SHOW, self.on_show)

        self.video_view.set_callback(self.get_image)
        self.update_status(driver.is_connected)

    def on_show(self, event):
        if event.GetShow():
            driver.board.lasers_off()
            self.update_status(driver.is_connected)
        else:
            try:
                self.video_view.stop()
            except:
                pass

    def get_image(self):
        if scanner_autocheck.image is not None:
            image = scanner_autocheck.image
        else:
            image = image_capture.capture_pattern()
            image = image_detection.detect_pattern(image)
        return image

    def on_unplugged(self):
        self.video_view.stop()
        scanner_autocheck.cancel()
        driver.disconnect()
        self.parent.on_exit(message=False)

    def on_connect_button_clicked(self, event):
        if driver.is_connected:
            driver.disconnect()
            self.update_status(driver.is_connected)
        else:
            # If no camera id is selected
            video_list = driver.camera.get_video_list()
            current_video_id = profile.settings['camera_id']
            if len(video_list) > 0:
                if current_video_id not in video_list:
                    profile.settings['camera_id'] = unicode(video_list[0])
                    driver.camera.camera_id = int(profile.settings['camera_id'][-1:])

            driver.set_callbacks(
                lambda: wx.CallAfter(self.before_connect),
                lambda r: wx.CallAfter(self.after_connect, r))
            driver.connect()

    def on_preferences_button_clicked(self, event):
        self.GetParent().parent.launch_preferences(basic=True)

    def before_connect(self):
        self.Disable()
        self.video_view.stop()
        driver.board.set_unplug_callback(None)
        driver.camera.set_unplug_callback(None)
        self.wait_cursor = wx.BusyCursor()

    def after_connect(self, response):
        ret, result = response

        if not ret:
            if isinstance(result, WrongFirmware):
                dlg = wx.MessageDialog(
                    self,
                    _("The board has the wrong firmware or an invalid baud rate.\n"
                      "Please select your board and press \"Upload firmware\""),
                    _(result), wx.OK | wx.ICON_INFORMATION)
                dlg.ShowModal()
                dlg.Destroy()
                self.update_status(False)
                self.GetParent().parent.launch_preferences()
            elif isinstance(result, BoardNotConnected):
                dlg = wx.MessageDialog(
                    self,
                    _("The board is not connected.\n"
                      "Please connect your board and select a valid Serial name"),
                    _(result), wx.OK | wx.ICON_INFORMATION)
                dlg.ShowModal()
                dlg.Destroy()
                self.update_status(False)
                self.GetParent().parent.launch_preferences(basic=True)
            elif isinstance(result, OldFirmware):
                dlg = wx.MessageDialog(
                    self,
                    _("The board has and old firmware.\n"
                      "Please select your board and press \"Upload firmware\""),
                    _(result), wx.OK | wx.ICON_INFORMATION)
                dlg.ShowModal()
                dlg.Destroy()
                self.update_status(False)
                self.GetParent().parent.launch_preferences()
            elif isinstance(result, WrongCamera):
                dlg = wx.MessageDialog(
                    self,
                    _("You probably have selected the wrong camera.\n"
                      "Please select another Camera ID"),
                    _(result), wx.OK | wx.ICON_INFORMATION)
                dlg.ShowModal()
                dlg.Destroy()
                self.update_status(False)
                wrong_camera_id = profile.settings['camera_id']
                self.GetParent().parent.launch_preferences(basic=True)
                # Do not save camera id if it is wrong
                if profile.settings['camera_id'] == wrong_camera_id:
                    profile.settings['camera_id'] = ''
            elif isinstance(result, CameraNotConnected):
                dlg = wx.MessageDialog(
                    self, _("Please plug your camera in and try to connect again"),
                    _(result), wx.OK | wx.ICON_ERROR)
                dlg.ShowModal()
                dlg.Destroy()
            elif isinstance(result, InvalidVideo):
                dlg = wx.MessageDialog(
                    self, _("Unplug and plug your camera USB cable and try to connect again"),
                    _(result), wx.OK | wx.ICON_ERROR)
                dlg.ShowModal()
                dlg.Destroy()
            elif isinstance(result, WrongDriver):
                if system.is_windows():
                    dlg = wx.MessageDialog(
                        self, _("Please, download and install the camera driver: \n"
                                "http://support.logitech.com/en_us/product/hd-webcam-c270"),
                        _(result), wx.OK | wx.ICON_ERROR)
                    dlg.ShowModal()
                    dlg.Destroy()

        self.update_status(driver.is_connected)
        self.Enable()
        del self.wait_cursor

    def on_auto_check_button_clicked(self, event):
        if profile.settings['adjust_laser']:
            profile.settings['adjust_laser'] = False
            dlg = wx.MessageDialog(
                self,
                _("It is recommended to adjust the line lasers vertically.\n"
                  "You will need to use the allen key.\n"
                  "Do you want to adjust it now?"),
                _("Manual laser adjustment"), wx.YES_NO | wx.ICON_QUESTION)
            result = dlg.ShowModal() == wx.ID_YES
            dlg.Destroy()
            if result:
                driver.board.lasers_on()
        else:
            # Perform auto check
            scanner_autocheck.set_callbacks(lambda: wx.CallAfter(self.before_auto_check),
                                            lambda p: wx.CallAfter(self.progress_auto_check, p),
                                            lambda r: wx.CallAfter(self.after_auto_check, r))
            scanner_autocheck.start()

    def before_auto_check(self):
        self.Disable()
        self.enable_next = False
        self.gauge.SetValue(0)
        self.result_label.Hide()
        self.gauge.Show()
        self.wait_cursor = wx.BusyCursor()
        self.Layout()

    def progress_auto_check(self, progress):
        self.gauge.SetValue(progress)

    def after_auto_check(self, response):
        ret, result = response

        if ret:
            self.result_label.SetLabel(_("Success. Please press \"Next\" to continue"))
            dlg = wx.MessageDialog(
                self, _("Scanner configured correctly"),
                _("Success"), wx.OK | wx.ICON_INFORMATION)
            dlg.ShowModal()
            dlg.Destroy()
        else:
            self.result_label.SetLabel(str(_(result)))
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
                self.GetParent().parent.launch_preferences(basic=True)
            elif isinstance(result, LaserNotDetected):
                dlg = wx.MessageDialog(
                    self, _("Please, check the lasers connection. "
                            "Also you can set up the calibration's capture and "
                            "segmentation settings in the \"Adjustment workbench\" "
                            "until the lasers are detected correctly"),
                    _(result), wx.OK | wx.ICON_ERROR)
                dlg.ShowModal()
                dlg.Destroy()

        if ret:
            self.skip_button.Disable()
            self.next_button.Enable()
        else:
            self.skip_button.Enable()
            self.next_button.Disable()

        self.Enable()
        self.enable_next = True
        self.result_label.Show()
        self.gauge.Hide()
        if hasattr(self, 'wait_cursor'):
            del self.wait_cursor
        self.panel.Fit()
        self.panel.Layout()
        self.Layout()

    def update_status(self, status):
        if status:
            driver.board.set_unplug_callback(
                lambda: wx.CallAfter(self.parent.on_board_unplugged))
            driver.camera.set_unplug_callback(
                lambda: wx.CallAfter(self.parent.on_camera_unplugged))
            self.GetParent().parent.workbench['calibration'].setup_engine()
            self.video_view.play()
            self.connect_button.SetLabel(_("Disconnect"))
            self.skip_button.Enable()
            self.enable_next = True
            self.auto_check_button.Enable()
        else:
            self.video_view.stop()
            self.video_view.reset()
            self.gauge.SetValue(0)
            self.gauge.Show()
            self.result_label.Hide()
            self.result_label.SetLabel("")
            self.connect_button.SetLabel(_("Connect"))
            self.skip_button.Disable()
            self.next_button.Disable()
            self.enable_next = False
            self.auto_check_button.Disable()
        self.Layout()
