# -*- coding: utf-8 -*-
# This file is part of the Horus Project

__author__ = 'Jesús Arroyo Torrens <jesus.arroyo@bq.com>'
__copyright__ = 'Copyright (C) 2014-2016 Mundo Reader S.L.'
__license__ = 'GNU General Public License v2 http://www.gnu.org/licenses/gpl2.html'

import numpy as np

from horus import Singleton
from horus.engine.calibration.calibration import Calibration, CalibrationCancel


class PatternNotDetected(Exception):

    def __init__(self):
        Exception.__init__(self, "Pattern Not Detected")


class WrongMotorDirection(Exception):

    def __init__(self):
        Exception.__init__(self, "Wrong Motor Direction")


class LaserNotDetected(Exception):

    def __init__(self):
        Exception.__init__(self, "Laser Not Detected")


class WrongLaserPosition(Exception):

    def __init__(self):
        Exception.__init__(self, "Wrong Laser Position")


@Singleton
class Autocheck(Calibration):

    """Auto check algorithm:
            - Check pattern detection
            - Check motor direction
            - Check lasers
    """

    def __init__(self):
        self.image = None
        self.current_angle = 0
        Calibration.__init__(self)

    def _start(self):
        if self.driver.is_connected:
            ret = False
            response = None
            self.image = None
            self.current_angle = 0
            self._is_calibrating = True
            self.image_capture.stream = False

            # Setup scanner
            self.driver.board.lasers_off()
            self.driver.board.motor_enable()
            self.driver.board.motor_reset_origin()
            self.driver.board.motor_speed(200)
            self.driver.board.motor_acceleration(200)

            # Perform autocheck
            try:
                self.check_pattern_and_motor()
                self.check_lasers()
                ret = True
            except Exception as exception:
                response = exception
                if isinstance(exception, CalibrationCancel):
                    self._cancel()
            finally:
                self._is_calibrating = False
                self.image_capture.stream = True
                self.driver.board.lasers_off()
                self.driver.board.motor_disable()
                if self._progress_callback is not None:
                    self._progress_callback(100)
                if self._after_callback is not None:
                    self._after_callback((ret, response))
                self.image = None

    def _cancel(self):
        # Return to origin
        if self.current_angle > 180:
            self.current_angle = self.current_angle - 360
        self.driver.board.motor_move(-self.current_angle)

    def check_pattern_and_motor(self):
        scan_step = 30
        patterns_detected = {}
        patterns_sorted = {}

        if self._progress_callback is not None:
            self._progress_callback(0)

        # Capture data
        for i in xrange(0, 360, scan_step):
            self.current_angle = i
            if not self._is_calibrating:
                raise CalibrationCancel()
            image = self.image_capture.capture_pattern()
            pose = self.image_detection.detect_pose(image)
            if pose is not None:
                self.image = self.image_detection.draw_pattern(image, pose[2])
                patterns_detected[i] = pose[0].T[2][0]
            else:
                self.image = self.image_detection.detect_pattern(image)
            if self._progress_callback is not None:
                self._progress_callback(i / 3.6)
            self.driver.board.motor_move(scan_step)

        # Check pattern detection
        if len(patterns_detected) == 0:
            raise PatternNotDetected()

        # Check motor direction
        max_x = max(patterns_detected.values())
        max_i = [key for key, value in patterns_detected.items() if value == max_x][0]
        min_v = max_x
        for i in xrange(max_i, max_i + 360, scan_step):
            if i % 360 in patterns_detected:
                v = patterns_detected[i % 360]
                patterns_sorted[i] = v
                if v <= min_v:
                    min_v = v
                else:
                    raise WrongMotorDirection()

        # Move to nearest position
        x = np.array(patterns_sorted.keys())
        y = np.array(patterns_sorted.values())
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y)[0]
        pos = -c / m % 360
        if pos > 180:
            pos = pos - 360
        self.driver.board.motor_move(pos)

    def check_lasers(self):
        image = self.image_capture.capture_pattern()
        corners = self.image_detection.detect_corners(image)
        self.image_capture.flush_laser()
        for i in xrange(2):
            if not self._is_calibrating:
                raise CalibrationCancel()
            image = self.image_capture.capture_laser(i)
            image = self.image_detection.pattern_mask(image, corners)
            lines = self.laser_segmentation.compute_hough_lines(image)
            if lines is None:
                raise LaserNotDetected()
