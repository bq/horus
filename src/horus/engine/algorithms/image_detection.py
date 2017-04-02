# -*- coding: utf-8 -*-
# This file is part of the Horus Project

__author__ = 'Jesús Arroyo Torrens <jesus.arroyo@bq.com>'
__copyright__ = 'Copyright (C) 2014-2016 Mundo Reader S.L.'
__license__ = 'GNU General Public License v2 http://www.gnu.org/licenses/gpl2.html'

import cv2
import numpy as np

from horus import Singleton
from horus.engine.calibration.pattern import Pattern
from horus.engine.calibration.calibration_data import CalibrationData


@Singleton
class ImageDetection(object):

    def __init__(self):
        self.pattern = Pattern()
        self.calibration_data = CalibrationData()
        self.chessboard_mask = None

        self._criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    def detect_pattern(self, image):
        corners = self._detect_chessboard(image)
        if corners is not None:
            image = self.draw_pattern(image, corners)
        return image

    def draw_pattern(self, image, corners):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.drawChessboardCorners(
            image, (self.pattern.columns, self.pattern.rows), corners, True)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def detect_corners(self, image):
        corners = self._detect_chessboard(image)
        return corners

    def detect_pose(self, image):
        corners = self._detect_chessboard(image)
        if corners is not None:
            ret, rvecs, tvecs = cv2.solvePnP(
                self.pattern.object_points, corners,
                self.calibration_data.camera_matrix, self.calibration_data.distortion_vector)
            if ret:
                return (cv2.Rodrigues(rvecs)[0], tvecs, corners)

    def detect_pattern_plane(self, pose):
        if pose is not None:
            R = pose[0]
            t = pose[1].T[0]
            c = pose[2]
            n = R.T[2]
            d = np.dot(n, t)
            return (d, n, c)

    def pattern_mask(self, image, corners):
        if image is not None:
            h, w, d = image.shape
            if corners is not None:
                corners = corners.astype(np.int)
                p1 = corners[0][0]
                p2 = corners[self.pattern.columns - 1][0]
                p3 = corners[self.pattern.columns * (self.pattern.rows - 1)][0]
                p4 = corners[self.pattern.columns * self.pattern.rows - 1][0]
                mask = np.zeros((h, w), np.uint8)
                points = np.array([p1, p2, p4, p3])
                cv2.fillConvexPoly(mask, points, 255)
                image = cv2.bitwise_and(image, image, mask=mask)
                if self.chessboard_mask is not None:
                    image = cv2.bitwise_and(image, image, mask=self.chessboard_mask)
        return image

    def _detect_chessboard(self, image):
        if image is not None:
            if self.pattern.rows > 2 and self.pattern.columns > 2:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                ret, corners = cv2.findChessboardCorners(
                    gray, (self.pattern.columns, self.pattern.rows), flags=cv2.CALIB_CB_FAST_CHECK)
                if ret:
                    self.chessboard_mask = cv2.threshold(
                        gray, gray.max() / 2, 255, cv2.THRESH_BINARY)[1]
                    cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self._criteria)
                    return corners
                else:
                    self.chessboard_mask = None
