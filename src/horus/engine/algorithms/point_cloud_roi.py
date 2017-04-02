# -*- coding: utf-8 -*-
# This file is part of the Horus Project

__author__ = 'Jesús Arroyo Torrens <jesus.arroyo@bq.com>'
__copyright__ = 'Copyright (C) 2014-2016 Mundo Reader S.L.'
__license__ = 'GNU General Public License v2 http://www.gnu.org/licenses/gpl2.html'

import cv2
import numpy as np

from horus import Singleton
from horus.engine.calibration.calibration_data import CalibrationData


@Singleton
class PointCloudROI(object):

    def __init__(self):
        self.calibration_data = CalibrationData()
        self._use_roi = False
        self._show_center = True
        self._height = 0
        self._radious = 0
        self._initialize()

    def _initialize(self):
        self._umin = 0
        self._umax = 0
        self._vmin = 0
        self._vmax = 0
        self._lower_vmin = 0
        self._lower_vmax = 0
        self._upper_vmin = 0
        self._upper_vmax = 0
        self._no_trimmed_umin = 0
        self._no_trimmed_umax = 0
        self._no_trimmed_vmin = 0
        self._no_trimmed_vmax = 0
        self._center_u = 0
        self._center_v = 0
        self._circle_resolution = 30
        self._circle_array = np.array([[np.cos(i * 2 * np.pi / self._circle_resolution)
                                        for i in xrange(self._circle_resolution)],
                                       [np.sin(i * 2 * np.pi / self._circle_resolution)
                                        for i in xrange(self._circle_resolution)],
                                       np.zeros(self._circle_resolution)])

    def set_diameter(self, value):
        self._radious = value / 2.0
        self._compute_roi()

    def set_height(self, value):
        self._height = value
        self._compute_roi()

    def set_use_roi(self, value):
        self._use_roi = value

    def set_show_center(self, value):
        self._show_center = value

    def mask_image(self, image):
        if self._center_v != 0 and self._center_u != 0 and self._use_roi:
            if image is not None:
                mask = np.zeros(image.shape, np.uint8)
                mask[self._vmin:self._vmax, self._umin:self._umax] = image[
                    self._vmin:self._vmax, self._umin:self._umax]
                return mask
        else:
            return image

    def mask_point_cloud(self, point_cloud, texture):
        if point_cloud is not None and texture is not None and len(point_cloud) > 0:
            rho = np.sqrt(np.square(point_cloud[0, :]) + np.square(point_cloud[1, :]))
            z = point_cloud[2, :]

            if self._use_roi:
                idx = np.where((z >= 0) &
                               (z <= self._height) &
                               (rho >= -self._radious) &
                               (rho <= self._radious))[0]
            else:
                idx = np.where((z >= 0) &
                               (rho >= -125) &
                               (rho <= 125))[0]

            return point_cloud[:, idx], texture[:, idx]

    def draw_cross(self, image):
        if self._center_v != 0 and self._center_u != 0 and self._show_center:
            thickness = 2
            v_max, u_max, _ = image.shape
            cv2.line(image, (0, self._center_v), (u_max, self._center_v), (200, 0, 0), thickness)
            cv2.line(image, (self._center_u, 0), (self._center_u, v_max), (200, 0, 0), thickness)
        return image

    def draw_roi(self, image):
        if self._center_v != 0 and self._center_u != 0:
            thickness = 6
            thickness_hiden = 1
            cy = self.calibration_data.camera_matrix[1][2]

            center_up_u = self._no_trimmed_umin + \
                (self._no_trimmed_umax - self._no_trimmed_umin) / 2
            center_up_v = self._upper_vmin + (self._upper_vmax - self._upper_vmin) / 2
            center_down_u = self._no_trimmed_umin + \
                (self._no_trimmed_umax - self._no_trimmed_umin) / 2
            center_down_v = self._lower_vmax + (self._lower_vmin - self._lower_vmax) / 2
            axes_up = ((self._no_trimmed_umax - self._no_trimmed_umin) / 2,
                       ((self._upper_vmax - self._upper_vmin) / 2))
            axes_down = ((self._no_trimmed_umax - self._no_trimmed_umin) / 2,
                         ((self._lower_vmin - self._lower_vmax) / 2))

            # upper ellipse
            if (center_up_v < cy):
                cv2.ellipse(image, (center_up_u, center_up_v), axes_up,
                            0, 180, 360, (0, 100, 200), thickness)
                cv2.ellipse(image, (center_up_u, center_up_v), axes_up,
                            0, 0, 180, (0, 100, 200), thickness_hiden)
            else:
                cv2.ellipse(image, (center_up_u, center_up_v), axes_up,
                            0, 180, 360, (0, 100, 200), thickness)
                cv2.ellipse(image, (center_up_u, center_up_v), axes_up,
                            0, 0, 180, (0, 100, 200), thickness)

            # lower ellipse
            cv2.ellipse(image, (center_down_u, center_down_v), axes_down,
                        0, 180, 360, (0, 100, 200), thickness_hiden)
            cv2.ellipse(image, (center_down_u, center_down_v),
                        axes_down, 0, 0, 180, (0, 100, 200), thickness)

            # cylinder lines
            cv2.line(image, (self._no_trimmed_umin, center_up_v),
                     (self._no_trimmed_umin, center_down_v), (0, 100, 200), thickness)
            cv2.line(image, (self._no_trimmed_umax, center_up_v),
                     (self._no_trimmed_umax, center_down_v), (0, 100, 200), thickness)

            # view center
            if axes_up[0] <= 0 or axes_up[1] <= 0:
                axes_up_center = (20, 1)
                axes_down_center = (20, 1)
            else:
                axes_up_center = (20, axes_up[1] * 20 / axes_up[0])
                axes_down_center = (20, axes_down[1] * 20 / axes_down[0])

            # upper center
            cv2.ellipse(image, (self._center_u, min(center_up_v, self._center_v)),
                        axes_up_center, 0, 0, 360, (0, 70, 120), -1)
            # lower center
            cv2.ellipse(image, (self._center_u, self._center_v),
                        axes_down_center, 0, 0, 360, (0, 70, 120), -1)
        return image

    def _compute_roi(self):
        if self.calibration_data.check_calibration() is False:
            self._initialize()
        else:
            # Load calibration values
            fx = self.calibration_data.camera_matrix[0][0]
            fy = self.calibration_data.camera_matrix[1][1]
            cx = self.calibration_data.camera_matrix[0][2]
            cy = self.calibration_data.camera_matrix[1][2]
            R = np.matrix(self.calibration_data.platform_rotation)
            t = np.matrix(self.calibration_data.platform_translation).T

            bottom = np.matrix(self._radious * self._circle_array)
            top = bottom + np.matrix([0, 0, self._height]).T
            data = np.concatenate((bottom, top), axis=1)

            # Compute center
            center = R * np.matrix(0 * self._circle_array) + t
            u = fx * center[0] / center[2] + cx
            v = fy * center[1] / center[2] + cy

            _umin = int(round(np.min(u)))
            _umax = int(round(np.max(u)))
            _vmin = int(round(np.min(v)))
            _vmax = int(round(np.max(v)))

            self._center_u = _umin + (_umax - _umin) / 2
            self._center_v = _vmin + (_vmax - _vmin) / 2

            # Compute cylinders
            data = R * data + t
            u = fx * data[0] / data[2] + cx
            v = fy * data[1] / data[2] + cy

            _umin = int(round(np.min(u)))
            _umax = int(round(np.max(u)))
            _vmin = int(round(np.min(v)))
            _vmax = int(round(np.max(v)))

            # Visualization
            v_ = np.array(v.T)

            # Lower cylinder base
            a = v_[:(len(v_) / 2)]
            # Upper cylinder base
            b = v_[(len(v_) / 2):]

            self._lower_vmin = int(round(np.max(a)))
            self._lower_vmax = int(round(np.min(a)))
            self._upper_vmin = int(round(np.min(b)))
            self._upper_vmax = int(round(np.max(b)))

            self._no_trimmed_umin = _umin
            self._no_trimmed_umax = int(round(np.max(u)))
            self._no_trimmed_vmin = int(round(np.min(v)))
            self._no_trimmed_vmax = int(round(np.max(v)))

            self._umin = max(_umin, 0)
            self._umax = min(_umax, self.calibration_data.width)
            self._vmin = max(_vmin, 0)
            self._vmax = min(_vmax, self.calibration_data.height)
