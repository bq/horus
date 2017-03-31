# -*- coding: utf-8 -*-
# This file is part of the Horus Project

__author__ = 'Jesús Arroyo Torrens <jesus.arroyo@bq.com>'
__copyright__ = 'Copyright (C) 2014-2016 Mundo Reader S.L.'
__license__ = 'GNU General Public License v2 http://www.gnu.org/licenses/gpl2.html'


import numpy as np
import cv2

from horus import Singleton
from horus.engine.calibration.calibration_data import CalibrationData


@Singleton
class PointCloudGeneration(object):

    def __init__(self):
        self.calibration_data = CalibrationData()

    def compute_point_cloud(self, theta, points_2d, index):
        if points_2d[0].size == 0 or points_2d[1].size == 0:
            return None
        
        # Load calibration values
        R = np.matrix(self.calibration_data.platform_rotation)
        t = np.matrix(self.calibration_data.platform_translation).T
        # Compute platform transformation
        Xwo = self.compute_platform_point_cloud(points_2d, R, t, index)
        # Rotate to world coordinates
        c, s = np.cos(-theta), np.sin(-theta)
        Rz = np.matrix([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        Xw = Rz * Xwo
        # Return point cloud
        return np.array(Xw)

    def compute_platform_point_cloud(self, points_2d, R, t, index):
        # Load calibration values
        n = self.calibration_data.laser_planes[index].normal
        d = self.calibration_data.laser_planes[index].distance
        # Camera system
        Xc = self.compute_camera_point_cloud(points_2d, d, n)
        # Transformation to platform system
        return R.T * Xc - R.T * t

    def compute_camera_point_cloud(self, points_2d, d, n):
        if points_2d[0].size == 0 or points_2d[1].size == 0:
            return np.array([]).reshape(3, 0)

        # Compute projection point
        u, v = points_2d
        points_for_undistort = np.array([np.concatenate((u, v)).reshape(2, len(u)).T])
        
        print points_for_undistort.shape
        # use opencv's undistortPoints, which incorporates the distortion coefficients
        points_undistorted = cv2.undistortPoints(points_for_undistort, self.calibration_data.camera_matrix, self.calibration_data.distortion_vector)

        u, v = np.hsplit(points_undistorted[0], points_undistorted[0].shape[1])
        
        # make homogenous coordinates
        x = np.concatenate((u.T[0], v.T[0], np.ones(len(u)))).reshape(3, len(u))
        # normalize to get unit direction vectors
        cam_point_direction = x / np.linalg.norm(x, axis=0)
        
        # Compute laser intersection:
        # dlc = dot(laser_normal, cam_point_direction) = projection of camera ray on laser-plane normal
        # d / dlc = distance from cam center to 3D point
        # cam_point_direction * d / dlc = 3D point
        return d / np.dot(n, cam_point_direction) * cam_point_direction
