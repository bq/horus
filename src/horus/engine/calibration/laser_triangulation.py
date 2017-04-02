# -*- coding: utf-8 -*-
# This file is part of the Horus Project

__author__ = 'Jesús Arroyo Torrens <jesus.arroyo@bq.com>'
__copyright__ = 'Copyright (C) 2014-2016 Mundo Reader S.L.'
__license__ = 'GNU General Public License v2 http://www.gnu.org/licenses/gpl2.html'

import struct
import math
import numpy as np

from horus import Singleton
from horus.engine.calibration.calibration import CalibrationCancel
from horus.engine.calibration.moving_calibration import MovingCalibration

import logging
logger = logging.getLogger(__name__)


class LaserTriangulationError(Exception):

    def __init__(self):
        Exception.__init__(self, "LaserTriangulationError")


@Singleton
class LaserTriangulation(MovingCalibration):

    """Laser triangulation algorithm:

            - Laser coordinates matrix
            - Pattern's origin
            - Pattern's normal
    """

    def __init__(self):
        self.image = None
        self.has_image = False
        MovingCalibration.__init__(self)

    def _initialize(self):
        self.image = None
        self.has_image = True
        self.image_capture.stream = False
        self._point_cloud = [None, None]

    def _capture(self, angle):
        image = self.image_capture.capture_pattern()
        pose = self.image_detection.detect_pose(image)
        plane = self.image_detection.detect_pattern_plane(pose)
        if plane is not None:
            distance, normal, corners = plane

            # Angle between the pattern and the camera
            alpha = np.rad2deg(math.acos(normal[2])) * math.copysign(1, normal[0])
            if abs(alpha) < 30:
                self.image_capture.flush_laser(14)
                for i in xrange(2):
                    if (i == 0 and alpha < 10) or (i == 1 and alpha > -10):
                        image = self.image_capture.capture_laser(i)
                        image = self.image_detection.pattern_mask(image, corners)
                        self.image = image
                        points_2d, image = self.laser_segmentation.compute_2d_points(image)
                        point_3d = self.point_cloud_generation.compute_camera_point_cloud(
                            points_2d, distance, normal)
                        if self._point_cloud[i] is None:
                            self._point_cloud[i] = point_3d.T
                        else:
                            self._point_cloud[i] = np.concatenate(
                                (self._point_cloud[i], point_3d.T))
            else:
                self.image = image
        else:
            self.image = image

    def _calibrate(self):
        self.has_image = False
        self.image_capture.stream = True

        # Save point clouds
        for i in xrange(2):
            save_point_cloud('PC' + str(i) + '.ply', self._point_cloud[i])

        self.distance = [None, None]
        self.normal = [None, None]
        self.std = [None, None]

        # Compute planes
        for i in xrange(2):
            if self._is_calibrating:
                plane = compute_plane(i, self._point_cloud[i])
                self.distance[i], self.normal[i], self.std[i] = plane

        if self._is_calibrating:
            if self.std[0] < 1.0 and self.std[1] < 1.0 and \
               self.normal[0] is not None and self.normal[1] is not None:
                response = (True, ((self.distance[0], self.normal[0], self.std[0]),
                                   (self.distance[1], self.normal[1], self.std[1])))
            else:
                response = (False, LaserTriangulationError())
        else:
            response = (False, CalibrationCancel())

        self._is_calibrating = False
        self.image = None

        return response

    def accept(self):
        for i in xrange(2):
            self.calibration_data.laser_planes[i].distance = self.distance[i]
            self.calibration_data.laser_planes[i].normal = self.normal[i]


def compute_plane(index, X):
    if X is not None and X.shape[0] > 3:
        model, inliers = ransac(X, PlaneDetection(), 3, 0.1)

        distance, normal, M = model
        std = np.dot(M.T, normal).std()

        logger.info("Laser calibration " + str(index))
        logger.info(" Distance: " + str(distance))
        logger.info(" Normal: " + str(normal))
        logger.info(" Standard deviation: " + str(std))
        logger.info(" Point cloud size: " + str(len(inliers)))

        return distance, normal, std
    else:
        return None, None, None

import numpy.linalg
# from scipy.sparse import linalg


class PlaneDetection(object):

    def fit(self, X):
        M, Xm = self._compute_m(X)
        # U = linalg.svds(M, k=2)[0]
        # normal = np.cross(U.T[0], U.T[1])
        normal = numpy.linalg.svd(M)[0][:, 2]
        if normal[2] < 0:
            normal *= -1
        dist = np.dot(normal, Xm)
        return dist, normal, M

    def residuals(self, model, X):
        _, normal, _ = model
        M, Xm = self._compute_m(X)
        return np.abs(np.dot(M.T, normal))

    def is_degenerate(self, sample):
        return False

    def _compute_m(self, X):
        n = X.shape[0]
        Xm = X.sum(axis=0) / n
        M = np.array(X - Xm).T
        return M, Xm


def ransac(data, model_class, min_samples, threshold, max_trials=500):
    best_model = None
    best_inlier_num = 0
    best_inliers = None
    data_idx = np.arange(data.shape[0])
    for _ in xrange(max_trials):
        sample = data[np.random.randint(0, data.shape[0], 3)]
        if model_class.is_degenerate(sample):
            continue
        sample_model = model_class.fit(sample)
        sample_model_residua = model_class.residuals(sample_model, data)
        sample_model_inliers = data_idx[sample_model_residua < threshold]
        inlier_num = sample_model_inliers.shape[0]
        if inlier_num > best_inlier_num:
            best_inlier_num = inlier_num
            best_inliers = sample_model_inliers
    if best_inliers is not None:
        best_model = model_class.fit(data[best_inliers])
    return best_model, best_inliers


def save_point_cloud(filename, point_cloud):
    if point_cloud is not None:
        f = open(filename, 'wb')
        save_point_cloud_stream(f, point_cloud)
        f.close()


def save_point_cloud_stream(stream, point_cloud):
    frame = "ply\n"
    frame += "format binary_little_endian 1.0\n"
    frame += "comment Generated by Horus software\n"
    frame += "element vertex {0}\n".format(len(point_cloud))
    frame += "property float x\n"
    frame += "property float y\n"
    frame += "property float z\n"
    frame += "property uchar red\n"
    frame += "property uchar green\n"
    frame += "property uchar blue\n"
    frame += "element face 0\n"
    frame += "property list uchar int vertex_indices\n"
    frame += "end_header\n"
    for point in point_cloud:
        frame += struct.pack("<fffBBB", point[0], point[1], point[2], 255, 0, 0)
    stream.write(frame)
