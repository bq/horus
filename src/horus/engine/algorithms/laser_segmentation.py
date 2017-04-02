# -*- coding: utf-8 -*-
# This file is part of the Horus Project

__author__ = 'Jesús Arroyo Torrens <jesus.arroyo@bq.com>'
__copyright__ = 'Copyright (C) 2014-2016 Mundo Reader S.L.'
__license__ = 'GNU General Public License v2 http://www.gnu.org/licenses/gpl2.html'

import cv2
import math
import numpy as np
import scipy.ndimage

from horus import Singleton
from horus.engine.calibration.calibration_data import CalibrationData
from horus.engine.algorithms.point_cloud_roi import PointCloudROI


@Singleton
class LaserSegmentation(object):

    def __init__(self):
        self.calibration_data = CalibrationData()
        self.point_cloud_roi = PointCloudROI()

        self.red_channel = 'R (RGB)'
        self.threshold_enable = False
        self.threshold_value = 0
        self.blur_enable = False
        self.blur_value = 0
        self.window_enable = False
        self.window_value = 0
        self.refinement_method = 'SGF'

    def set_red_channel(self, value):
        self.red_channel = value

    def set_threshold_enable(self, value):
        self.threshold_enable = value

    def set_threshold_value(self, value):
        self.threshold_value = value

    def set_blur_enable(self, value):
        self.blur_enable = value

    def set_blur_value(self, value):
        self.blur_value = 2 * value + 1

    def set_window_enable(self, value):
        self.window_enable = value

    def set_window_value(self, value):
        self.window_value = value

    def set_refinement_method(self, value):
        self.refinement_method = value

    def compute_2d_points(self, image):
        if image is not None:
            image = self.compute_line_segmentation(image)
            # Peak detection: center of mass
            s = image.sum(axis=1)
            v = np.where(s > 0)[0]
            u = (self.calibration_data.weight_matrix * image).sum(axis=1)[v] / s[v]
            if self.refinement_method == 'SGF':
                # Segmented gaussian filter
                u = self._sgf(u, s)
            elif self.refinement_method == 'RANSAC':
                # Random sample consensus
                u = self._ransac(u, v)
            # Saturate u
            u = np.clip(u, 0, self.calibration_data.width - 1)
            return (u, v), image

    def compute_hough_lines(self, image):
        if image is not None:
            image = self.compute_line_segmentation(image)
            lines = cv2.HoughLines(image, 1, np.pi / 180, 120)
            # if lines is not None:
            #   rho, theta = lines[0][0]
            #   ## Calculate coordinates
            #   u1 = rho / np.cos(theta)
            #   u2 = u1 - height * np.tan(theta)
            return lines

    def compute_line_segmentation(self, image):
        if image is not None:
            # Apply ROI mask
            image = self.point_cloud_roi.mask_image(image)
            image = self._obtain_red_channel(image)
            image = self._threshold_image(image)
            image = self._window_mask(image)
            return image

    def _obtain_red_channel(self, image):
        ret = None
        if self.red_channel == 'R (RGB)':
            ret = cv2.split(image)[0]
            # ret = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif self.red_channel == 'Cr (YCrCb)':
            ret = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2YCR_CB))[1]
        elif self.red_channel == 'U (YUV)':
            ret = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2YUV))[1]
        return ret

    def _threshold_image(self, image):
        if self.threshold_enable:
            if image is not None:
                image = cv2.threshold(
                    image, self.threshold_value, 255, cv2.THRESH_TOZERO)[1]
                if self.blur_enable:
                    image = cv2.blur(image, (self.blur_value, self.blur_value))
                image = cv2.threshold(
                    image, self.threshold_value, 255, cv2.THRESH_TOZERO)[1]
        return image

    def _window_mask(self, image):
        if self.window_enable:
            if image is not None:
                peak = image.argmax(axis=1)
                _min = peak - self.window_value
                _max = peak + self.window_value + 1
                mask = np.zeros_like(image)
                for i in xrange(self.calibration_data.height):
                    mask[i, _min[i]:_max[i]] = 255
                # Apply mask
                image = cv2.bitwise_and(image, mask)
        return image

    # Segmented gaussian filter

    def _sgf(self, u, s):
        if len(u) > 1:
            i = 0
            sigma = 2.0
            f = np.array([])
            segments = [s[_r] for _r in np.ma.clump_unmasked(np.ma.masked_equal(s, 0))]
            # Detect stripe segments
            for segment in segments:
                j = len(segment)
                # Apply gaussian filter
                fseg = scipy.ndimage.gaussian_filter(u[i:i + j], sigma=sigma)
                f = np.concatenate((f, fseg))
                i += j
            return f
        else:
            return u

    # RANSAC implementation: https://github.com/ahojnnes/numpy-snippets/blob/master/ransac.py

    def _ransac(self, u, v):
        if len(u) > 1:
            data = np.vstack((v.ravel(), u.ravel())).T
            dr, thetar = self.ransac(data, self.LinearLeastSquares2D(), 2, 1)
            # v = np.array(range(min(v), max(v)))
            u = (dr - v * math.sin(thetar)) / math.cos(thetar)
        return u

    class LinearLeastSquares2D(object):
        '''
        2D linear least squares using the hesse normal form:
            d = x*sin(theta) + y*cos(theta)
        which allows you to have vertical lines.
        '''

        def fit(self, data):
            data_mean = data.mean(axis=0)
            x0, y0 = data_mean
            if data.shape[0] > 2:  # over determined
                u, v, w = np.linalg.svd(data - data_mean)
                vec = w[0]
                theta = math.atan2(vec[0], vec[1])
            elif data.shape[0] == 2:  # well determined
                theta = math.atan2(data[1, 0] - data[0, 0], data[1, 1] - data[0, 1])
            theta = (theta + math.pi * 5 / 2) % (2 * math.pi)
            d = x0 * math.sin(theta) + y0 * math.cos(theta)
            return d, theta

        def residuals(self, model, data):
            d, theta = model
            dfit = data[:, 0] * math.sin(theta) + data[:, 1] * math.cos(theta)
            return np.abs(d - dfit)

        def is_degenerate(self, sample):
            return False

    def ransac(self, data, model_class, min_samples, threshold, max_trials=100):
        '''
        Fits a model to data with the RANSAC algorithm.
        :param data: numpy.ndarray
            data set to which the model is fitted, must be of shape NxD where
            N is the number of data points and D the dimensionality of the data
        :param model_class: object
            object with the following methods implemented:
             * fit(data): return the computed model
             * residuals(model, data): return residuals for each data point
             * is_degenerate(sample): return boolean value if sample choice is
                degenerate
            see LinearLeastSquares2D class for a sample implementation
        :param min_samples: int
            the minimum number of data points to fit a model
        :param threshold: int or float
            maximum distance for a data point to count as an inlier
        :param max_trials: int, optional
            maximum number of iterations for random sample selection, default 100
        :returns: tuple
            best model returned by model_class.fit, best inlier indices
        '''

        best_model = None
        best_inlier_num = 0
        best_inliers = None
        data_idx = np.arange(data.shape[0])
        for _ in xrange(max_trials):
            sample = data[np.random.randint(0, data.shape[0], 2)]
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
        return best_model
