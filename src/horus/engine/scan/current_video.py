# -*- coding: utf-8 -*-
# This file is part of the Horus Project

__author__ = 'Jesús Arroyo Torrens <jesus.arroyo@bq.com>'
__copyright__ = 'Copyright (C) 2014-2016 Mundo Reader S.L.'
__license__ = 'GNU General Public License v2 http://www.gnu.org/licenses/gpl2.html'

import cv2
import numpy as np

from horus import Singleton


@Singleton
class CurrentVideo(object):

    def __init__(self):
        self.mode = 'Texture'

        self.images = {}
        self.images['Texture'] = None
        self.images['Laser'] = None
        self.images['Gray'] = None
        self.images['Line'] = None

    def set_texture(self, image):
        self.images['Texture'] = image

    def set_laser(self, images):
        image = self._combine_images(images)
        self.images['Laser'] = image

    def set_gray(self, images):
        image = self._combine_images(images)
        image = cv2.merge((image, image, image))
        self.images['Gray'] = image

    def set_line(self, points, image):
        images = [None, None]
        if image is not None:
            for i in xrange(2):
                if points[i]:
                    images[i] = self._compute_line_image(points[i], image)
            image = self._combine_images(images)
            image = cv2.merge((image, image, image))
            self.images['Line'] = image

    def _combine_images(self, images):
        if images[0] is not None and images[1] is not None:
            return np.maximum(images[0], images[1])

        if images[0] is not None:
            return images[0]

        if images[1] is not None:
            return images[1]

    def _compute_line_image(self, points, image):
        if points is not None:
            u, v = points
            image = np.zeros_like(image)
            image[v.astype(int), np.around(u).astype(int) - 1] = 255
            image[v.astype(int), np.around(u).astype(int)] = 255
            image[v.astype(int), np.around(u).astype(int) + 1] = 255
            return image

    def capture(self):
        return self.images[self.mode]
