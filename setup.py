#!/usr/bin/python
# coding=utf-8

import os
from setuptools import setup, find_packages

def package_data_dirs(source, sub_folders):
	dirs = []

	for d in sub_folders:
		for dirname, _, files in os.walk(os.path.join(source, d)):
			dirname = os.path.relpath(dirname, source)
			for f in files:
				dirs.append(os.path.join(dirname, f))
	return dirs

setup(name='Horus',
      version='0.1.1.1',
      author='Jesús Arroyo Torrens',
      author_email='jesus.arroyo@bq.com',
      description='Horus is a full software solution for 3D scanning',

      license='GPLv2',
      keywords = "horus ciclop scanning 3d",
      url='https://www.diwo.bq.com',

      packages = ['horus'],
      package_dir = {'horus': '.'},
      package_data = {'horus': package_data_dirs('.', ['doc', 'res', 'src'])},
      
      scripts=['pkg/linux/horus'],
      data_files=[('/usr/share/applications', ['pkg/linux/horus.desktop'])]
     )