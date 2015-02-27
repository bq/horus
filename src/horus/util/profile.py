#!/usr/bin/python
# -*- coding: utf-8 -*-
#-----------------------------------------------------------------------#
#                                                                       #
# This file is part of the Horus Project                                #
#                                                                       #
# Copyright (C) 2014-2015 Mundo Reader S.L.                             #
# Copyright (C) 2013 David Braam from Cura Project                      #
#                                                                       #
# Date: June 2014                                                       #
# Author: Jesús Arroyo Torrens <jesus.arroyo@bq.com>                    #
#                                                                       #
# This program is free software: you can redistribute it and/or modify  #
# it under the terms of the GNU General Public License as published by  #
# the Free Software Foundation, either version 2 of the License, or     #
# (at your option) any later version.                                   #
#                                                                       #
# This program is distributed in the hope that it will be useful,       #
# but WITHOUT ANY WARRANTY; without even the implied warranty of        #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the          #
# GNU General Public License for more details.                          #
#                                                                       #
# You should have received a copy of the GNU General Public License     #
# along with this program. If not, see <http://www.gnu.org/licenses/>.  #
#                                                                       #
#-----------------------------------------------------------------------#

__author__ = "Jesús Arroyo Torrens <jesus.arroyo@bq.com>"
__license__ = "GNU General Public License v2 http://www.gnu.org/licenses/gpl.html"

import os
import traceback
import math
import numpy
import re
import zlib
import base64
import sys
import platform
import types

if sys.version_info[0] < 3:
	import ConfigParser
else:
	import configparser as ConfigParser

from horus.util import validators

#The settings dictionary contains a key/value reference to all possible settings. With the setting name as key.
settingsDictionary = {}
#The settings list is used to keep a full list of all the settings. This is needed to keep the settings in the proper order,
# as the dictionary will not contain insertion order.
settingsList = []

#Currently selected machine (by index) Cura support multiple machines in the same preferences and can switch between them.
# Each machine has it's own index and unique name.
_selectedMachineIndex = 0

class setting(object):
	"""
		A setting object contains a configuration setting. These are globally accessible trough the quick access functions
		and trough the settingsDictionary function.
		Settings can be:
		* profile settings (settings that effect the scan process and the scan result)
		* preferences (settings that effect how horus works and acts)
		* machine settings (settings that relate to the physical configuration of your machine)
		Settings have validators that check if the value is valid, but do not prevent invalid values!
		Settings have conditions that enable/disable this setting depending on other settings.
	"""
	def __init__(self, name, default, type, category, subcategory, store=True, tag=None):
		self._name = name
		self._label = subcategory
		self._tooltip = ''
		self._default = unicode(default)
		self._values = []
		self._type = type
		self._category = category
		self._subcategory = subcategory
		self._validators = []
		self._conditions = []
		self._store = store
		self._tag = tag

		if type is types.FloatType:
			validators.validFloat(self)
		elif type is types.IntType:
			validators.validInt(self)

		global settingsDictionary
		settingsDictionary[name] = self
		global settingsList
		settingsList.append(self)

	def setLabel(self, label, tooltip = ''):
		self._label = label
		self._tooltip = tooltip
		return self

	def setRange(self, minValue=None, maxValue=None):
		if len(self._validators) < 1:
			return
		self._validators[0].minValue = minValue
		self._validators[0].maxValue = maxValue
		return self

	def getMinValue(self):
		if len(self._validators) > 0:
			return _(self._validators[0].minValue)

	def getMaxValue(self):
		if len(self._validators) > 0:
			return _(self._validators[0].maxValue)

	def getLabel(self):
		return _(self._label)

	def getTooltip(self):
		return _(self._tooltip)

	def getCategory(self):
		return self._category

	def getTag(self):
		return self._tag

	def getSubCategory(self):
		return self._subcategory

	def isPreference(self):
		return self._category == 'preference'

	def isMachineSetting(self):
		return self._category == 'machine'

	def isProfile(self):
		return not self.isPreference() and not self.isMachineSetting()

	def isStorable(self):
		return self._store

	def getName(self):
		return self._name

	def getType(self):
		return self._type

	def getValue(self, index = None):
		if index is None:
			index = self.getValueIndex()
		if index >= len(self._values):
			return self._default
		return self._values[index]

	def getDefault(self):
		return self._default

	def setValue(self, value, index = None):
		if index is None:
			index = self.getValueIndex()
		while index >= len(self._values):
			self._values.append(self._default)
		self._values[index] = unicode(value)

	def getValueIndex(self):
		if self.isMachineSetting() or self.isProfile():
			global _selectedMachineIndex
			return _selectedMachineIndex
		return 0

	def validate(self):
		result = validators.SUCCESS
		msgs = []
		for validator in self._validators:
			res, err = validator.validate()
			if res == validators.ERROR:
				result = res
			elif res == validators.WARNING and result != validators.ERROR:
				result = res
			if res != validators.SUCCESS:
				msgs.append(err)
		return result, '\n'.join(msgs)

	def addCondition(self, conditionFunction):
		self._conditions.append(conditionFunction)

	def checkConditions(self):
		for condition in self._conditions:
			if not condition():
				return False
		return True

#########################################################
## Settings
#########################################################

#Define a fake _() function to fake the gettext tools in to generating strings for the profile settings.
def _(n):
	return n

#-- Settings

setting('serial_name', '/dev/ttyUSB0', str, 'basic', _('Serial Name'))
setting('baud_rate', 115200, [9600, 14400, 19200, 38400, 57600, 115200], 'basic', _('Baud rate'))
setting('camera_id', '/dev/video0', str, 'basic', _('Camera Id'))
setting('board', 'BT ATmega328', ['Arduino Uno', 'BT ATmega328'], 'basic', _('Board'))

setting('luminosity', _("Medium"), [_("High"), _("Medium"), _("Low")], 'basic', _('Luminosity'))

setting('brightness_control', 128, int, 'advanced', _('Brightness')).setRange(0, 255)
setting('contrast_control', 32, int, 'advanced', _('Contrast')).setRange(0, 255)
setting('saturation_control', 32, int, 'advanced', _('Saturation')).setRange(0, 255)
setting('exposure_control', 16, int, 'basic', _('Exposure')).setRange(1, 512)
setting('framerate_control', str('30'), [str('30'), str('25'), str('20'), str('15'), str('10'), str('5')], 'advanced', _('Framerate'))
setting('resolution_control', str('1280x960'), [str('1280x960'), str('960x720'), str('800x600'), str('320x240'), str('160x120')], 'advanced', _('Resolution'))
setting('use_distortion_control', False, bool, 'advanced', _('Use Distortion'))

setting('step_degrees_control', -0.45, float, 'basic', _('Step Degrees')).setRange(0.01)
setting('feed_rate_control', 200, int, 'advanced', _('Feed Rate')).setRange(1, 1000)
setting('acceleration_control', 200, int, 'advanced', _('Acceleration')).setRange(1, 1000)

setting('brightness_calibration', 100, int, 'advanced', _('Brightness')).setRange(0, 255)
setting('contrast_calibration', 32, int, 'advanced', _('Contrast')).setRange(0, 255)
setting('saturation_calibration', 100, int, 'advanced', _('Saturation')).setRange(0, 255)
setting('exposure_calibration', 16, int, 'basic', _('Exposure')).setRange(1, 512)
setting('framerate_calibration', str('30'), [str('30'), str('25'), str('20'), str('15'), str('10'), str('5')], 'advanced', _('Framerate'))
setting('resolution_calibration', str('1280x960'), [str('1280x960'), str('960x720'), str('800x600'), str('320x240'), str('160x120')], 'advanced', _('Resolution'))
setting('use_distortion_calibration', False, bool, 'advanced', _('Use Distortion'))

setting('scan_type', _("With Texture"), [_("Without Texture"), _("With Texture")], 'basic', _('Scan Type'))
setting('use_laser', _("Right"), [_("Left"), _("Right"), _("Both")], 'basic', _('Use Laser'))
setting('fast_scan', False, bool, 'advanced', _('Fast Scan (experimental)'))

setting('step_degrees_scanning', 0.45, float, 'basic', _('Step Degrees')).setRange(0.01)
setting('feed_rate_scanning', 200, int, 'advanced', _('Feed Rate')).setRange(1, 1000)
setting('acceleration_scanning', 300, int, 'advanced', _('Acceleration')).setRange(1, 1000)

setting('brightness_scanning', 100, int, 'advanced', _('Brightness')).setRange(0, 255)
setting('contrast_scanning', 32, int, 'advanced', _('Contrast')).setRange(0, 255)
setting('saturation_scanning', 32, int, 'advanced', _('Saturation')).setRange(0, 255)
setting('exposure_scanning', 16, int, 'basic', _('Exposure')).setRange(1, 512)
setting('laser_exposure_scanning', 6, int, 'basic', _('Exposure'), tag='no_texture').setRange(1, 512)
setting('color_exposure_scanning', 10, int, 'basic', _('Exposure'), tag='texture').setRange(1, 512)
setting('framerate_scanning', str('30'), [str('30'), str('25'), str('20'), str('15'), str('10'), str('5')], 'advanced', _('Framerate'))
setting('resolution_scanning', str('1280x960'), [str('1280x960'), str('960x720'), str('800x600'), str('320x240'), str('160x120')], 'advanced', _('Resolution'))
setting('use_distortion_scanning', False, bool, 'advanced', _('Use Distortion'))

setting('img_type', 'laser', ['laser', 'gray', 'line', 'color'], 'advanced', _('Image Type'))

setting('use_open', True, bool, 'advanced', _('Use Open'), tag='texture')
setting('open_value', 2, int, 'advanced', _('Open'), tag='texture').setRange(1, 10)
setting('use_threshold', True, bool, 'advanced', _('Use Threshold'), tag='texture')
setting('threshold_value', 25, int, 'advanced', _('Threshold'), tag='texture').setRange(0, 255)
setting('use_cr_threshold', True, bool, 'advanced', _('Use Threshold'), tag='no_texture')
setting('cr_threshold_value', 140, int, 'advanced', _('Threshold'), tag='no_texture').setRange(0, 255)

setting('view_roi', False, bool, 'advanced', _('View ROI'))

setting('roi_diameter', 200, int, 'advanced', _('Diameter')).setRange(0, 250)
setting('roi_height', 200, int, 'advanced', _('Height')).setRange(0, 250)

setting('point_cloud_color', 'AAAAAA', str, 'advanced', _('Choose Point Cloud Color'))

setting('adjust_laser', True, bool, 'advanced', _('Adjust Laser'))

setting('camera_matrix', ([[1425.0,0.0,480.0],[0.0,1425.0,640.0],[0.0,0.0,1.0]]), numpy.ndarray, 'advanced', _('Calibration Matrix'))
setting('distortion_vector',([0.0,0.0,0.0,0.0,0.0]),numpy.ndarray,'advanced',_('Distortion Vector'))

setting('laser_threshold_value', 120., int, 'advanced', _('Laser Threshold')).setRange(0, 255)

setting('distance_left', 0.0, float, 'advanced', _('Distance'))
setting('normal_left', ([0.0,0.0,0.0]), numpy.ndarray, 'advanced', _('Normal'))
setting('distance_right', 0.0, float, 'advanced', _('Distance'))
setting('normal_right', ([0.0,0.0,0.0]), numpy.ndarray, 'advanced', _('Normal'))

setting('rotation_matrix', ([[0.0,1.0,0.0],[0.0,0.0,-1.0],[-1.0,0.0,0.0]]), numpy.ndarray, 'advanced', _('Rotation Matrix'))
setting('translation_vector', ([5.0,80.0,320.0]), numpy.ndarray, 'advanced', _('Translation Matrix'))

setting('pattern_rows', 6, int, 'advanced', _('Pattern Rows'))
setting('pattern_columns', 11, int, 'advanced', _('Pattern Columns'))
setting('square_width', 13, int, 'advanced', _('Square width'))
setting('pattern_distance', 0, float, 'advanced', _('Pattern Distance'))
setting('extrinsics_step', -5.0, float, 'advanced', _('Extrinsics Step'), False)

setting('laser_coordinates', ([[480.0,480.0],[480.0,480.0]]), numpy.ndarray, 'advanced', _('Laser Coordinates'))
setting('laser_origin', ([0.0,0.0,0.0]), numpy.ndarray, 'advanced', _('Laser Origin'))
setting('laser_normal', ([0.0,0.0,0.0]), numpy.ndarray, 'advanced', _('Laser Normal'))

setting('left_button', '', str, 'basic', _('Left'), False)
setting('right_button', '', str, 'basic', _('Right'), False)
setting('move_button', '', str, 'basic', _('Move'), False)
setting('enable_button', '', str, 'basic', _('Enable'), False)
setting('gcode_gui', '', str, 'advanced', _('Send'), False)
setting('ldr_value', '', str, 'advanced', _('Send'), False)

setting('machine_name', '', str, 'machine', 'hidden')
setting('machine_type', 'ciclop', str, 'machine', 'hidden')
setting('machine_width', '200', float, 'machine', 'hidden').setLabel(_("Maximum width (mm)"), _("Size of the machine in mm"))
setting('machine_depth', '200', float, 'machine', 'hidden').setLabel(_("Maximum depth (mm)"), _("Size of the machine in mm"))
setting('machine_height', '200', float, 'machine', 'hidden').setLabel(_("Maximum height (mm)"), _("Size of the machine in mm"))
setting('machine_center_is_zero', 'True', bool, 'machine', 'hidden').setLabel(_("Machine center 0,0"), _("Machines firmware defines the center of the bed as 0,0 instead of the front left corner."))
setting('machine_shape', 'Circular', ['Square','Circular'], 'machine', 'hidden').setLabel(_("Build area shape"), _("The shape of machine build area."))


##-- Preferences

setting('language', 'English', str, 'preference', 'hidden').setLabel(_('Language'), _('Change the language in which Horus runs. Switching language requires a restart of Horus'))
setting('workbench', 'scanning', ['control', 'calibration', 'scanning'], 'preference', 'hidden')
setting('show_welcome', True, bool, 'preference', 'hidden')
setting('basic_mode', False, bool, 'preference', 'hidden')
setting('view_control_panel', True, bool, 'preference', 'hidden')
setting('view_control_video', True, bool, 'preference', 'hidden')
setting('view_calibration_panel', True, bool, 'preference', 'hidden')
setting('view_calibration_video', True, bool, 'preference', 'hidden')
setting('view_scanning_panel', False, bool, 'preference', 'hidden')
setting('view_scanning_video', False, bool, 'preference', 'hidden')
setting('view_scanning_scene', True, bool, 'preference', 'hidden')

# TODO: change default last file
setting('last_files', [], str, 'preference', 'hidden')
setting('last_file', os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'resources', 'example', 'default.stl')), str, 'preference', 'hidden')
setting('last_profile', os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'resources', 'example', 'default.ini')), str, 'preference', 'hidden')

setting('model_colour', '#888899', str, 'preference', 'hidden').setLabel(_('Model colour'), _('Display color for first extruder'))

#Remove fake defined _() because later the localization will define a global _()
del _

#########################################################
## Profile and preferences functions
#########################################################

def getSubCategoriesFor(category):
	done = {}
	ret = []
	for s in settingsList:
		if s.getCategory() == category and not s.getSubCategory() in done and s.checkConditions():
			done[s.getSubCategory()] = True
			ret.append(s.getSubCategory())
	return ret

def getSettingsForCategory(category, subCategory = None):
	ret = []
	for s in settingsList:
		if s.getCategory() == category and (subCategory is None or s.getSubCategory() == subCategory) and s.checkConditions():
			ret.append(s)
	return ret

## Profile functions
def getBasePath():
	"""
	:return: The path in which the current configuration files are stored. This depends on the used OS.
	"""
	if platform.system() == "Windows":
		basePath = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
		#If we have a frozen python install, we need to step out of the library.zip
		if hasattr(sys, 'frozen'):
			basePath = os.path.normpath(os.path.join(basePath, ".."))
	else:
		basePath = os.path.expanduser('~/.horus/')
	if not os.path.isdir(basePath):
		try:
			os.makedirs(basePath)
		except:
			print "Failed to create directory: %s" % (basePath)
	return basePath

def getAlternativeBasePaths():
	"""
	Search for alternative installations of Horus and their preference files. Used to load configuration from older versions of Horus.
	"""
	paths = []
	basePath = os.path.normpath(os.path.join(getBasePath(), '..'))
	for subPath in os.listdir(basePath):
		path = os.path.join(basePath, subPath)
		if os.path.isdir(path) and os.path.isfile(os.path.join(path, 'preferences.ini')) and path != getBasePath():
			paths.append(path)
		path = os.path.join(basePath, subPath, 'Horus')
		if os.path.isdir(path) and os.path.isfile(os.path.join(path, 'preferences.ini')) and path != getBasePath():
			paths.append(path)
	return paths

def getDefaultProfilePath():
	"""
	:return: The default path where the currently used profile is stored and loaded on open and close of Horus.
	"""
	return os.path.join(getBasePath(), 'current_profile.ini')

def loadProfile(filename, allMachines = False):
	"""
		Read a profile file as active profile settings.
	:param filename:    The ini filename to save the profile in.
	:param allMachines: When False only the current active profile is saved. If True all profiles for all machines are saved.
	"""
	global settingsList
	profileParser = ConfigParser.ConfigParser()
	try:
		profileParser.read(filename)
	except ConfigParser.ParsingError:
		return
	if allMachines:
		n = 0
		while profileParser.has_section('profile_%d' % (n)):
			for set in settingsList:
				if set.isPreference():
					continue
				section = 'profile_%d' % (n)
				if profileParser.has_option(section, set.getName()):
					set.setValue(unicode(profileParser.get(section, set.getName()), 'utf-8', 'replace'), n)
			n += 1
	else:
		for set in settingsList:
			if set.isPreference():
				continue
			section = 'profile'
			if profileParser.has_option(section, set.getName()):
				set.setValue(unicode(profileParser.get(section, set.getName()), 'utf-8', 'replace'))
				
def saveProfile(filename, allMachines = False):
	"""
		Save the current profile to an ini file.
	:param filename:    The ini filename to save the profile in.
	:param allMachines: When False only the current active profile is saved. If True all profiles for all machines are saved.
	"""
	global settingsList
	profileParser = ConfigParser.ConfigParser()
	if allMachines:
		for set in settingsList:
			if set.isPreference() or set.isMachineSetting():
				continue
			for n in xrange(0, getMachineCount()):
				section = 'profile_%d' % (n)
				if not profileParser.has_section(section):
					profileParser.add_section(section)
				profileParser.set(section, set.getName(), set.getValue(n).encode('utf-8'))
	else:
		profileParser.add_section('profile')
		for set in settingsList:
			if set.isPreference() or set.isMachineSetting():
				continue
			if set.isStorable():
				profileParser.set('profile', set.getName(), set.getValue().encode('utf-8'))

	profileParser.write(open(filename, 'w'))

def resetProfile():
	""" Reset the profile for the current machine to default. """
	global settingsList
	for set in settingsList:
		if not set.isProfile():
			continue
		set.setValue(set.getDefault())

def resetProfileSetting(name):
	""" Reset only the especified profile setting """
	global settingsDictionary
	if name in settingsDictionary and settingsDictionary[name].isProfile():
		settingsDictionary[name].setValue(settingsDictionary[name]._default)

def setProfileFromString(options):
	"""
	Parse an encoded string which has all the profile settings stored inside of it.
	Used in combination with getProfileString to ease sharing of profiles.
	"""
	options = base64.b64decode(options)
	options = zlib.decompress(options)
	(profileOpts, alt) = options.split('\f', 1)
	global settingsDictionary
	for option in profileOpts.split('\b'):
		if len(option) > 0:
			(key, value) = option.split('=', 1)
			if key in settingsDictionary:
				if settingsDictionary[key].isProfile():
					settingsDictionary[key].setValue(value)
	for option in alt.split('\b'):
		if len(option) > 0:
			(key, value) = option.split('=', 1)
			if key in settingsDictionary:
				settingsDictionary[key].setValue(value)

def getProfileString():
	"""
	Get an encoded string which contains all profile settings.
	Used in combination with setProfileFromString to share settings in files, forums or other text based ways.
	"""
	p = []
	alt = []
	global settingsList
	for set in settingsList:
		if set.isProfile():
			p.append(set.getName() + "=" + set.getValue().encode('utf-8'))
	ret = '\b'.join(p) + '\f' + '\b'.join(alt)
	ret = base64.b64encode(zlib.compress(ret, 9))
	return ret

def insertNewlines(string, every=64): #This should be moved to a better place then profile.
	lines = []
	for i in xrange(0, len(string), every):
		lines.append(string[i:i+every])
	return '\n'.join(lines)

def getPreferencesString():
	"""
	:return: An encoded string which contains all the current preferences.
	"""
	p = []
	global settingsList
	for set in settingsList:
		if set.isPreference():
			p.append(set.getName() + "=" + set.getValue().encode('utf-8'))
	ret = '\b'.join(p)
	ret = base64.b64encode(zlib.compress(ret, 9))
	return ret

def getProfileSettingObject(name):
	""" """
	global settingsList
	for set in settingsList:
		if set.getName() is name:
			return set

#TODO: get profile setting using getType

def getProfileSetting(name):
	"""
		Get the value of an profile setting.
	:param name: Name of the setting to retrieve.
	:return:     Value of the current setting.
	"""
	global settingsDictionary
	if name in settingsDictionary and settingsDictionary[name].isProfile():
		return settingsDictionary[name].getValue()
	traceback.print_stack()
	sys.stderr.write('Error: "%s" not found in profile settings\n' % (name))
	return ''

def getDefaultProfileSetting(name):
	"""
		Get the default value of a profile setting.
	:param name: Name of the setting to retrieve.
	:return:     Value of the current setting.
	"""
	global settingsDictionary
	if name in settingsDictionary and settingsDictionary[name].isProfile():
		return settingsDictionary[name].getDefault()
	traceback.print_stack()
	sys.stderr.write('Error: "%s" not found in profile settings\n' % (name))
	return ''

def getProfileSettingInteger(name):
	try:
		setting = getProfileSetting(name)
		return int(eval(setting, {}, {}))
	except:
		return 0.0

def getProfileSettingFloat(name):
	try:
		setting = getProfileSetting(name).replace(',', '.')
		return float(eval(setting, {}, {}))
	except:
		return 0.0

def getProfileSettingBool(name):
	try:
		setting = getProfileSetting(name)
		return bool(eval(setting, {}, {}))
	except:
		return False

def getProfileSettingNumpy(name):
	try:
		setting = getProfileSetting(name)

		return numpy.array(eval(setting, {}, {}))
	except:
		return []

def putProfileSetting(name, value):
	""" Store a certain value in a profile setting. """
	global settingsDictionary
	if name in settingsDictionary and settingsDictionary[name].isProfile():
		settingsDictionary[name].setValue(value)

def putProfileSettingNumpy(name, value):
	reprValue=repr(value)
	reprValue=reprValue.replace('\n','')
	reprValue=reprValue.replace('array(','')
	reprValue=reprValue.replace(')','')
	reprValue=reprValue.replace(' ','')
	putProfileSetting(name,reprValue)

def isProfileSetting(name):
	""" Check if a certain key name is actually a profile value. """
	global settingsDictionary
	if name in settingsDictionary and settingsDictionary[name].isProfile():
		return True
	return False

## Preferences functions
def getPreferencePath():
	"""
	:return: The full path of the preference ini file.
	"""
	return os.path.join(getBasePath(), 'preferences.ini')

def getPreferenceFloat(name):
	"""
	Get the float value of a preference, returns 0.0 if the preference is not a invalid float
	"""
	try:
		setting = getPreference(name).replace(',', '.')
		return float(eval(setting, {}, {}))
	except:
		return 0.0

def getPreferenceBool(name):
	"""
	Get the float value of a preference, returns 0.0 if the preference is not a invalid float
	"""
	try:
		setting = getPreference(name)
		return bool(eval(setting, {}, {}))
	except:
		return False

def getPreferenceColour(name):
	"""
	Get a preference setting value as a color array. The color is stored as #RRGGBB hex string in the setting.
	"""
	colorString = getPreference(name)
	return [float(int(colorString[1:3], 16)) / 255, float(int(colorString[3:5], 16)) / 255, float(int(colorString[5:7], 16)) / 255, 1.0]

def loadPreferences(filename):
	"""
	Read a configuration file as global config
	"""
	global settingsList
	profileParser = ConfigParser.ConfigParser()
	try:
		profileParser.read(filename)
	except ConfigParser.ParsingError:
		return

	for set in settingsList:
		if set.isPreference():
			if profileParser.has_option('preference', set.getName()):
				set.setValue(unicode(profileParser.get('preference', set.getName()), 'utf-8', 'replace'))

	n = 0
	while profileParser.has_section('machine_%d' % (n)):
		for set in settingsList:
			if set.isMachineSetting():
				if profileParser.has_option('machine_%d' % (n), set.getName()):
					set.setValue(unicode(profileParser.get('machine_%d' % (n), set.getName()), 'utf-8', 'replace'), n)
		n += 1

def savePreferences(filename):
	global settingsList
	#Save the current profile to an ini file
	parser = ConfigParser.ConfigParser()
	parser.add_section('preference')

	for set in settingsList:
		if set.isPreference():
			parser.set('preference', set.getName(), set.getValue().encode('utf-8'))

	n = 0
	while getMachineSetting('machine_name', n) != '':
		parser.add_section('machine_%d' % (n))
		for set in settingsList:
			if set.isMachineSetting():
				parser.set('machine_%d' % (n), set.getName(), set.getValue(n).encode('utf-8'))
		n += 1
	parser.write(open(filename, 'w'))

def getPreference(name):
	global settingsDictionary
	if name in settingsDictionary and settingsDictionary[name].isPreference():
		return settingsDictionary[name].getValue()
	traceback.print_stack()
	sys.stderr.write('Error: "%s" not found in preferences\n' % (name))
	return ''

def putPreference(name, value):
	#Check if we have a configuration file loaded, else load the default.
	global settingsDictionary
	if name in settingsDictionary and settingsDictionary[name].isPreference():
		settingsDictionary[name].setValue(value)
		savePreferences(getPreferencePath())
		return
	traceback.print_stack()
	sys.stderr.write('Error: "%s" not found in preferences\n' % (name))

def isPreference(name):
	global settingsDictionary
	if name in settingsDictionary and settingsDictionary[name].isPreference():
		return True
	return False

def getMachineSettingFloat(name, index = None):
	try:
		setting = getMachineSetting(name, index).replace(',', '.')
		return float(eval(setting, {}, {}))
	except:
		return 0.0

def loadMachineSettings(filename):
	global settingsList
	#Read a configuration file as global config
	profileParser = ConfigParser.ConfigParser()
	try:
		profileParser.read(filename)
	except ConfigParser.ParsingError:
		return

	for set in settingsList:
		if set.isMachineSetting():
			if profileParser.has_option('machine', set.getName()):
				set.setValue(unicode(profileParser.get('machine', set.getName()), 'utf-8', 'replace'))
	checkAndUpdateMachineName()

def getMachineSetting(name, index = None):
	global settingsDictionary
	if name in settingsDictionary and settingsDictionary[name].isMachineSetting():
		return settingsDictionary[name].getValue(index)
	traceback.print_stack()
	sys.stderr.write('Error: "%s" not found in machine settings\n' % (name))
	return ''

def putMachineSetting(name, value):
	#Check if we have a configuration file loaded, else load the default.
	global settingsDictionary
	if name in settingsDictionary and settingsDictionary[name].isMachineSetting():
		settingsDictionary[name].setValue(value)
	savePreferences(getPreferencePath())

def isMachineSetting(name):
	global settingsDictionary
	if name in settingsDictionary and settingsDictionary[name].isMachineSetting():
		return True
	return False

def getMachineCenterCoords():
	if getMachineSetting('machine_center_is_zero') == 'True':
		return [0, 0]
	return [getMachineSettingFloat('machine_width') / 2, getMachineSettingFloat('machine_depth') / 2]

#Returns a list of convex polygons, first polygon is the allowed area of the machine,
# the rest of the polygons are the dis-allowed areas of the machine.
def getMachineSizePolygons():
	size = numpy.array([getMachineSettingFloat('machine_width'), getMachineSettingFloat('machine_depth'), getMachineSettingFloat('machine_height')], numpy.float32)
	return getSizePolygons(size)

def getSizePolygons(size):
	ret = []
	if getMachineSetting('machine_shape') == 'Circular':
		circle = []
		steps = 32
		for n in xrange(0, steps):
			circle.append([math.cos(float(n)/steps*2*math.pi) * size[0]/2, math.sin(float(n)/steps*2*math.pi) * size[1]/2])
		ret.append(numpy.array(circle, numpy.float32))

	if getMachineSetting('machine_type') == 'ciclop':
		w = 20
		h = 20
		ret.append(numpy.array([[-size[0]/2,-size[1]/2],[-size[0]/2+w+2,-size[1]/2], [-size[0]/2+w,-size[1]/2+h], [-size[0]/2,-size[1]/2+h]], numpy.float32))
		ret.append(numpy.array([[ size[0]/2-w-2,-size[1]/2],[ size[0]/2,-size[1]/2], [ size[0]/2,-size[1]/2+h],[ size[0]/2-w,-size[1]/2+h]], numpy.float32))
		ret.append(numpy.array([[-size[0]/2+w+2, size[1]/2],[-size[0]/2, size[1]/2], [-size[0]/2, size[1]/2-h],[-size[0]/2+w, size[1]/2-h]], numpy.float32))
		ret.append(numpy.array([[ size[0]/2, size[1]/2],[ size[0]/2-w-2, size[1]/2], [ size[0]/2-w, size[1]/2-h],[ size[0]/2, size[1]/2-h]], numpy.float32))
	
	return ret