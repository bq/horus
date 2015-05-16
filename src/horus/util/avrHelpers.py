#!/usr/bin/python
# -*- coding: utf-8 -*-
#-----------------------------------------------------------------------#
#                                                                       #
# This file is part of the Horus Project                                #
#                                                                       #
# Copyright (C) 2014-2015 Mundo Reader S.L.                             #
#                                                                       #
# Date: July, October 2014                                              #
# Author: Jesús Arroyo Torrens <jesus.arroyo@bq.com>                    #
#         Álvaro Velad Galván <alvaro.velad@bq.com>                     #
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

##                    ##
##-- TODO: refactor --##
##                    ##

import os
import system as sys
if not sys.isWindows():
    import fcntl
import resources
from subprocess import Popen, PIPE, CalledProcessError

from pathHelpers import path
from serialDevice import SerialDevice, ConnectionError

class FirmwareError(Exception):
    pass

class AvrDude(SerialDevice):
    def __init__(self, protocol="arduino", microcontroller="atmega328p", baudRate="19200", confPath=None, port=None):
        self.protocol = protocol
        self.microcontroller = microcontroller
        self.baudRate = baudRate

        if sys.isWindows():
        	self.avrdude = path(resources.getPathForToolsWindows("avrdude.exe")).abspath()
        else:
            self.avrdude = 'avrdude'
        
        if self.avrdude is None:
            raise FirmwareError('avrdude not installed')

        if confPath is None:
            self.avrconf = path(resources.getPathForTools("avrdude.conf")).abspath()
        else:
            self.avrconf = path(confPath).abspath()
        
        if port:
            self.port = port
            #print 'avrdude set to connect on port: %s' % self.port
        else:
            self.port = self.get_port(baudRate)
	    #print 'avrdude successfully connected on port: %s' % self.port

    def _runCommand(self, flags):
        config = dict(avrdude=self.avrdude, avrconf=self.avrconf)
        cmd = ['%(avrdude)s'] + flags
        cmd = [v % config for v in cmd]
        if sys.isWindows():
            p = Popen(cmd, stderr=PIPE)
        else:
            p = Popen(cmd, stderr=PIPE, close_fds=True)
            fcntl.fcntl(p.stderr.fileno(), fcntl.F_SETFL, fcntl.fcntl(p.stderr.fileno(), fcntl.F_GETFL) | os.O_NONBLOCK)

        return p

    def flash(self, hexPath=resources.getPathForFirmware("horus-fw.hex"), extraFlags=None):
        hexPath = path(hexPath)
        flags = ['-c', self.protocol, '-b', str(self.baudRate), '-p',
                 self.microcontroller, '-P', '%s' % self.port, '-U',
                 'flash:w:%s:i' % hexPath.name, '-C', '%(avrconf)s']
        if extraFlags is not None:
            flags.extend(extraFlags)
        try:
            cwd = os.getcwd()
            os.chdir(hexPath.parent)
            p = self._runCommand(flags)
        finally:
            os.chdir(cwd)
        return p