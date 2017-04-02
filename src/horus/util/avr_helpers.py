# -*- coding: utf-8 -*-
# This file is part of the Horus Project

__author__ = 'Jesús Arroyo Torrens <jesus.arroyo@bq.com>'
__copyright__ = 'Copyright (C) 2014-2016 Mundo Reader S.L.'
__license__ = 'GNU General Public License v2 http://www.gnu.org/licenses/gpl2.html'

import os
import resources
from subprocess import Popen, PIPE, STDOUT

import logging
logger = logging.getLogger(__name__)

from horus.util import system as sys


class AvrError(Exception):
    pass


class AvrDude(object):

    def __init__(self, protocol="arduino", microcontroller="atmega328p",
                 baud_rate="19200", port=None):
        self.protocol = protocol
        self.microcontroller = microcontroller
        self.baud_rate = baud_rate

        if sys.is_windows():
            self.avrdude = os.path.abspath(resources.get_path_for_tools("avrdude.exe"))
        elif sys.is_darwin():
            self.avrdude = os.path.abspath(resources.get_path_for_tools("avrdude"))
        else:
            try:
                Popen(["avrdude"], stdout=PIPE, stderr=STDOUT)
                self.avrdude = "avrdude"
            except:
                self.avrdude = None

        if self.avrdude is None:
            raise AvrError('avrdude not installed')

        self.avrconf = os.path.abspath(resources.get_path_for_tools("avrdude.conf"))
        self.port = port

    def _run_command(self, flags=[], callback=None):
        config = dict(avrdude=self.avrdude, avrconf=self.avrconf)
        cmd = ['%(avrdude)s'] + flags
        cmd = [v % config for v in cmd]
        logger.info(' ' + ' '.join(cmd))
        p = Popen(cmd, stdout=PIPE, stderr=STDOUT, shell=sys.is_windows())
        out = ''
        while True:
            char = p.stdout.read(1)
            if not char:
                break
            out += char
            if 'not in sync' in out or \
               'Invalid' in out or \
               'is not responding' in out:
                break
            if char == '#':
                if callback is not None:
                    callback()
        p.kill()
        return out

    def flash(self, hex_path=None, clear_eeprom=False, callback=None):
        if hex_path is None:
            hex_path = resources.get_path_for_firmware("horus-fw.hex")
        if clear_eeprom:
            hex_path = resources.get_path_for_firmware("eeprom_clear.hex")
        flags = ['-C', '%(avrconf)s', '-c', self.protocol, '-p', self.microcontroller,
                 '-P', '%s' % self.port, '-b', str(self.baud_rate), '-D', '-U',
                 'flash:w:%s' % os.path.basename(hex_path)]
        try:
            cwd = os.getcwd()
            os.chdir(os.path.dirname(os.path.abspath(hex_path)))
            out = self._run_command(flags, callback)
            logger.info(' Upload completed')
        except Exception as e:
            print(e)
        finally:
            os.chdir(cwd)
        return out
