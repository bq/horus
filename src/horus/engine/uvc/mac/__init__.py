'''
 ----------------------------------------------
 -  Horus Project
 -  Jesus Arroyo Torrens <jesus.arroyo@bq.com>
 -  May 2015 Mundo Readers S.L.
 ----------------------------------------------

(*)~----------------------------------------------------------------------------------
 Pupil - eye tracking platform
 Copyright (C) 2012-2014  Pupil Labs

 Distributed under the terms of the CC BY-NC-SA License.
 License details are in the file license.txt, distributed as part of this software.
----------------------------------------------------------------------------------~(*)
'''

"""
OOP style interface for uvcc c_types binding

Three classes:
    Camera_List holds Cam's instances,
    Cam is a class that contains infos about attached cameras
    Control is the actual Control with methods for getting and setting them.
"""
import sys
from raw import *

#logging
import logging
logger = logging.getLogger(__name__)

class Control(object):
    """docstring for uvcc_Control"""
    def __init__(self,name,i,handle):
        self.handle = handle
        self.name = name
        self.atb_name = name[9:].capitalize() #pretify the name
        self.order = i
        self.value = None
        self.assess_type()


    def assess_type(self):
        """
        find out if a control is active
        find out if the range is bool or int
        """
        self.value = None
        self.min = None
        self.max = None
        self.step    = None
        self.default = None
        self.menu    = None #some day in the future we will extract the control menu entries here.

        self.info = self.get_info()
        """
        D0 1 = Supports GET value requests      Capability
        D1 1 = Supports SET value requests      Capability
        D2 1 = Disabled due to automatic mode (under device control)    State
        D3 1 = Autoupdate Control   Capability
        D4 1 = Asynchronous Control Capability
        D5 1 = Disabled due to incompatibility with Commit state.   State
        """
        if self.info > 0 :  # Control supported
            self.value = self.get_val_from_device()
            self.min =  self.get_(UVC_GET_MIN)
            self.max =  self.get_(UVC_GET_MAX)
            self.step    =  self.get_(UVC_GET_RES)
            self.default =  self.get_(UVC_GET_DEF)

            if ((self.max,self.min) == (None,None)) or ((self.max,self.min) == (1,0)) :
                self.type  = "bool"
            # elif (self.max,self.min) == (None,None):
            #     ###I guess this should be a menu
            #     self.type  = "int"
            #     self.flags = "active"
            #     self.min = 0
            #     self.max = 20
            #     self.step = 1
            else:
                self.type  = "int"

            if self.info >> 3 & 1: # Disabled due to automatic mode (under device control)
                self.flags = "inactive"
            else:
                self.flags = "active"
        else:
            self.type  = "unknown type"
            self.flags = "control not supported"
            self.value = None

    def get_val_from_device(self):
        return uvccGetVal(self.name,self.handle)

    def get_val(self):
        return self.value

    def set_val(self,val):
        self.value = val
        return uvccSetVal(val,self.name,self.handle)

    def get_info(self):
        return uvccRequestInfo(self.name,self.handle)

    def get_(self,request):
        return uvccSendRequest(self.name,request,self.handle)


class Controls(dict):
    """docstring for Controls"""
    def __init__(self,uId):
        uvccInit()
        self.handle = uvccGetCamWithQTUniqueID(uId)
        assert self.handle is not None, "UVCC could not open camera based on uId %s" %uId
         # list of all controls implemented by uvcc,
         # the names evaluate to ints using a dict lookup in raw.py
        controls_str = uvcc_controls[:-1] #the last one is not a real control
        for i,c in enumerate(controls_str):
            self[c] = Control(c,i,self.handle)

    def update_from_device(self):
        for c in self.itervalues():
            if c.flags == "active":
                c.value = c.get_val_from_device()

    def load_defaults(self):
        for c in self.itervalues():
            if c.flags == "active" and c.default is not None:
                c.set_val(c.default)

    def __del__(self):
        #uvccReleaseCam(self.handle)
        uvccExit()

class Cam():
    """a simple class that only contains info about a camera"""
    def __init__(self,name,uId,src_id):
        self.src_id = src_id
        self.uId = uId
        self.name = name

class Camera_List(list):
    """docstring for uvcc_control"""

    def __init__(self):
        if getattr(sys, 'frozen', False):
            #explicit import needed when frozen
            import QTKit

        from QTKit import QTCaptureDevice,QTMediaTypeVideo
        qt_cameras =  QTCaptureDevice.inputDevicesWithMediaType_(QTMediaTypeVideo)
        for src_id,q in enumerate(qt_cameras):
            uId =  q.uniqueID()
            name = q.localizedDisplayName().encode('utf-8')
            self.append(Cam(name,uId,src_id))