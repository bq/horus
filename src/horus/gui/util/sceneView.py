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

import wx
import numpy
import time
import os
import gc
import traceback
import threading
import math
import platform
import cStringIO as StringIO

import OpenGL
OpenGL.ERROR_CHECKING = False
from OpenGL.GLU import *
from OpenGL.GL import *

from horus.util import profile, resources, meshLoader, model
from horus.gui.util import openglHelpers
from horus.gui.util import openglGui

class SceneView(openglGui.glGuiPanel):
	def __init__(self, parent):
		super(SceneView, self).__init__(parent)

		self._yaw = 30
		self._pitch = 60
		self._zoom = 300
		self._object = None
		self._objectShader = None
		self._objectLoadShader = None
		self._objColor = None
		self._mouseX = -1
		self._mouseY = -1
		self._mouseState = None
		self._mouse3Dpos = numpy.array([0,0,0], numpy.float32)
		self._viewTarget = numpy.array([0,0,0], numpy.float32)
		self._animView = None
		self._animZoom = None
		self._platformMesh = {}
		self._platformTexture = None

		self._viewport = None
		self._modelMatrix = None
		self._projMatrix = None
		self.tempMatrix = None

		self.viewMode = 'ply'

		self._moveVertical = False
		self._showDeleteMenu = True

		self._zOffset = 0
		self._hOffset = 20

		self.Bind(wx.EVT_MOUSEWHEEL, self.OnMouseWheel)
		self.Bind(wx.EVT_LEAVE_WINDOW, self.OnMouseLeave)
		self.Bind(wx.EVT_SHOW, self.onShow)

		self.updateProfileToControls()

	def onShow(self, event):
		if event.GetShow():
			self.GetParent().Layout()
			self.Layout()

	def __del__(self):
		if self._objectShader is not None:
			self._objectShader.release()
		if self._objectShaderNoLight is not None:
			self._objectShaderNoLight.release()
		if self._objectLoadShader is not None:
			self._objectLoadShader.release()
		if self._object is not None:
			if self._object._mesh is not None:
				if self._object._mesh.vbo is not None and self._object._mesh.vbo.decRef():
					self.glReleaseList.append(self._object._mesh.vbo)
					self._object._mesh.vbo.release()
				del self._object._mesh
			del self._object
		if self._platformMesh is not None:
			for _object in self._platformMesh.values():
				if _object._mesh is not None:
					if _object._mesh.vbo is not None and _object._mesh.vbo.decRef():
						self.glReleaseList.append(_object._mesh.vbo)
						_object._mesh.vbo.release()
					del _object._mesh
				del _object
		gc.collect()

	def createDefaultObject(self):
		self._clearScene()
		self._object = model.Model(None, isPointCloud=True)
		self._object._addMesh()
		self._object._mesh._prepareVertexCount(4000000)

	def appendPointCloud(self, point, color):
		#TODO: optimize
		if self._object is not None:
			if self._object._mesh is not None:
				for i in xrange(point.shape[1]):
					self._object._mesh._addVertex(point[0][i], point[1][i], point[2][i], color[0][i], color[1][i], color[2][i])
			#-- Conpute Z center
			if point.shape[1] > 0:
				zmax = max(point[2])
				if zmax > self._object._size[2]:
					self._object._size[2] = zmax
					self.centerHeight()
				self.QueueRefresh()
		#-- Delete objects
		del point
		del color

	def loadFile(self, filename):
		#-- Only one STL / PLY file can be active
		if filename is not None:
			ext = os.path.splitext(filename)[1].lower()
			if ext == '.ply' or ext == '.stl':
				modelFilename = filename
			if modelFilename:
				self.loadScene(modelFilename)
				self.centerObject()

	def centerHeight(self):
		if self._object is None:
			return
		height = self._object.getSize()[2] / 2
		if abs(height) > abs(self._hOffset):
			height -= self._hOffset
		newViewPos = numpy.array([self._object.getPosition()[0], self._object.getPosition()[1], height-self._zOffset])
		self._animView = openglGui.animation(self, self._viewTarget.copy(), newViewPos, 0.5)

	def loadScene(self, filename):
		try:
			self._clearScene()
			self._object = meshLoader.loadMesh(filename)
		except:
			traceback.print_exc()

	def _clearScene(self):
		if self._object is not None:
			if self._object._mesh is not None:
				if self._object._mesh.vbo is not None and self._object._mesh.vbo.decRef():
					self.glReleaseList.append(self._object._mesh.vbo)
					self._object._mesh.vbo.release()
				del self._object._mesh
			del self._object
			self._object = None
			self.centerObject()
			gc.collect()

	def centerObject(self):
		if self._object is None:
			newViewPos = numpy.array([0,0,-self._zOffset], numpy.float32)
			newZoom = 300
		else:
			height = self._object.getSize()[2] / 2
			if abs(height) > abs(self._hOffset):
				height -= self._hOffset
			newViewPos = numpy.array([self._object.getPosition()[0], self._object.getPosition()[1], height-self._zOffset])
			newZoom = self._object.getBoundaryCircle() * 4
		
		if newZoom > numpy.max(self._machineSize) * 3:
			newZoom = numpy.max(self._machineSize) * 3

		self._animZoom = openglGui.animation(self, self._zoom, newZoom, 0.5)
		self._animView = openglGui.animation(self, self._viewTarget.copy(), newViewPos, 0.5)

	def updateProfileToControls(self):
		self._machineSize = numpy.array([profile.getMachineSettingFloat('machine_width'), profile.getMachineSettingFloat('machine_depth'), profile.getMachineSettingFloat('machine_height')])
		self._objColor = profile.getPreferenceColor('model_color')
		
	def ShaderUpdate(self, v, f):
		s = openglHelpers.GLShader(v, f)
		if s.isValid():
			self._objectLoadShader.release()
			self._objectLoadShader = s
			self.QueueRefresh()

	def OnKeyDown(self, keyCode):
		if keyCode == wx.WXK_DELETE or keyCode == wx.WXK_NUMPAD_DELETE or (keyCode == wx.WXK_BACK and platform.system() == "Darwin"):
			if self._showDeleteMenu:
				if self._object is not None:
					self.onDeleteObject(None)
					self.QueueRefresh()
		if keyCode == wx.WXK_DOWN:
			if wx.GetKeyState(wx.WXK_SHIFT):
				self._zoom *= 1.2
				if self._zoom > numpy.max(self._machineSize) * 3:
					self._zoom = numpy.max(self._machineSize) * 3
			elif wx.GetKeyState(wx.WXK_CONTROL):
				self._zOffset += 5
			else:
				self._pitch -= 15
			self.QueueRefresh()
		elif keyCode == wx.WXK_UP:
			if wx.GetKeyState(wx.WXK_SHIFT):
				self._zoom /= 1.2
				if self._zoom < 1:
					self._zoom = 1
			elif wx.GetKeyState(wx.WXK_CONTROL):
				self._zOffset -= 5
			else:
				self._pitch += 15
			self.QueueRefresh()
		elif keyCode == wx.WXK_LEFT:
			self._yaw -= 15
			self.QueueRefresh()
		elif keyCode == wx.WXK_RIGHT:
			self._yaw += 15
			self.QueueRefresh()
		elif keyCode == wx.WXK_NUMPAD_ADD or keyCode == wx.WXK_ADD or keyCode == ord('+') or keyCode == ord('='):
			self._zoom /= 1.2
			if self._zoom < 1:
				self._zoom = 1
			self.QueueRefresh()
		elif keyCode == wx.WXK_NUMPAD_SUBTRACT or keyCode == wx.WXK_SUBTRACT or keyCode == ord('-'):
			self._zoom *= 1.2
			if self._zoom > numpy.max(self._machineSize) * 3:
				self._zoom = numpy.max(self._machineSize) * 3
			self.QueueRefresh()
		elif keyCode == wx.WXK_HOME:
			self._yaw = 30
			self._pitch = 60
			self.QueueRefresh()
		elif keyCode == wx.WXK_PAGEUP:
			self._yaw = 0
			self._pitch = 0
			self.QueueRefresh()
		elif keyCode == wx.WXK_PAGEDOWN:
			self._yaw = 0
			self._pitch = 90
			self.QueueRefresh()
		elif keyCode == wx.WXK_END:
			self._yaw = 90
			self._pitch = 90
			self.QueueRefresh()

		if keyCode == wx.WXK_F3 and wx.GetKeyState(wx.WXK_SHIFT):
			shaderEditor(self, self.ShaderUpdate, self._objectLoadShader.getVertexShader(), self._objectLoadShader.getFragmentShader())
		if keyCode == wx.WXK_F4 and wx.GetKeyState(wx.WXK_SHIFT):
			from collections import defaultdict
			from gc import get_objects
			self._beforeLeakTest = defaultdict(int)
			for i in get_objects():
				self._beforeLeakTest[type(i)] += 1
		if keyCode == wx.WXK_F5 and wx.GetKeyState(wx.WXK_SHIFT):
			from collections import defaultdict
			from gc import get_objects
			self._afterLeakTest = defaultdict(int)
			for i in get_objects():
				self._afterLeakTest[type(i)] += 1
			for k in self._afterLeakTest:
				if self._afterLeakTest[k]-self._beforeLeakTest[k]:
					print k, self._afterLeakTest[k], self._beforeLeakTest[k], self._afterLeakTest[k] - self._beforeLeakTest[k]

		if keyCode == wx.WXK_CONTROL:
			self._moveVertical = True

	def OnKeyUp(self, keyCode):
		if keyCode == wx.WXK_CONTROL:
			self._moveVertical = False

	def OnMouseDown(self,e):
		self._mouseX = e.GetX()
		self._mouseY = e.GetY()
		self._mouseClick3DPos = self._mouse3Dpos
		self._mouseClickFocus = self._object
		if e.ButtonDClick():
			self._mouseState = 'doubleClick'
		else:
			self._mouseState = 'dragOrClick'
		if self._mouseState == 'doubleClick':
			if e.GetButton() == 1:
				self.centerObject()
				self.QueueRefresh()

	def OnMouseUp(self, e):
		if e.LeftIsDown() or e.MiddleIsDown() or e.RightIsDown():
			return
		if self._mouseState == 'dragOrClick':
			if e.GetButton() == 3:
				if self._showDeleteMenu:
					menu = wx.Menu()
					if self._object is not None:
						self.Bind(wx.EVT_MENU, self.onDeleteObject, menu.Append(-1, _("Delete object")))
					if menu.MenuItemCount > 0:
						self.PopupMenu(menu)
					menu.Destroy()
		self._mouseState = None

	def setShowDeleteMenu(self, value=True):
		self._showDeleteMenu = value

	def onDeleteObject(self, event):
		if self._object is not None:
			dlg = wx.MessageDialog(self, _("Your current model will be erased.\nDo you really want to do it?"), _("Clear Point Cloud"), wx.YES_NO | wx.ICON_QUESTION)
			result = dlg.ShowModal() == wx.ID_YES
			dlg.Destroy()
			if result:
				self._clearScene()

	def OnMouseMotion(self,e):
		if e.Dragging() and self._mouseState is not None:
			if e.LeftIsDown() and not e.RightIsDown():
				self._mouseState = 'drag'
				if wx.GetKeyState(wx.WXK_SHIFT):
					a = math.cos(math.radians(self._yaw)) / 3.0
					b = math.sin(math.radians(self._yaw)) / 3.0
					self._viewTarget[0] += float(e.GetX() - self._mouseX) * -a
					self._viewTarget[1] += float(e.GetX() - self._mouseX) * b
					self._viewTarget[0] += float(e.GetY() - self._mouseY) * b
					self._viewTarget[1] += float(e.GetY() - self._mouseY) * a
				else:
					self._yaw += e.GetX() - self._mouseX
					self._pitch -= e.GetY() - self._mouseY
				if self._pitch > 170:
					self._pitch = 170
				if self._pitch < 10:
					self._pitch = 10
			elif (e.LeftIsDown() and e.RightIsDown()) or e.MiddleIsDown():
				self._mouseState = 'drag'
				self._zoom += e.GetY() - self._mouseY
				if self._zoom < 1:
					self._zoom = 1
				if self._zoom > numpy.max(self._machineSize) * 3:
					self._zoom = numpy.max(self._machineSize) * 3

		self._mouseX = e.GetX()
		self._mouseY = e.GetY()

	def OnMouseWheel(self, e):
		delta = float(e.GetWheelRotation()) / float(e.GetWheelDelta())
		delta = max(min(delta,4),-4)
		if self._moveVertical:
			self._zOffset -= 5 * delta
		else:
			self._zoom *= 1.0 - delta / 10.0
			if self._zoom < 1.0:
				self._zoom = 1.0
			if self._zoom > numpy.max(self._machineSize) * 3:
				self._zoom = numpy.max(self._machineSize) * 3
		self.Refresh()

	def OnMouseLeave(self, e):
		self._mouseX = -1

	def getMouseRay(self, x, y):
		if self._viewport is None:
			return numpy.array([0,0,0],numpy.float32), numpy.array([0,0,1],numpy.float32)

		p0 = openglHelpers.unproject(x, self._viewport[1] + self._viewport[3] - y, 0, self._modelMatrix, self._projMatrix, self._viewport)
		p1 = openglHelpers.unproject(x, self._viewport[1] + self._viewport[3] - y, 1, self._modelMatrix, self._projMatrix, self._viewport)
		if type(p0)!=type(None) and type(p1)!=type(None):
			p0 -= self._viewTarget
			p1 -= self._viewTarget
			return p0, p1
		else:
			return numpy.array([0,0,0],numpy.float32), numpy.array([0,0,1],numpy.float32)


	def _init3DView(self):
		# set viewing projection
		size = self.GetSize()
		glViewport(0, 0, size.GetWidth(), size.GetHeight())
		glLoadIdentity()

		glLightfv(GL_LIGHT0, GL_POSITION, [0.2, 0.2, 1.0, 0.0])

		glDisable(GL_RESCALE_NORMAL)
		glDisable(GL_LIGHTING)
		glDisable(GL_LIGHT0)
		glEnable(GL_DEPTH_TEST)
		glDisable(GL_CULL_FACE)
		glDisable(GL_BLEND)
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

		#glClearColor(0.0, 0.0, 0.0, 1.0)
		#glClearStencil(0)
		#glClearDepth(1.0)

		glMatrixMode(GL_PROJECTION)
		glLoadIdentity()
		aspect = float(size.GetWidth()) / float(size.GetHeight())
		gluPerspective(45.0, aspect, 1.0, numpy.max(self._machineSize) * 4)

		glMatrixMode(GL_MODELVIEW)
		glLoadIdentity()

		glBegin(GL_QUADS)
		glColor3f(1,1,1)
		glVertex3f(-1,-1,-1)
		glVertex3f(1,-1,-1)
		glColor3f(0,0,0)
		glVertex3f(1,1,-1)
		glVertex3f(-1,1,-1)
		glEnd()

		glClear(GL_DEPTH_BUFFER_BIT)
		#glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)
			
	def OnPaint(self,e):
		if self._animView is not None:
			self._viewTarget = self._animView.getPosition()
			if self._animView.isDone():
				self._animView = None
		if self._animZoom is not None:
			self._zoom = self._animZoom.getPosition()
			if self._animZoom.isDone():
				self._animZoom = None
		if self._objectShader is None: #TODO: add loading shaders from file(s)
			if openglHelpers.hasShaderSupport():
				self._objectShader = openglHelpers.GLShader("""
					varying float light_amount;

					void main(void)
					{
						gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
						gl_FrontColor = gl_Color;

						light_amount = abs(dot(normalize(gl_NormalMatrix * gl_Normal), normalize(gl_LightSource[0].position.xyz)));
						light_amount += 0.2;
					}
									""","""
					varying float light_amount;

					void main(void)
					{
						gl_FragColor = vec4(gl_Color.xyz * light_amount, gl_Color[3]);
					}
				""")
				self._objectShaderNoLight = openglHelpers.GLShader("""
					varying float light_amount;

					void main(void)
					{
						gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
						gl_FrontColor = gl_Color;

						light_amount = 1.0;
					}
									""","""
					varying float light_amount;

					void main(void)
					{
						gl_FragColor = vec4(gl_Color.xyz * light_amount, gl_Color[3]);
					}
				""")
				self._objectLoadShader = openglHelpers.GLShader("""
					uniform float intensity;
					uniform float scale;
					varying float light_amount;

					void main(void)
					{
						vec4 tmp = gl_Vertex;
						tmp.x += sin(tmp.z/5.0+intensity*30.0) * scale * intensity;
						tmp.y += sin(tmp.z/3.0+intensity*40.0) * scale * intensity;
						gl_Position = gl_ModelViewProjectionMatrix * tmp;
						gl_FrontColor = gl_Color;

						light_amount = abs(dot(normalize(gl_NormalMatrix * gl_Normal), normalize(gl_LightSource[0].position.xyz)));
						light_amount += 0.2;
					}
			""","""
				uniform float intensity;
				varying float light_amount;

				void main(void)
				{
					gl_FragColor = vec4(gl_Color.xyz * light_amount, 1.0-intensity);
				}
				""")
			if self._objectShader is None or not self._objectShader.isValid(): #Could not make shader.
				self._objectShader = openglHelpers.GLFakeShader()
				self._objectLoadShader = None

		self._init3DView()
		glTranslate(0,0,-self._zoom)
		glRotate(-self._pitch, 1,0,0)
		glRotate(self._yaw, 0,0,1)
		glTranslate(-self._viewTarget[0],-self._viewTarget[1],-self._viewTarget[2]-self._zOffset)

		self._viewport = glGetIntegerv(GL_VIEWPORT)
		self._modelMatrix = glGetDoublev(GL_MODELVIEW_MATRIX)
		self._projMatrix = glGetDoublev(GL_PROJECTION_MATRIX)

		glClearColor(1,1,1,1)
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)

		if self._mouseX > -1: # mouse has not passed over the opengl window.
			glFlush()
			n = glReadPixels(self._mouseX, self.GetSize().GetHeight() - 1 - self._mouseY, 1, 1, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8)[0][0] >> 8
			f = glReadPixels(self._mouseX, self.GetSize().GetHeight() - 1 - self._mouseY, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT)[0][0]
			#self.GetTopLevelParent().SetTitle(hex(n) + " " + str(f))
			self._mouse3Dpos = openglHelpers.unproject(self._mouseX, self._viewport[1] + self._viewport[3] - self._mouseY, f, self._modelMatrix, self._projMatrix, self._viewport)
			self._mouse3Dpos -= self._viewTarget
			self._mouse3Dpos[2] -= self._zOffset

		self._init3DView()
		glTranslate(0,0,-self._zoom)
		glRotate(-self._pitch, 1,0,0)
		glRotate(self._yaw, 0,0,1)
		glTranslate(-self._viewTarget[0],-self._viewTarget[1],-self._viewTarget[2]-self._zOffset)

		glStencilFunc(GL_ALWAYS, 1, 1)
		glStencilOp(GL_INCR, GL_INCR, GL_INCR)

		if self._object is not None:

			if self._object.isPointCloud() and openglHelpers.hasShaderSupport():
				self._objectShaderNoLight.bind()
			else:
				self._objectShader.bind()

			brightness = 1.0
			glStencilOp(GL_INCR, GL_INCR, GL_INCR)
			glEnable(GL_STENCIL_TEST)
			self._renderObject(self._object, brightness)

			glDisable(GL_STENCIL_TEST)
			glDisable(GL_BLEND)
			glEnable(GL_DEPTH_TEST)
			glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE)

			if self._object.isPointCloud() and openglHelpers.hasShaderSupport():
				self._objectShaderNoLight.unbind()
			else:
				self._objectShader.unbind()

		self._drawMachine()

	def _renderObject(self, obj, brightness = 0):
		glPushMatrix()
		glTranslate(obj.getPosition()[0], obj.getPosition()[1], obj.getSize()[2] / 2)

		if self.tempMatrix is not None:
			glMultMatrixf(openglHelpers.convert3x3MatrixTo4x4(self.tempMatrix))

		offset = obj.getDrawOffset()
		glTranslate(-offset[0], -offset[1], -offset[2] - obj.getSize()[2] / 2)

		glMultMatrixf(openglHelpers.convert3x3MatrixTo4x4(obj.getMatrix()))

		if obj.isPointCloud():
			if obj._mesh is not None:
				if obj._mesh.vbo is None or obj._mesh.vertexCount > obj._mesh.vbo._size:
					if obj._mesh.vbo is not None:
						obj._mesh.vbo.release()
					obj._mesh.vbo = openglHelpers.GLVBO(GL_POINTS, obj._mesh.vertexes[:obj._mesh.vertexCount], colorArray=obj._mesh.colors[:obj._mesh.vertexCount])
				obj._mesh.vbo.render()
		else:
			if obj._mesh is not None:
				if obj._mesh.vbo is None:
					obj._mesh.vbo = openglHelpers.GLVBO(GL_TRIANGLES, obj._mesh.vertexes[:obj._mesh.vertexCount], obj._mesh.normal[:obj._mesh.vertexCount])
				if brightness != 0:
					glColor4fv(map(lambda idx: idx * brightness, self._objColor))
				obj._mesh.vbo.render()
		glPopMatrix()

	def _drawMachine(self):
		glEnable(GL_BLEND)

		machine = profile.getMachineSetting('machine_type')
		if machine.startswith('ciclop'):

			glEnable(GL_CULL_FACE)
			#-- Draw Platform
			if machine not in self._platformMesh:
				mesh = meshLoader.loadMesh(resources.getPathForMesh(machine + '_platform.stl'))
				if mesh is not None:
					self._platformMesh[machine] = mesh
				else:
					self._platformMesh[machine] = None
				self._platformMesh[machine]._drawOffset = numpy.array([0,0,8.05], numpy.float32)
			glColor4f(0.6,0.6,0.6,0.5)
			self._objectShader.bind()
			self._renderObject(self._platformMesh[machine])
			self._objectShader.unbind()
			glDisable(GL_CULL_FACE)

		glDepthMask(False)
		
		size = numpy.array([profile.getProfileSettingFloat('roi_diameter'),
							profile.getProfileSettingFloat('roi_diameter'),
							profile.getProfileSettingFloat('roi_height')], numpy.float32)

		if profile.getProfileSettingBool('view_roi'):
			polys = profile.getSizePolygons(size)
			height = profile.getProfileSettingFloat('roi_height')
			circular = profile.getMachineSetting('machine_shape') == 'Circular'
			# Draw the sides of the build volume.
			glBegin(GL_QUADS)
			for n in xrange(0, len(polys[0])):
				if not circular:
					if n % 2 == 0:
						glColor4ub(5, 171, 231, 96)
					else:
						glColor4ub(5, 171, 231, 64)
				else:
					glColor4ub(5, 171, 231, 96)
					#glColor4ub(200, 200, 200, 150)

				glVertex3f(polys[0][n][0], polys[0][n][1], height)
				glVertex3f(polys[0][n][0], polys[0][n][1], 0)
				glVertex3f(polys[0][n-1][0], polys[0][n-1][1], 0)
				glVertex3f(polys[0][n-1][0], polys[0][n-1][1], height)
			glEnd()

			#Draw bottom and top of build volume.
			glColor4ub(5, 171, 231, 150)#128)
			#glColor4ub(200, 200, 200, 200)
			glBegin(GL_TRIANGLE_FAN)
			for p in polys[0][::-1]:
				glVertex3f(p[0], p[1], 0)
			glEnd()
			glBegin(GL_TRIANGLE_FAN)
			for p in polys[0][::-1]:
				glVertex3f(p[0], p[1], height)
			glEnd()

			quadric=gluNewQuadric();
			gluQuadricNormals(quadric, GLU_SMOOTH);
			gluQuadricTexture(quadric, GL_TRUE);
			glColor4ub(0, 100, 200, 150)
			
			gluCylinder(quadric,6,6,1,32,16);
			gluDisk(quadric, 0.0, 6, 32, 1); 

			glTranslate(0,0,height-1)
			gluDisk(quadric, 0.0, 6, 32, 1); 
			gluCylinder(quadric,6,6,1,32,16);
			glTranslate(0,0,-height+1)

		polys = profile.getMachineSizePolygons()
		
		#-- Draw checkerboard
		if self._platformTexture is None:
			self._platformTexture = openglHelpers.loadGLTexture('checkerboard.png')
			glBindTexture(GL_TEXTURE_2D, self._platformTexture)
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
		glColor4f(1,1,1,0.5)
		glBindTexture(GL_TEXTURE_2D, self._platformTexture)
		glEnable(GL_TEXTURE_2D)
		glBegin(GL_TRIANGLE_FAN)
		for p in polys[0]:
			glTexCoord2f(p[0]/20, p[1]/20)
			glVertex3f(p[0], p[1], 0)
		glEnd()

		glDepthMask(True)
		glDisable(GL_BLEND)

#TODO: Remove this or put it in a seperate file
class shaderEditor(wx.Dialog):
	def __init__(self, parent, callback, v, f):
		super(shaderEditor, self).__init__(parent, title="Shader editor", style=wx.DEFAULT_DIALOG_STYLE|wx.RESIZE_BORDER)
		self._callback = callback
		s = wx.BoxSizer(wx.VERTICAL)
		self.SetSizer(s)
		self._vertex = wx.TextCtrl(self, -1, v, style=wx.TE_MULTILINE)
		self._fragment = wx.TextCtrl(self, -1, f, style=wx.TE_MULTILINE)
		s.Add(self._vertex, 1, flag=wx.EXPAND)
		s.Add(self._fragment, 1, flag=wx.EXPAND)

		self._vertex.Bind(wx.EVT_TEXT, self.OnText, self._vertex)
		self._fragment.Bind(wx.EVT_TEXT, self.OnText, self._fragment)

		self.SetPosition(self.GetParent().GetPosition())
		self.SetSize((self.GetSize().GetWidth(), self.GetParent().GetSize().GetHeight()))
		self.Show()

	def OnText(self, e):
		self._callback(self._vertex.GetValue(), self._fragment.GetValue())
