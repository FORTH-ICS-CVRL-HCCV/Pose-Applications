import cv2
import time
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import random

from clock import *
from square import *
from circle import *
from button import *
from functions import *
from hands import *
from json_handling import *

'''
This class takes care of the Add and Edit Menu
    Setters
    Getters
    Draw => Call the call function of every button
    Touch => Determines if any of the buttons are touched and does the appropriate action
    Finish => Does the Add or Edit action when we hit done
    Reset => Resets all the button values to the default ones so that we have them ready for the next Add
    Start_Edit => When we hit edit the button values are set to the ones of the object we are editing
'''

class Menu:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.status = False
        self.edit = ""
        self.rect = Button(x, y, (0, 255, 0), 75, "r")
        self.circle = Button(x+100, y, (255, 255, 0), 75, "c")
        self.red = Button(x, y+100, (255, 0, 0), 75, "0")
        self.blue = Button(x+100, y+100, (0, 0, 255), 75, "0")
        self.green = Button(x+200, y+100, (0, 255, 0), 75, "0")
        self.size = 100
        self.type = "rect"
        self.color_r = 0
        self.color_g = 0
        self.color_b = 0
        self.size_b = Button(x, y+200, (0, 255, 255), 75, "100")
        self.done = Button(x, y+300, (0, 255, 0), 75, "Done")
        self.cancel = Button(x+100, y+300, (255, 0, 0), 75, "Stop")


    def Get_Status(self):
        return self.status

    def Get_Edit(self):
        return self.edit

    def Set_Status(self, new_s):
        self.status = new_s

    def Set_Edit(self, new_e):
        self.edit = new_e

    def Draw(self, image):
        if(self.status == True):
            self.rect.Draw(image)
            self.circle.Draw(image)
            self.red.Draw(image)
            self.blue.Draw(image)
            self.green.Draw(image)
            self.size_b.Draw(image)
            self.done.Draw(image)
            self.cancel.Draw(image)

    def Touch(self, finger, img_w, img_h, rectangles, circles, index, type):
        if self.rect.DetectButtonTouch(finger, img_h, img_w) == True:
            if self.edit == "Add":
                self.type = "rect"
                self.rect.Set_Color((0, 255, 0))
                self.circle.Set_Color((255, 255, 0))
        elif self.circle.DetectButtonTouch(finger, img_h, img_w) == True:
            if self.edit == "Add":
                self.type = "circle"
                self.rect.Set_Color((255, 255, 0))
                self.circle.Set_Color((0, 255, 0))
        elif self.red.DetectButtonTouch(finger, img_h, img_w) == True:
            if self.color_r <255:
                self.color_r += 1
            else:
                self.color_r = 0
            self.red.Set_Text(str(self.color_r))
        elif self.green.DetectButtonTouch(finger, img_h, img_w) == True:
            if self.color_g <255:
                self.color_g += 1
            else:
                self.color_g = 0
            self.green.Set_Text(str(self.color_g))
        elif self.blue.DetectButtonTouch(finger, img_h, img_w) == True:
            if self.color_b <255:
                self.color_b += 1
            else:
                self.color_b = 0
            self.blue.Set_Text(str(self.color_b))
        elif self.size_b.DetectButtonTouch(finger, img_h, img_w) == True:
            if self.size < 300:
                self.size += 1
            else:
                self.size = 0
            self.size_b.Set_Text(str(self.size))
        elif self.done.DetectButtonTouch(finger, img_h, img_w) == True:
            self.Finish(rectangles, circles, index, type)
        elif self.cancel.DetectButtonTouch(finger, img_h, img_w) == True:
            self.Reset()

    def Finish(self, rectangles, circles, index, type):
        if self.edit == "Add":
            if self.type == "rect":
                CreateObject(rectangles, "rect", 300, 300, self.size, (self.color_r, self.color_g, self.color_b))
            elif self.type == "circle":
                CreateObject(circles, "circle", 300, 300, self.size, (self.color_r, self.color_g, self.color_b))
        elif self.edit == "Edit":
            self.type = type
            if self.type == "rect":
                EditObject(rectangles, "rect", 300, 300, self.size, (self.color_r, self.color_g, self.color_b), index)
            elif self.type == "circle":
                EditObject(circles, "circle", 300, 300, self.size, (self.color_r, self.color_g, self.color_b), index)
        self.Reset()

    def Reset(self):
        self.status = False
        self.size = 100
        self.type = "rect"
        self.color_r = 0
        self.color_g = 0
        self.color_b = 0

        self.red.Set_Text(str(self.color_r))
        self.green.Set_Text(str(self.color_g))
        self.blue.Set_Text(str(self.color_b))
        self.size_b.Set_Text(str(self.size))

    def Start_Edit(self, size, color, obj):
        if obj == "rect":
            self.rect.Set_Color((0, 255, 0))
            self.circle.Set_Color((255, 255, 0))
        elif obj == "circle":
            self.rect.Set_Color((255, 255, 0))
            self.circle.Set_Color((0, 255, 0))
        self.size = size
        self.color_r = color[0]
        self.color_g = color[1]
        self.color_b = color[2]

        self.red.Set_Text(str(self.color_r))
        self.green.Set_Text(str(self.color_g))
        self.blue.Set_Text(str(self.color_b))
        self.size_b.Set_Text(str(self.size))


