import cv2
import time
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import random


class Circle:
    def __init__(self, center_x, center_y, radius):
        self.center_x = center_x
        self.center_y = center_y
        self.radius = radius
        self.edit = False
        self.type = "circle"

        self.dis_x = 0
        self.dis_y = 0

    def Get_Center_X(self):
        return self.center_x

    def Get_Center_Y(self):
        return self.center_y

    def Get_Radius(self):
        return self.radius

    def Get_Edit(self):
        return self.edit

    def Get_Type(self):
        return self.type

    def Set_Center_X(self, center_x):
        self.center_x = center_x

    def Set_Center_Y(self, center_y):
        self.center_y = center_y    

    def Set_Radius(self, new_radius):
        self.radius = new_radius

    def Set_Edit(self, new_edit):
        self.edit = new_edit

    def Move(self, index_finger_tip, img_w, img_h):
        finger_x = int(index_finger_tip[0] * img_w)
        finger_y = int(index_finger_tip[1] * img_h)


        if self.edit == False:
            self.dis_x = np.abs(self.center_x - finger_x)
            self.dis_y = np.abs(self.center_y - finger_y)
            self.edit = True
        if(self.center_x > finger_x):
            self.center_x = finger_x + self.dis_x
        else:
            self.center_x = finger_x - self.dis_x

        if(self.center_y > finger_y):
            self.center_y = finger_y + self.dis_y
        else:
            self.center_y = finger_y - self.dis_y

    def Draw(self, image):
        cv2.circle(image, (self.center_x, self.center_y), self.radius, (0, 255, 0), -1)
