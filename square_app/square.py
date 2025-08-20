import cv2
import time
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import random


class Rectangle:
    def __init__(self, x, y, size):
        self.x = x
        self.y = y
        self.size = size
        self.edit = False
        self.type = "rectangle"

        self.dis_x = 0
        self.dis_y = 0

    def Get_X(self):
        return self.x

    def Get_Y(self):
        return self.y

    def Get_Size(self):
        return self.size

    def Get_Edit(self):
        return self.edit

    def Get_Type(self):
        return self.type

    def Set_X(self, new_x):
        self.x = new_x

    def Set_Y(self, new_y):
        self.y = new_y

    def Set_Size(self, new_size):
        self.size = new_size

    def Set_Edit(self, new_edit):
        self.edit = new_edit

    def Move(self, index_finger_tip, img_w, img_h):
        finger_x = int(index_finger_tip[0] * img_w)
        finger_y = int(index_finger_tip[1] * img_h)


        if self.edit == False:
            self.dis_x = np.abs(self.x - finger_x)
            self.dis_y = np.abs(self.y - finger_y)
            self.edit = True
        if(self.x > finger_x):
            self.x = finger_x + self.dis_x
        else:
            self.x = finger_x - self.dis_x

        if(self.y > finger_y):
            self.y = finger_y + self.dis_y
        else:
            self.y = finger_y - self.dis_y

    def Draw(self, image):
        cv2.rectangle(image, (self.x, self.y), (self.x + self.size, self.y + self.size), (0, 255, 0), -1)