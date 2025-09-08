import cv2
import time
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import random


class Rectangle:
    def __init__(self, x, y, size, color):
        self.x = x
        self.y = y
        self.size = size
        self.edit = False
        self.type = "rectangle"
        self.color = color

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

    def Get_Color(self):
        return self.color

    def Set_X(self, new_x):
        self.x = new_x

    def Set_Y(self, new_y):
        self.y = new_y

    def Set_Size(self, new_size):
        self.size = new_size

    def Set_Edit(self, new_edit):
        self.edit = new_edit
    
    def Set_Color(self, new_color):
        self.color = new_color

    def Set_Type(self, new_type):
        self.type = new_type

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
        cv2.rectangle(image, (self.x, self.y), (self.x + self.size, self.y + self.size), self.color, -1)

    def DetectRectTouch(self, index_finger_tip, img_h, img_w):
        buffer = 0
        if self.edit == False:
            buffer = 10
        else:
            buffer = 100
        
        finger_x = int(index_finger_tip[0] * img_w)
        finger_y = int(index_finger_tip[1] * img_h)

        if (finger_x > self.x - buffer and finger_x < self.x + self.x + buffer):
            if (finger_y > self.y - buffer and finger_y < self.y + self.size + buffer):
                return True
        
        return False