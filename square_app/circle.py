import cv2
import time
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import random


class Circle:
    def __init__(self, center_x, center_y, radius, color):
        self.center_x = center_x
        self.center_y = center_y
        self.radius = radius
        self.edit = False
        self.last_frame = False
        self.user = -1
        self.type = "circle"
        self.color = color

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

    def Get_User(self):
        return self.user
    
    def Get_Last_Frame(self):
        return self.last_frame

    def Get_Type(self):
        return self.type

    def Get_Color(self):
        return self.color

    def Set_Center_X(self, center_x):
        self.center_x = center_x

    def Set_Center_Y(self, center_y):
        self.center_y = center_y    

    def Set_Radius(self, new_radius):
        self.radius = new_radius

    def Set_Edit(self, new_edit):
        self.edit = new_edit

    def Set_User(self, new_user):
        self.user = new_user

    def Set_Last_Frame(self, new_frame):
        self.last_frame = new_frame
    
    def Set_Type(self, new_type):
        self.type = new_type

    def Set_Color(self, new_color):
        self.color = new_color

    def Move(self, index_finger_tip, img_w, img_h):
        self.last_frame = True
        
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
        cv2.circle(image, (self.center_x, self.center_y), self.radius, self.color, -1)

    def DetectCircleTouch(self, index_finger_tip, img_h, img_w):
        buffer = 0
        if self.edit == False:
            buffer = 10
        else:
            buffer = 100
        
        
        finger_x = int(index_finger_tip[0] * img_w)
        finger_y = int(index_finger_tip[1] * img_h)

        distance = np.sqrt((finger_x - self.center_x)**2 + (finger_y - self.center_y)**2)
        if (distance < self.radius + buffer):
            return True
        
        return False
