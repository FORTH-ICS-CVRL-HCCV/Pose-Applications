import cv2
import time
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import random

class Button:
    def __init__(self, x, y, color, size, text):
        self.x = x
        self.y = y
        self.color = color
        self.size = size
        self.text = text

    def Get_X(self):
        return self.x

    def Get_Y(self):
        return self.y
    
    def Get_Color(self):
        return self.color

    def Get_Size(self):
        return self.size

    def Get_Text(self):
        return self.text

    def Set_X(self, x):
        self.x = x

    def Set_Y(self, y):
        self.y = y    

    def Set_Size(self, size):
        self.size = size

    def Set_Color(self, color):
        self.color = color

    def Set_Text(self, text):
        self.text = text

    def Draw(self, image):
        cv2.rectangle(image, (self.x, self.y), (self.x + self.size, self.y + self.size), self.color, -1)
        cv2.putText(image, self.text, (self.x, self.y + self.size//2),  cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 1)

    def DetectButtonTouch(self, index_finger_tip, img_h, img_w):
        buffer = 10
        finger_x = int(index_finger_tip[0] * img_w)
        finger_y = int(index_finger_tip[1] * img_h)

        if (finger_x > self.x - buffer and finger_x < self.x + self.size + buffer):
            if (finger_y > self.y - buffer and finger_y < self.y + self.size + buffer):
                return True
        
        return False
