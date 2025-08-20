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

    def Set_Text(self, text):
        self.text = text

    def Draw(self, image):
        cv2.rectangle(image, (self.x, self.y), (self.x + self.size, self.y + self.size), self.color, -1)
        cv2.putText(image, self.text, (self.x, self.y + self.size//2),  cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 1)
