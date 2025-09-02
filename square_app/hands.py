import cv2
import time
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import random

class Hand:
    def __init__(self):
        self.landmarks = None
        self.handedness = None
        self.last_landmarks = None
        self.last_handedness = None

    def setup(self, land, side):
        self.last_handedness = self.handedness
        self.last_landmarks = self.landmarks
        self.handedness = side
        self.landmarks = land

    def isEmpty(self):
        self.last_handedness = self.handedness
        self.last_landmarks = self.landmarks
        self.handedness = None
        self.landmarks = None

    def Get_Landmark(self):
        return self.landmarks

    def Get_Last_Landmark(self):
        return self.last_landmarks

    def Get_Last_Handedness(self):
        return self.handedness

    def Get_Last_Handedness(self):
        return self.handedness

    def isActive(self):
        if(self.handedness == None or self.landmarks == None):
            return False
        else:
            return True
