import cv2
import time
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import random

'''
The Hand class takes care of all the hand functionality
    Getters
    Setters
    Setup => Sets up the Hand by coping the handedness nad landmarks
    isEmpty => If no hand of this type is found we take the handedness and landmarks to the old ones and keep the current empty
    isActive => Checks if the Hand is active or not
    checkPinch => We check if the hand is pinching by smoothing the distance and relative distance using medians and then having 
                a 3state system(Lower = Pinch, Middle = Maintain, Top = Off)
    GetPosition => Return the current position
    AveragePos => We get the current pinching position by smoothing the x and y cordinates using medians
    ClearPos => We clear the buffers for the AveragePos when the hand is not pinching to aviod miscalculations on the new position
'''

class Hand:
    def __init__(self):
        self.landmarks = None
        self.handedness = None
        self.last_landmarks = None
        self.last_handedness = None
        self.buffer_pinch = []
        self.buffer_rel = []
        self.buffer_pos = []
        self.state = False
        self.type = ""
        self.index = 0
        self.editing = False
        self.last_pinched_type = ""
        self.last_pinched_index = ""


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

    def Get_Type(self):
        return self.type
    
    def Get_Index(self):
        return self.index

    def Get_Editing(self):
        return self.editing
    
    def Get_Last_Pinched_Type(self):
        return self.last_pinched_type

    def Get_Last_Pinched_Index(self):
        return self.last_pinched_index

    def Set_Last_Pinched_Type(self, new_type):
        self.last_pinched_type = new_type

    def Set_Last_Pinched_Index(self, new_index):
        self.last_pinched_index = new_index

    def Set_Type(self, new_type):
        self.type = new_type
    
    def Set_Index(self, new_index):
        self.index = new_index

    def Set_Editing(self, new_editing):
        self.editing = new_editing

    def isActive(self):
        if(self.handedness == None or self.landmarks == None):
            return False
        else:
            return True

    def checkPinch(self, dist, rel):
        self.buffer_pinch.append(dist)
        distance = sorted(self.buffer_pinch)[len(self.buffer_pinch)//2]

        if(len(self.buffer_pinch) == 7):
            self.buffer_pinch.pop(0)

        self.buffer_rel.append(rel)
        relative_dist = sorted(self.buffer_rel)[len(self.buffer_rel)//2]

        if(len(self.buffer_rel) == 7):
            self.buffer_rel.pop(0)

        if distance > (relative_dist/3):
            self.state = False
        elif distance < (relative_dist/4):
            self.state = True
        
        return self.state

    def GetPosition(self, position):
        return self.AveragePos(position)

    
    def AveragePos(self, pos):
        self.buffer_pos.append(pos)
        buf_x = []
        buf_y = []
        for t in self.buffer_pos:
            buf_x.append(t[0])
            buf_y.append(t[1])

        position_x = sorted(buf_x)[len(buf_x)//2]
        position_y = sorted(buf_y)[len(buf_y)//2]

        if(len(self.buffer_pos) == 7):
            self.buffer_pos.pop(0)

        return [position_x, position_y]

    def ClearPos(self):
        self.buffer_pos.clear()
        self.index = 0
        self.type = ""
        self.editing = False
