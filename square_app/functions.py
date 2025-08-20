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

def CameraSet():
    #Run from webcam
    videoWidth=1920 # Or 640 
    videoHeight=1080  # Or 480

    cv2.namedWindow("Hands Detection", cv2.WND_PROP_FULLSCREEN)

    cv2.setWindowProperty("Hands Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


    cap = cv2.VideoCapture(0)  # Change to filename for video input
    if (videoHeight): 
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, videoWidth)
    if (videoWidth): 
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, videoHeight)

    return cap

def Finish(cap):
    cap.release()
    cv2.destroyAllWindows()

def CalculateRelativeDistance(landmarks_normalized, mp_hands):
    rel1 = landmarks_normalized[mp_hands.HandLandmark.WRIST.value]
    rel2 = landmarks_normalized[mp_hands.HandLandmark.INDEX_FINGER_MCP.value]
    rel_distance = np.linalg.norm(rel1 - rel2)
    return rel_distance

def DetectRectTouch(index_finger_tip, rect, img_h, img_w):
    buffer = 50
    finger_x = int(index_finger_tip[0] * img_w)
    finger_y = int(index_finger_tip[1] * img_h)

    if (finger_x > rect.Get_X() - buffer and finger_x < rect.Get_X() + rect.Get_Size() + buffer):
        if (finger_y > rect.Get_Y() - buffer and finger_y < rect.Get_Y() + rect.Get_Size() + buffer):
            return True
    
    return False

def DetectCircleTouch(index_finger_tip, circle, img_h, img_w):
    buffer = 50
    finger_x = int(index_finger_tip[0] * img_w)
    finger_y = int(index_finger_tip[1] * img_h)

    distance = np.sqrt((finger_x - circle.Get_Center_X())**2 + (finger_y - circle.Get_Center_Y())**2)
    if (distance < circle.Get_Radius() + buffer):
        return True
    
    return False

def DetectButton(index_finger_tip, rect, img_h, img_w):
    buffer = 10
    finger_x = int(index_finger_tip[0] * img_w)
    finger_y = int(index_finger_tip[1] * img_h)

    if (finger_x > rect.Get_X() - buffer and finger_x < rect.Get_X() + rect.Get_Size() + buffer):
        if (finger_y > rect.Get_Y() - buffer and finger_y < rect.Get_Y() + rect.Get_Size() + buffer):
            return True
    
    return False

def DeleteObject(arr, index):
    obj = arr[index]
    arr.pop(index)
    del obj

def CreateObject(arr, type, x, y, size):
    if type == "rect":
        r = Rectangle(x, y, size)
        arr.append(r)

    elif type == "circle":
        c = Circle(x, y, size)
        arr.append(c)
