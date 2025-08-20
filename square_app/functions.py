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

def DeleteObject(arr, index):
    obj = arr[index]
    arr.pop(index)
    del obj

def CreateObject(arr, type, x, y, size, color):
    if type == "rect":
        r = Rectangle(x, y, size, color)
        arr.append(r)

    elif type == "circle":
        c = Circle(x, y, size, color)
        arr.append(c)
