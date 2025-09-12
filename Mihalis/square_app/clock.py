import cv2
import time
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import random

'''
The Clock class is used to count the framerate
    start_clock => records the start time
    end_clock => records the end time
    result => calculates framerate
'''

class Clock:
    def __init__(self):
        self.start = 0
        self.end = 0
        self.seconds = 0
    

    def start_clock(self):
        self.start = time.time()

    def end_clock(self):
        self.end = time.time()

    def result(self):
        return 1 / ((self.end-self.start)+0.0001)