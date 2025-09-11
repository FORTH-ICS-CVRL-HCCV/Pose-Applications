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
from hands import *
import json


def save(data):
    with open(".venv\Pose-Applications\square_app\data.json", "w") as f:
        json.dump(data, f, indent=2)

def read():
    with open(".venv\Pose-Applications\square_app\data.json", "r") as f:
        data = json.load(f)
    return data

def initObjects(rectangles, circles, buttons, hands):
    data = read()
    for d in data:
        if d["type"] == "rect":
            rect = Rectangle(d["x"], d["y"], 100, (0, 255, 0))
            rectangles.append(rect)
        elif d["type"] == "circle":
            circ = Circle(d["x"], d["y"], 50, (0, 255, 0))
            circles.append(circ)

    but1 = Button(50, 50, (0, 255, 0), 100, "Add")
    but2 = Button(200, 50, (255, 0, 0), 100, "Delete")
    but3 = Button(350, 50, (0, 0, 250), 100, "Edit")
    left_hand = Hand()
    right_hand = Hand()

    buttons.append(but1)
    buttons.append(but2)
    buttons.append(but3)
    hands.append(left_hand)
    hands.append(right_hand)

def SaveObjects(rectangles, circles):
    data = []
    for r in rectangles:
        add_object(data, "rect", int(r.Get_X()), int(r.Get_Y()))
    
    for c in circles:
        add_object(data, "circle", int(c.Get_Center_X()), int(c.Get_Center_Y()))

    save(data)
        

def add_object(data, type, x, y):
    data.append({"type": type, "x": x, "y": y})

