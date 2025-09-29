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

#Save the data at the data.json file
def save(data):
    with open("Pose-Applications//Mihalis//square_app//data.json", "w") as f:
        json.dump(data, f, indent=2)

#read the data from the data.json file
def read():
    try:
        with open("Pose-Applications//Mihalis//square_app//data.json", "r") as f:
            data = json.load(f)
    except:
        with open("Pose-Applications//Mihalis//square_app//data.json", "w") as f:
            data = []
    return data

#Create the objects and initialize the arrays
def initObjects(rectangles, circles, buttons, hands):
    data = read()
    for d in data:
        if d["type"] == "rect":
            rect = Rectangle(d["x"], d["y"], d["size"], d["color"])
            rectangles.append(rect)
        elif d["type"] == "circle":
            circ = Circle(d["x"], d["y"], d["size"], d["color"])
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


#Create the json string that will be saved
def SaveObjects(rectangles, circles):
    data = []
    for r in rectangles:
        add_object(data, "rect", int(r.Get_X()), int(r.Get_Y()), r.Get_Size(), r.Get_Color())
    
    for c in circles:
        add_object(data, "circle", int(c.Get_Center_X()), int(c.Get_Center_Y()), c.Get_Radius(), c.Get_Color())

    save(data)
        
#Create the json string about each object
def add_object(data, type, x, y, size, color):
    data.append({"type": type, "x": x, "y": y, "size": size, "color": color})

