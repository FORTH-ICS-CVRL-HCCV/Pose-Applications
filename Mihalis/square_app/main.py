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
from json_handling import *
from menu import *

def main():
    buffer = 0
    clock = Clock() #Initializing Clock
    cap = CameraSet() #Initianlizing Camera

    #Initializing screen objects and their arrays
    

    rectangles = []
    circles = []
    buttons = []
    hands = []

    initObjects(rectangles, circles, buttons, hands)
    menu = Menu(50, 200)

     
    #Initializing the detector
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    mp_drawing_styles = mp.solutions.drawing_styles
    hands_detector = mp_hands.Hands(static_image_mode=False, max_num_hands = 2, model_complexity=0, min_detection_confidence=0.5)

    while cap.isOpened():
        clock.start_clock()
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture video")
            break

        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(image)


        if results.multi_hand_landmarks:
            #We initialize the hands and we copy the attributes from the detected hands to the left and right hands
            for landmark, handed in zip(results.multi_hand_landmarks, results.multi_handedness):
                if(results.multi_hand_landmarks == None):
                    hands[0].isEmpty()
                    hands[1].isEmpty()
                elif(len(results.multi_hand_landmarks) == 1):
                    if handed.classification[0].label == "Right":
                        hands[1].setup(landmark, handed)
                        hands[0].isEmpty()
                    elif handed.classification[0].label == "Left":
                        hands[0].setup(landmark, handed)
                        hands[1].isEmpty()
                elif(len(results.multi_hand_landmarks) == 2):
                    if handed.classification[0].label == "Right":
                        hands[1].setup(landmark, handed)
                    elif handed.classification[0].label == "Left":
                        hands[0].setup(landmark, handed)
  
            for h in hands:   
                hand = h.Get_Landmark()
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS)
                
                img_h, img_w, _ = image.shape

                distance = 0
                rel_distance = 0
                finger_position = 0

                #We initialize the distance and relative distance to use for pinching Detection
                if(h.isActive()):
                    landmarks_normalized = np.array([[landmark.x, landmark.y] for landmark in hand.landmark])
                    index_finger_tip = landmarks_normalized[mp_hands.HandLandmark.INDEX_FINGER_TIP.value]
                    thumb_tip = landmarks_normalized[mp_hands.HandLandmark.THUMB_TIP.value]
                    distance = np.linalg.norm(index_finger_tip - thumb_tip)
                    rel_distance = CalculateRelativeDistance(landmarks_normalized, mp_hands)
                

                if h.checkPinch(distance, rel_distance):
                    finger_position = h.GetPosition(index_finger_tip)
                    if menu.Get_Status() == False:
                        if h.Get_Editing() == False:
                            #If we are not editing something we search if we are touching something
                            for i in range(0, len(rectangles)):
                                if (rectangles[i].DetectRectTouch(finger_position, img_h, img_w)):
                                    #rectangles[i].Move(finger_position, img_w, img_h)
                                    h.Set_Last_Pinched_Type(rectangles[i].Get_Type())
                                    h.Set_Last_Pinched_Index(i)
                                    h.Set_Type("rect")
                                    h.Set_Index(i)
                                    h.Set_Editing(True)
                                        
                                    break
                            for i in range(0, len(circles)):
                                if (circles[i].DetectCircleTouch(finger_position, img_h, img_w)):
                                    #circles[i].Move(finger_position, img_w, img_h)
                                    h.Set_Last_Pinched_Type(circles[i].Get_Type())
                                    h.Set_Last_Pinched_Index(i)
                                    h.Set_Type("circle")
                                    h.Set_Index(i)
                                    h.Set_Editing(True)
                                        
                                    break
                            for i in range(0, len(buttons)):
                                if (buttons[i].DetectButtonTouch(finger_position, img_h, img_w)):
                                    if(buttons[i].Get_Text() == "Add" and buffer == 0):
                                        menu.Set_Status(True)
                                        menu.Set_Edit("Add")
                                        buffer = 50
                                    elif(buttons[i].Get_Text() == "Delete"):
                                        if h.Get_Last_Pinched_Type() == "rect":
                                            DeleteObject(rectangles, h.Get_Last_Pinched_Index())
                                            h.Set_Last_Pinched_Type("")
                                            h.Set_Last_Pinched_Index(0)
                                        elif h.Get_Last_Pinched_Type() == "circle":
                                            DeleteObject(circles, h.Get_Last_Pinched_Index())
                                            h.Set_Last_Pinched_Type("")
                                            h.Set_Last_Pinched_Index(0)
                                    elif(buttons[i].Get_Text() == "Edit"):
                                        if h.Get_Last_Pinched_Type() == "rect":
                                            obj = rectangles[h.Get_Last_Pinched_Index()]
                                            menu.Start_Edit(obj.Get_Size(), obj.Get_Color(), "rect")
                                        elif h.Get_Last_Pinched_Type() == "circle":
                                            obj = circle[h.Get_Last_Pinched_Index()]
                                            menu.Start_Edit(obj.Get_Size(), obj.Get_Color(), "circle")
                                        menu.Set_Status(True)
                                        menu.Set_Edit("Edit")
                                    break
                        else:
                            #if we were editing something we continue moving it
                            if h.Get_Type() == "rect":
                                rectangles[h.Get_Index()].Move(finger_position, img_w, img_h)
                            elif h.Get_Type() == "circle":
                                circles[h.Get_Index()].Move(finger_position, img_w, img_h)
                    else:
                        if h.Get_Editing() == True:
                            if h.Get_Type() == "rect":
                                rectangles[h.Get_Index()].Set_Edit(False)
                            elif h.Get_Type() == "circle":
                                circles[h.Get_Index()].Set_Edit(False)
                        menu.Touch(finger_position, img_w, img_h, rectangles, circles, h.Get_Last_Pinched_Index(), h.Get_Last_Pinched_Type())
                        h.ClearPos()
                else:
                    #If we are no longer pinching and we were editing something we reset
                    if h.Get_Editing() == True:
                        if h.Get_Type() == "rect":
                            rectangles[h.Get_Index()].Set_Edit(False)
                        elif h.Get_Type() == "circle":
                            circles[h.Get_Index()].Set_Edit(False)
                        h.ClearPos()
        clock.end_clock()
        hz = clock.result()
        draw(image, hz, rectangles, circles, buttons, menu, hands[0].Get_Last_Pinched_Type(), hands[1].Get_Last_Pinched_Type())
        cv2.imshow("Hands Detection", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        #Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break                

        

        if (buffer > 0):
            buffer -= 1
    Finish(cap)                
    SaveObjects(rectangles, circles)

#Run

main()