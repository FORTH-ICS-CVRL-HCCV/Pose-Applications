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

def main():
    buffer = 0
    clock = Clock()
    cap = CameraSet()

    rect = Rectangle(100, 100, 100)
    circ = Circle(300, 300, 50)
    but1 = Button(50, 50, (0, 255, 0), 100, "Add")
    but2 = Button(200, 50, (255, 0, 0), 100, "Delete")

    rectangles = []
    circles = []
    buttons = []

    last_pinched_type = ""
    last_pinched_index = 0

    rectangles.append(rect)
    circles.append(circ)
    buttons.append(but1)
    buttons.append(but2)
    

     

    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    mp_drawing_styles = mp.solutions.drawing_styles
    hands_detector = mp_hands.Hands(static_image_mode=False, max_num_hands = 2, model_complexity=1, min_detection_confidence=0.5)

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
            size = len(results.multi_hand_landmarks)
            count = 0
            for hands in results.multi_hand_landmarks:     
                mp_drawing.draw_landmarks(image, hands, mp_hands.HAND_CONNECTIONS)
                

                img_h, img_w, _ = image.shape
        

                for idx, landmark in enumerate(hands.landmark):
                    x = int(landmark.x * img_w)
                    y = int(landmark.y * img_h)
                    cv2.putText(image, str(idx), (x, y),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                landmarks_normalized = np.array([[landmark.x, landmark.y] for landmark in hands.landmark])
                index_finger_tip = landmarks_normalized[mp_hands.HandLandmark.INDEX_FINGER_TIP.value]
                thumb_tip = landmarks_normalized[mp_hands.HandLandmark.THUMB_TIP.value]
                distance = np.linalg.norm(index_finger_tip - thumb_tip)

                rel_distance = CalculateRelativeDistance(landmarks_normalized, mp_hands)

                if(size == 1):
                    if distance < (rel_distance/4):
                        for i in range(0, len(rectangles)): 
                            if (DetectRectTouch(index_finger_tip, rectangles[i], img_h, img_w)):
                                rectangles[i].Move(index_finger_tip, img_w, img_h)
                                last_pinched_type = rectangles[i].Get_Type()
                                last_pinched_index = i
                                
                                break

                            else:
                                rectangles[i].Set_Edit(False)
                        for i in range(0, len(circles)): 
                            if (DetectCircleTouch(index_finger_tip, circles[i], img_h, img_w)):
                                circles[i].Move(index_finger_tip, img_w, img_h)
                                last_pinched_type = circles[i].Get_Type()
                                last_pinched_index = i
                                
                                break

                            else:
                                circles[i].Set_Edit(False)
                        for i in range(0, len(buttons)):
                            if (DetectButton(index_finger_tip, buttons[i], img_h, img_w)):
                                if(buttons[i].Get_Text() == "Add" and buffer == 0):
                                    if(random.randint(0, 1) == 0):
                                        CreateObject(rectangles, "rect", 300, 300, 100)
                                    else:
                                        CreateObject(circles, "circle", 300, 300, 50)
                                    buffer = 50
                                elif(buttons[i].Get_Text() == "Delete"):
                                    if last_pinched_type == "rectangle":
                                        DeleteObject(rectangles, last_pinched_index)
                                        last_pinched_index = 0
                                        last_pinched_type = ""
                                    elif last_pinched_type == "circle":
                                        DeleteObject(circles, last_pinched_index)
                                        last_pinched_index = 0
                                        last_pinched_type = ""
                                    
                                break
                    else:
                        #cv2.putText(image, "Not Pinching hand 1", (8,70),  cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 1)
                        continue
                elif(size == 2):
                    if count == 0:
                        if distance < (rel_distance/4):
                            for i in range(0, len(rectangles)):
                                if (DetectRectTouch(index_finger_tip, rectangles[i], img_h, img_w)):
                                    rectangles[i].Move(index_finger_tip, img_w, img_h)
                                    last_pinched_type = rectangles[i].Get_Type()
                                    last_pinched_index = i
                        
                                    break

                                else:
                                    rectangles[i].Set_Edit(False)
                            for i in range(0, len(circles)): 
                                if (DetectCircleTouch(index_finger_tip, circles[i], img_h, img_w)):
                                    circles[i].Move(index_finger_tip, img_w, img_h)
                                    last_pinched_type = circles[i].Get_Type()
                                    last_pinched_index = i
                                    
                                    break

                                else:
                                    circles[i].Set_Edit(False)
                            for i in range(0, len(buttons)):
                                if (DetectButton(index_finger_tip, buttons[i], img_h, img_w)):
                                    if(buttons[i].Get_Text() == "Add" and buffer == 0):
                                        if(random.randint(0, 1) == 0):
                                            CreateObject(rectangles, "rect", 300, 300, 100)
                                        else:
                                            CreateObject(circles, "circle", 300, 300, 50)
                                        buffer = 50
                                    elif(buttons[i].Get_Text() == "Delete"):
                                        if last_pinched_type == "rectangle":
                                            DeleteObject(rectangles, last_pinched_index)
                                            last_pinched_index = 0
                                            last_pinched_type = ""
                                        elif last_pinched_type == "circle":
                                            DeleteObject(circles, last_pinched_index)
                                            last_pinched_index = 0
                                            last_pinched_type = ""
                                    break   
                            
                        else:
                            continue

                        count += 1
                    elif count == 1: 
                        if distance < (rel_distance/4):
                            
                            for i in range(0, len(rectangles)):
                                if (DetectRectTouch(index_finger_tip, rectangles[i], img_h, img_w)):
                                    rectangles[i].Move(index_finger_tip, img_w, img_h)
                                    last_pinched_type = rectangles[i].Get_Type()
                                    last_pinched_index = i
                        
                                    break

                                else:
                                    rectangles[i].Set_Edit(False)
                            for i in range(0, len(circles)): 
                                if (DetectCircleTouch(index_finger_tip, circles[i], img_h, img_w)):
                                    circles[i].Move(index_finger_tip, img_w, img_h)
                                    last_pinched_type = circles[i].Get_Type()
                                    last_pinched_index = i
                                    
                                    break

                                else:
                                    circles[i].Set_Edit(False)
                            for i in range(0, len(buttons)):
                                if (DetectButton(index_finger_tip, buttons[i], img_h, img_w)):
                                    if(buttons[i].Get_Text() == "Add" and buffer == 0):
                                        if(random.randint(0, 1) == 0):
                                            CreateObject(rectangles, "rect", 300, 300, 100)
                                        else:
                                            CreateObject(circles, "circle", 300, 300, 50)
                                        buffer = 50
                                    elif(buttons[i].Get_Text() == "Delete"):
                                        if last_pinched_type == "rectangle":
                                            DeleteObject(rectangles, last_pinched_index)
                                            last_pinched_index = 0
                                            last_pinched_type = ""
                                        elif last_pinched_type == "circle":
                                            DeleteObject(circles, last_pinched_index)
                                            last_pinched_index = 0
                                            last_pinched_type = ""
                                    break
                            
                        else:
                            
                            continue

                        count -= 1
        clock.end_clock()
        hz = clock.result()
        cv2.putText(image, "Framerate: %0.2f Hz" % hz, (500,40),  cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 1)
        cv2.putText(image, "Selected: " + last_pinched_type, (500, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 1)
        for shape in rectangles:
            shape.Draw(image)

        for shape in circles:
            shape.Draw(image)

        for shape in buttons:
            shape.Draw(image)
        cv2.imshow("Hands Detection", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        #Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break                

        if (buffer > 0):
            buffer -= 1
    Finish(cap)                


#Run

main()