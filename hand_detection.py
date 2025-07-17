#Dependencies : 
#pip install opencv-python mediapipe numpy matplotlib

import cv2
import time
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

#Run from webcam
videoWidth=None # Or 640 
videoHeight=None  # Or 480

cap = cv2.VideoCapture(0)  # Change to filename for video input
if (videoHeight): 
   cap.set(cv2.CAP_PROP_FRAME_WIDTH, videoWidth)
if (videoWidth): 
   cap.set(cv2.CAP_PROP_FRAME_HEIGHT, videoHeight)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles
hands_detector = mp_hands.Hands(static_image_mode=False, max_num_hands = 2, model_complexity=1, min_detection_confidence=0.5)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture video")
        break
    
    frame = cv2.flip(frame, 1)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    start      = time.time()
    results = hands_detector.process(image)
    seconds    = time.time() - start
    hz    = 1 / (seconds+0.0001)
    
    if results.multi_hand_landmarks:
        for hands in results.multi_hand_landmarks:     
            mp_drawing.draw_landmarks(image, hands, mp_hands.HAND_CONNECTIONS)
            

            img_h, img_w, _ = image.shape
    

            #Printing the coordinates of the landmarks and the name of the landmarks
            for idx, landmark in enumerate(hands.landmark):
                x = int(landmark.x * img_w)
                y = int(landmark.y * img_h)
                print(idx, " ", x, " ", y)
                cv2.putText(image, str(idx), (x, y),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                #if idx == 0:
                    #cv2.circle(image, (x, y), 20, (0, 0, 0), cv2.FILLED)
    
    cv2.putText(image, "Framerate: %0.2f Hz" % hz, (8,40),  cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 1)
    cv2.imshow('Hands Detection', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    #Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




