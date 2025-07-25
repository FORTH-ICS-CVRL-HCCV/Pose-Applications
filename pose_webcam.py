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
mp_pose = mp.solutions.pose
mp_drawing_styles = mp.solutions.drawing_styles
pose_detector = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=True, min_detection_confidence=0.5)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture video")
        break
    
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    start      = time.time()
    results = pose_detector.process(image)
    seconds    = time.time() - start
    hz    = 1 / (seconds+0.0001)
    
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    cv2.putText(image, "Framerate: %0.2f Hz" % hz, (8,40),  cv2.FONT_HERSHEY_SIMPLEX, 1.0, (123, 123, 123), 1)
    cv2.imshow('Pose Detection', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    #Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




