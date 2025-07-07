#Dependencies : 
#pip install opencv-python mediapipe numpy matplotlib

import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
#Run from webcam
cap = cv2.VideoCapture(0)  # Change to filename for video input
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_drawing_styles = mp.solutions.drawing_styles
pose_detector = mp_pose.Pose(static_image_mode=False, model_complexity=2, enable_segmentation=True, min_detection_confidence=0.5)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture video")
        break
    
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    center_x = int(w / 2)      
    

    frame1 = frame[:, 0:center_x]
    frame2 = frame[:, center_x:w]

    image1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

    results1 = pose_detector.process(image1)
    results2 = pose_detector.process(image2)
    
    if results1.pose_landmarks:
        mp_drawing.draw_landmarks(image1, results1.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    
    
    if results2.pose_landmarks:
        mp_drawing.draw_landmarks(image2, results2.pose_landmarks, mp_pose.POSE_CONNECTIONS)


    cv2.imshow('Pose Detection1', cv2.cvtColor(image1, cv2.COLOR_RGB2BGR))
    cv2.imshow('Pose Detection2', cv2.cvtColor(image2, cv2.COLOR_RGB2BGR))
    
    #Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()




