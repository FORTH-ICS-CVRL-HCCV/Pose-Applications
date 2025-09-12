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

square_x = 100
square_y = 100
square_size = 100
dis_x = 0
dis_y = 0
edit = False


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
    
    
    if results.multi_hand_landmarks:
        size = len(results.multi_hand_landmarks)
        count = 0
        for hands in results.multi_hand_landmarks:     
            mp_drawing.draw_landmarks(image, hands, mp_hands.HAND_CONNECTIONS)
            

            img_h, img_w, _ = image.shape
    

            for idx, landmark in enumerate(hands.landmark):
                x = int(landmark.x * img_w)
                y = int(landmark.y * img_h)
                #print(idx, " ", x, " ", y)
                cv2.putText(image, str(idx), (x, y),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                #if idx == 0:
                    #cv2.circle(image, (x, y), 20, (0, 0, 0), cv2.FILLED)

            
            landmarks_normalized = np.array([[landmark.x, landmark.y] for landmark in hands.landmark])
            index_finger_tip = landmarks_normalized[mp_hands.HandLandmark.INDEX_FINGER_TIP.value]
            thumb_tip = landmarks_normalized[mp_hands.HandLandmark.THUMB_TIP.value]
            distance = np.linalg.norm(index_finger_tip - thumb_tip)

            rel1 = landmarks_normalized[mp_hands.HandLandmark.WRIST.value]
            rel2 = landmarks_normalized[mp_hands.HandLandmark.INDEX_FINGER_MCP.value]
            rel_distance = np.linalg.norm(rel1 - rel2)
            
            if size == 1:
                if distance < (rel_distance/4):
                    cv2.putText(image, "Pinching hand 1", (8,70),  cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 1)
                    if (int(index_finger_tip[0] * img_w) > square_x - 10 and int(index_finger_tip[0] * img_w) < square_x + square_size + 10):
                        if (int(index_finger_tip[1] * img_h) > square_y - 10 and int(index_finger_tip[1] * img_h) < square_y + square_size + 10):
                            if edit == False:
                                dis_x = np.abs(square_x - int(index_finger_tip[0] * img_w))
                                dis_y = np.abs(square_y - int(index_finger_tip[1] * img_h))
                                edit = True
                            if(square_x > int(index_finger_tip[0] * img_w)):
                                square_x = int(index_finger_tip[0] * img_w) + dis_x
                            else:
                                square_x = int(index_finger_tip[0] * img_w) - dis_x

                            if(square_y > int(index_finger_tip[1] * img_h)):
                                square_y = int(index_finger_tip[1] * img_h) + dis_y
                            else:
                                square_y = int(index_finger_tip[1] * img_h) - dis_y
                            break
                        else:
                            edit = False
                    else:
                        edit = False
                else:
                    cv2.putText(image, "Not Pinching hand 1", (8,70),  cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 1)
            elif size == 2:
                if count == 0:
                    if distance < (rel_distance/4):
                        cv2.putText(image, "Pinching hand 1", (8,70),  cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 1)
                        if (int(index_finger_tip[0] * img_w) > square_x - 10 and int(index_finger_tip[0] * img_w) < square_x + square_size + 10):
                            if (int(index_finger_tip[1] * img_h) > square_y - 10 and int(index_finger_tip[1] * img_h) < square_y + square_size + 10):
                                if edit == False:
                                    dis_x = int(np.abs(square_x - index_finger_tip[0] * img_w))
                                    dis_y = int(np.abs(square_y - index_finger_tip[1] * img_h))
                                    edit = True
                                if(square_x > int(index_finger_tip[0] * img_w)):
                                    square_x = int(index_finger_tip[0] * img_w) + dis_x
                                else:
                                    square_x = int(index_finger_tip[0] * img_w) - dis_x

                                if(square_y > int(index_finger_tip[1] * img_h)):
                                    square_y = int(index_finger_tip[1] * img_h) + dis_y
                                else:
                                    square_y = int(index_finger_tip[1] * img_h) - dis_y
                                break
                            else:
                                edit = False
                        else:
                            edit = False
                    else:
                        cv2.putText(image, "Not Pinching hand 1", (8,70),  cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 1)
                    count += 1
                elif count == 1:
                    if distance < (rel_distance/4):
                        cv2.putText(image, "Pinching hand 2", (8,100),  cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 1)
                        if (int(index_finger_tip[0] * img_w) > square_x - 10 and int(index_finger_tip[0] * img_w) < square_x + square_size + 10):
                            if (int(index_finger_tip[1] * img_h) > square_y - 10 and int(index_finger_tip[1] * img_h) < square_y + square_size + 10):
                                if edit == False:
                                    dis_x = int(np.abs(square_x - (index_finger_tip[0] * img_w)))
                                    dis_y = int(np.abs(square_y - (index_finger_tip[1] * img_h)))
                                    edit = True
                                if(square_x > int(index_finger_tip[0] * img_w)):
                                    square_x = int(index_finger_tip[0] * img_w) + dis_x
                                else:
                                    square_x = int(index_finger_tip[0] * img_w) - dis_x

                                if(square_y > int(index_finger_tip[1] * img_h)):
                                    square_y = int(index_finger_tip[1] * img_h) + dis_y
                                else:
                                    square_y = int(index_finger_tip[1] * img_h) - dis_y
                                break
                            else:
                                edit = False
                        else:
                            edit = False
                    else:
                        cv2.putText(image, "Not Pinching hand 2", (8,100),  cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 1)
                    count -= 1
    seconds    = time.time() - start
    hz    = 1 / (seconds+0.0001)
    cv2.putText(image, "Framerate: %0.2f Hz" % hz, (8,40),  cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 1)
    cv2.rectangle(image, (square_x, square_y), (square_x + square_size, square_y + square_size), (0, 255, 0), -1)
    cv2.imshow('Hands Detection', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    #Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()