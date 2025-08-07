#Dependencies : 
#pip install opencv-python mediapipe numpy matplotlib

import cv2
import time
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

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

class Rectangle:
    def __init__(self, x, y, size):
        self.x = x
        self.y = y
        self.size = size
        self.edit = False

    def Get_X(self):
        return self.x

    def Get_Y(self):
        return self.y

    def Get_Size(self):
        return self.size

    def Get_Edit(self):
        return self.edit

    def Set_X(self, new_x):
        self.x = new_x

    def Set_Y(self, new_y):
        self.y = new_y

    def Set_Size(self, new_size):
        self.size = new_size

    def Set_Edit(self, new_edit):
        self.size = new_edit

    def Move(self, index_finger_tip, img_w, img_h):
        finger_x = int(index_finger_tip[0] * img_w)
        finger_y = int(index_finger_tip[1] * img_h)


        if self.edit == False:
            dis_x = np.abs(self.x - finger_x)
            dis_y = np.abs(self.y - finger_y)
            self.edit = True
            if(self.x > finger_x):
                self.x = finger_x + dis_x
            else:
                self.x = finger_x - dis_x

            if(self.y > finger_y):
                self.y = finger_y + dis_y
            else:
                self.y = finger_y - dis_y
            
            return True
        return False




def CameraSet():
    #Run from webcam
    videoWidth=None # Or 640 
    videoHeight=None  # Or 480

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

def DetectTouch(index_finger_tip, rect, img_h, img_w):
    buffer = 10
    finger_x = int(index_finger_tip[0] * img_w)
    finger_y = int(index_finger_tip[1] * img_h)

    if (finger_x > rect.Get_X() - buffer and finger_x < rect.Get_X() + rect.Get_Size() + buffer):
        if (finger_y > rect.Get_X() - buffer and finger_y < rect.Get_Y() + rect.Get_Size() + buffer):
            return True
    
    return False

def main():
    clock = Clock()
    cap = CameraSet()

    rect = Rectangle(100, 100, 100)

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
                        cv2.putText(image, "Pinching hand 1", (8,70),  cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 1)
                        if (DetectTouch(index_finger_tip, rect, img_h, img_w)):
                            result = rect.Move(index_finger_tip, img_w, img_h)
                            
                            if(result == True):
                                break

                        else:
                            rect.Set_Edit(False)
                    else:
                        cv2.putText(image, "Not Pinching hand 1", (8,70),  cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 1)
                elif(size == 2):
                    if count == 0:
                        if distance < (rel_distance/4):
                            cv2.putText(image, "Pinching hand 1", (8,70),  cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 1)
                            if (DetectTouch(index_finger_tip, rect, img_h, img_w)):
                                result = rect.Move(index_finger_tip, img_w, img_h)
                            
                                if(result == True):
                                    break

                            else:
                                rect.Set_Edit(False)
                        else:
                            cv2.putText(image, "Not Pinching hand 1", (8,100),  cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 1)

                        count += 1
                    elif count == 1: 
                        if distance < (rel_distance/4):
                            cv2.putText(image, "Pinching hand 2", (8,70),  cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 1)
                            if (DetectTouch(index_finger_tip, rect, img_h, img_w)):
                                result = rect.Move(index_finger_tip, img_w, img_h)
                            
                                if(result == True):
                                    break

                            else:
                                rect.Set_Edit(False)
                        else:
                            cv2.putText(image, "Not Pinching hand 2", (8,100),  cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 1)

                        count -= 1
        clock.end_clock()
        hz = clock.result()
        cv2.putText(image, "Framerate: %0.2f Hz" % hz, (8,40),  cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 1)
        cv2.rectangle(image, (rect.Get_X(), rect.Get_Y()), (rect.Get_X() + rect.Get_Size(), rect.Get_Y() + rect.Get_Size()), (0, 255, 0), -1)
        cv2.imshow('Hands Detection', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        #Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break                

    Finish(cap)                


#Run

main()
