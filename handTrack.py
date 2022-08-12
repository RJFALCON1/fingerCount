import mediapipe as mp
import cv2
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.8,
                       min_tracking_confidence=0.5)
                
def drawHandmarks(img,handPositions) :
    if handPositions:
        for x in handPositions:
            mp_drawing.draw_landmarks(img,x,mp_hands.HAND_CONNECTIONS)

tipID = [4,8,12,16,20]

def countFingers(img,handPositions,handNumber = 0):
    if handPositions:
        landmarks = handPositions[handNumber].landmark
        fingers = []
        for x in tipID :
            finger_tip_y = landmarks[x].y
            finger_bottom_y = landmarks[x-2].y
            thumbTip_x = landmarks[x].x
            thumbTip_bottom_x = landmarks[x-2].x
            if(x != 4) :
                if (finger_tip_y < finger_bottom_y) :
                    fingers.append(1)
                    print('open finger')
                if (finger_tip_y > finger_bottom_y) :
                    fingers.append(0)
                    print('Closed finger')
            elif(x==4):
                if (thumbTip_x >= thumbTip_bottom_x) :
                    fingers.append(1)
                if (thumbTip_x < thumbTip_bottom_x) :
                    fingers.append(0)

        fingerCount = fingers.count(1)
        text = f'fingers: {fingerCount}'
        cv2.putText(img,text,(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(125,125,0),2)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    allHands = hands.process(img)
    handPositions = allHands.multi_hand_landmarks
    drawHandmarks(img,handPositions)
    countFingers(img, handPositions)
    cv2.imshow('hand',img)
    if (cv2.waitKey(25)==32) :
        break
cv2.destroyAllWindows()