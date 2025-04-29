import cv2
import numpy as np
import mediapipe as mp
from collections import deque


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0)


canvas = np.zeros((480, 640, 3), dtype=np.uint8)
canvas.fill(255)
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
current_color = 0


points = [deque(maxlen=512) for _ in range(len(colors))]


prev_tip_x, prev_tip_y = None, None


drawing_enabled = True

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    frame = cv2.flip(frame, 1)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for idx, landmark in enumerate(hand_landmarks.landmark):

                    cx, cy = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 255), cv2.FILLED)
                    tip_x, tip_y = cx, cy
                    if drawing_enabled:
                        cv2.circle(canvas, (tip_x, tip_y), 5, colors[current_color], cv2.FILLED)
                        if prev_tip_x is not None and prev_tip_y is not None:
                            cv2.line(canvas, (prev_tip_x, prev_tip_y), (tip_x, tip_y), colors[current_color], 5)
                    prev_tip_x, prev_tip_y = tip_x, tip_y

    cv2.imshow('Camera', frame)
    cv2.imshow('Canvas', canvas)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        canvas.fill(255)
        points = [deque(maxlen=512) for _ in range(len(colors))]
        current_color = 0  
    elif key == ord('e'):  
        current_color = -1
    elif key == ord('d'):  
        drawing_enabled = not drawing_enabled
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
