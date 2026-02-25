# this needs to be updated or maybe webcam on laptop performs poorly
from ultralytics import YOLO
import cv2
from collections import deque, Counter

model = YOLO("models/PlayingCardsM.pt")

cap = cv2.VideoCapture(0)
history = deque(maxlen=10)
last_stable = []

while True:
    ret, frame = cap.read()
    results = model.track(frame, persist=True, conf=0.75)
    
    boxes = results[0].boxes
    detections = [model.names[int(c)] for c in boxes.cls]
    history.append(detections)
    # trying to make detection smoother but i think its webcam diff
    stable = [k for k, v in Counter(c for f in history for c in f).items() if v >= 6]
    
    if stable:
        last_stable = stable

    # draw boxes manually 
    annotated = frame.copy()
    for box, cls in zip(boxes.xyxy, boxes.cls):
        x1, y1, x2, y2 = map(int, box)
        label = model.names[int(cls)]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(annotated, f"Cards: {len(last_stable)}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(annotated, f"Detected: {', '.join(last_stable)}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2) # maybe get rid of

    cv2.imshow("Card Detector", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'): # q to quit 
        break

cap.release()
cv2.destroyAllWindows()