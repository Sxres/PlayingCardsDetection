from ultralytics import YOLO
import cv2
from collections import deque, Counter
import torch

model = YOLO("models/PlayingCardsM.pt")

# Use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using: {device}")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

history = deque(maxlen=10)
last_stable = []
last_boxes = []
last_cls = []
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Only run YOLO every 3rd frame
    if frame_count % 3 == 0:
        results = model.track(frame, persist=True, conf=0.75, verbose=False, 
                              imgsz=320, device=device)
        boxes = results[0].boxes
        last_boxes = boxes.xyxy
        last_cls = boxes.cls

        detections = [model.names[int(c)] for c in last_cls]
        history.append(detections)
        stable = [k for k, v in Counter(c for f in history for c in f).items() if v >= 6]
        if stable:
            last_stable = stable

    # Draw using cached boxes from last YOLO run
    annotated = frame.copy()
    for box, cls in zip(last_boxes, last_cls):
        x1, y1, x2, y2 = map(int, box)
        label = model.names[int(cls)]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(annotated, f"Cards: {len(last_stable)}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(annotated, f"Detected: {', '.join(last_stable)}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Card Detector", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()