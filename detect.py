from ultralytics import YOLO
import cv2

model = YOLO("models/PlayingCards.pt")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    results = model(frame)
    
    boxes = results[0].boxes
    card_count = len(boxes)
    
    annotated = results[0].plot()  # draws boxes + labels automatically
    cv2.putText(annotated, f"Cards: {card_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Cards", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break