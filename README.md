# Playing Card Detector

Real-time playing card detection using YOLOv8 and OpenCV. Detects card rank and suit, counts cards on screen, and displays live bounding boxes via webcam or video feed.

## Demo
<img width="597" height="812" alt="output" src="https://github.com/user-attachments/assets/aa051844-ad22-4c49-87ec-506ecbe18100" />



## Features
- Detects all 52 playing cards (rank + suit)
- Real-time card counting
- Works with webcam or video file
- Bounding boxes with confidence scores

## Requirements
- Python 3.8+
- Ultralytics YOLOv26
- OpenCV

## Installation
```bash
git clone https://github.com/Sxres/PlayingCardsDetection
cd PlayingCardsDetection
uv sync 
```

## Usage

**Webcam:**
```bash
python detect.py
```

**Video file:**
```bash
python detect.py --source myvideo.mp4
```

## Model
Pre-trained model from [link to original repo]. Fine-tuned on custom card data.

Place the model file at `models/cards.pt`.

## Project Structure
```
card-detector/
├── detect.py
├── models/
│   └── cards.pt
├── requirements.txt
└── README.md
```

## Credits
- YOLOv8 by [Ultralytics](https://github.com/ultralytics/ultralytics)
- Base model from [original repo]
