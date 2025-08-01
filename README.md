# Video Object Detection with Retrained YOLOv5

This project takes the KITTI public database and the YOLOv5 open source model from ultralytics,
and retrain the model with the KITTI database after performing some manulpulation and preparation of the dataset.

#

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```


## Model Details

The model is trained on the KITTI dataset with the following classes:
- Car (Green)
- Van (Blue)
- Truck (Red)
- Pedestrian (Cyan)
- Person_sitting (Magenta)
- Cyclist (Yellow)
- Tram (Purple)
- Misc (Gray)
- DontCare (Dark Gray)

## Configuration

You can modify the following parameters in the video_inference.py script:
- `model_path`: Path to your trained model
- `input_video`: Path to your input video
- `output_video`: Path for the output video
- Confidence threshold (currently set to 0.3)

## Output

The script will create `output_detection.mp4` with:
- Bounding boxes around detected objects
- Class labels with confidence scores
- Color-coded boxes for different object classes 



