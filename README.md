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

## Output

The training.ipynb will download the public dataset and perform munipilation to prepare the dataset so the pretrained yolo5 can accept it. 

I trained only using 1 epoh for demonstration on already seen improving results: 

<img width="1814" height="142" alt="image" src="https://github.com/user-attachments/assets/edefd50b-3659-40d6-a58b-a21a85908be3" />

the model.pt is saved in the training retults for reference.


Tht script video_inference.py will create `output_detection.mp4` on a video of chooosing with:
- Bounding boxes around detected objects
- Class labels with confidence scores
- Color-coded boxes for different object classes 
