import cv2
import torch
import numpy as np
from ultralytics import YOLO
import os

def run_video_inference():
    """
    Run YOLOv5 inference on a video file and save the results with bounding boxes and labels.
    """
    
    # Configuration
    model_path = "training_results/model.pt"
    input_video = "italy-time-lapse-car-on-the-street.mov"
    output_video = "output_detection.mp4"
    
    # Class names for KITTI dataset
    class_names = ["Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram", "Misc", "DontCare"]
    
    # Colors for different classes (BGR format for OpenCV)
    colors = [
        (0, 255, 0),    # Green for Car
        (255, 0, 0),    # Blue for Van
        (0, 0, 255),    # Red for Truck
        (255, 255, 0),  # Cyan for Pedestrian
        (255, 0, 255),  # Magenta for Person_sitting
        (0, 255, 255),  # Yellow for Cyclist
        (128, 0, 128),  # Purple for Tram
        (128, 128, 128), # Gray for Misc
        (64, 64, 64)    # Dark gray for DontCare
    ]
    
    # Drawing parameters
    box_thickness = 4      # Increased thickness for bounding boxes
    font_scale = 1.0       # Increased font scale for larger text
    font_thickness = 3     # Increased thickness for text
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    # Check if input video exists
    if not os.path.exists(input_video):
        print(f"Error: Input video not found at {input_video}")
        return
    
    print("Loading YOLOv5 model...")
    try:
        # Load the model
        model = YOLO(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Open the input video
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_video}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties:")
    print(f"  FPS: {fps}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Total frames: {total_frames}")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"Error: Could not create output video file {output_video}")
        cap.release()
        return
    
    print("Starting video processing...")
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % 30 == 0:  # Print progress every 30 frames
            progress = (frame_count / total_frames) * 100
            print(f"Processing frame {frame_count}/{total_frames} ({progress:.1f}%)")
        
        # Run inference
        results = model(frame, verbose=False)
        
        # Process results
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Get class and confidence
                    cls = int(box.cls[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())
                    
                    # Skip if confidence is too low
                    if conf < 0.3:
                        continue
                    
                    # Get class name and color
                    class_name = class_names[cls] if cls < len(class_names) else f"Class_{cls}"
                    color = colors[cls] if cls < len(colors) else (255, 255, 255)
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)
                    
                    # Create label text
                    label = f"{class_name}: {conf:.2f}"
                    
                    # Get label size
                    (label_width, label_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
                    )
                    
                    # Draw label background
                    cv2.rectangle(
                        frame,
                        (x1, y1 - label_height - baseline - 5),
                        (x1 + label_width, y1),
                        color,
                        -1
                    )
                    
                    # Draw label text
                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - baseline - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (255, 255, 255),
                        font_thickness
                    )
        
        # Write frame to output video
        out.write(frame)
    
    # Clean up
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Video processing completed!")
    print(f"Output saved to: {output_video}")

if __name__ == "__main__":
    run_video_inference() 