import argparse
import matplotlib.pyplot as plt
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 Object Detection on Hotdog Images")
    parser.add_argument('--image', type=str, required=True, help='Path to the image for inference')
    parser.add_argument('--output', type=str, default='inferences/inference_output.jpg', help='Path to save the inference output image')
    return parser.parse_args()

args = parse_args()

model = YOLO('weights/best.pt')  # Load your fine-tuned weights

img_path = args.image
inference_results = model(img_path) # Inference call

# Extract and print inference details
for result in inference_results:
    result.show()
    result.save(filename=args.output)
    print(f'Image shape: {result.orig_shape}')
    print(f'Predicted labels: {result.names}')
    print(f'Bounding boxes: {result.boxes.xyxy}')  # Bounding box coordinates
    print(f'Confidences: {result.boxes.conf}')     # Confidence scores
    print(f'Classes: {result.boxes.cls}')          # Class indices