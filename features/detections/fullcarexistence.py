import cv2
import numpy as np
from ultralytics import YOLO
from io import BytesIO
from my_logging_script import log_to_json

current_file_name = "features/detections/fullcarexistence.py"
# Define model paths
model_path_windshield = './models/frontwindshield/front-windshield-best.pt'
model_path_wheels = "./models/wheels/wheeldetect3folder/wheel-3folder-best.pt"

# Load YOLO models
model_windshield = YOLO(model_path_windshield)
model_wheels = YOLO(model_path_wheels)

def is_full_car(input_image_bytesio: BytesIO) -> bool:
    # Convert BytesIO to numpy array
    try:
        print("is_full_car: detecting full car or not")
        image_bytes = np.asarray(bytearray(input_image_bytesio.read()), dtype=np.uint8)
        img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            print("Error: Unable to decode image from bytes")
            return False
        
        # Detect windshield
        results_windshield = model_windshield(img)
        has_windshield = any(len(result.boxes) > 0 for result in results_windshield)
        print(f"windshield detected: {has_windshield}")
        
        # Detect wheels
        results_wheels = model_wheels(img)
        wheel_count = sum(len(result.boxes) for result in results_wheels)
        has_multiple_wheels = wheel_count > 1

        print(f"wheel count: {wheel_count}")
        
        # Return True if both conditions are met
        return has_windshield and has_multiple_wheels
    except Exception as e:
        log_to_json(f"is_full_car: Exceptional error {e}", current_file_name)
        return  None