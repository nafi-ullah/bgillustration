import cv2
import numpy as np
import os
from ultralytics import YOLO
from my_logging_script import log_to_json
from io import BytesIO
from PIL import Image
wheel_model_path = 'models/wheels/wheeldetect3folder/wheel-3folder-best.pt'
headlight_model_path = 'models/headlights/v900/v900_best.pt'

current_file_name = "features/auto5_6angle/process_angle_5_6.py"

def wheel_coordinate_for_wall(image_bytes: BytesIO) -> dict:
    log_to_json("Got the file in wheel_headlight_process function", current_file_name)

    # Load the input image
    try:
        image_array = np.frombuffer(image_bytes.read(), np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Error: Unable to read the image from BytesIO.")

        original_height, original_width = img.shape[:2]

        # Load the YOLO model
        model = YOLO(wheel_model_path)
      

        # Run inference on the image
        results = model(img)
        if not results:
            log_to_json("No wheel found in car by the YOLO model.", current_file_name)
            return {}

        # Combine all masks into a single binary mask
        combined_mask = np.zeros((original_height, original_width), dtype=np.uint8)
        for result in results:
            if result.masks is None:
                log_to_json("No masks found for the detected object.", current_file_name)
                continue
            for mask in result.masks.data:
                mask = (mask.cpu().numpy() * 255).astype('uint8')
                mask_resized = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
                combined_mask = cv2.bitwise_or(combined_mask, mask_resized)

        # Find contours in the combined mask
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print("No segmented areas found.")
            return {}

        # Sort contours by area (largest to smallest)
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

        if len(sorted_contours) < 2:
            print("Less than two segmented areas found.")
            return {}

        # Get the second largest contour
        second_largest_contour = sorted_contours[1]

        # Find the topmost point of the second largest contour
        topmost_point = tuple(second_largest_contour[second_largest_contour[:, :, 1].argmin()][0])

        # Return the coordinates of the topmost point
        return {"top_x": topmost_point[0], "top_y": topmost_point[1]}

    except Exception as e:
        log_to_json(f"An error occurred: {e}", current_file_name)
        return {}