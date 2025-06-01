from io import BytesIO
import cv2
import numpy as np
from ultralytics import YOLO
from my_logging_script import log_to_json
model_path = "models/bottomChassis/bottomchassis_front/bottom-chassis-front-best.pt"
current_file_name = "carbottomcoordinates"

def find_car_bottom_points(image_bytes):
    try:
    # Load the input image from BytesIO
        image_bytes.seek(0)  # Ensure we're at the start of the stream
        file_bytes = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            print("Error: Unable to read image from BytesIO.")
            return None

        original_height, original_width = img.shape[:2]

        # Load the YOLO model
        model = YOLO(model_path)

        # Run inference on the image
        results = model(img)
        if not results:
            log_to_json("No car found in car by the YOLO model.", current_file_name)
            return None

        # Combine all masks into a single binary mask
        combined_mask = np.zeros((original_height, original_width), dtype=np.uint8)
        for result in results:
            if result.masks is None:
                log_to_json("No masks found for the detected object.", current_file_name)
                continue
            for mask in result.masks.data:
                # Move the mask to CPU if needed, scale to 0-255, and convert to uint8
                mask = (mask.cpu().numpy() * 255).astype('uint8')

                # Resize the mask to match the dimensions of the original image
                mask_resized = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

                # Combine the resized mask with the combined_mask using bitwise OR
                combined_mask = cv2.bitwise_or(combined_mask, mask_resized)

        # Find contours in the combined mask
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print("No segmented areas found.")
            return None

        # Find the largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)

        # Find extreme points of the largest contour
        bottom_left = tuple(largest_contour[np.argmin(largest_contour[:, 0, 0] - largest_contour[:, 0, 1])][0])
        bottom_right = tuple(largest_contour[np.argmax(largest_contour[:, 0, 0] + largest_contour[:, 0, 1])][0])

        # Return the bottom-left and bottom-right points
        return bottom_left, bottom_right
    except Exception as e:
        log_to_json(f"Error occurs for {e}", current_file_name)
        return None