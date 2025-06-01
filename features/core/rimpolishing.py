from my_logging_script import log_to_json
import requests
current_file_name = "rimpolishing.py"
from io import BytesIO
from PIL import Image, ImageOps
import cv2
import numpy as np
from ultralytics import YOLO
import os

model_path = 'models/wheels/wheeldetect3folder/wheel-3folder-best.pt'

def increase_brightness(image, mask, factor=1.5):
    try:
    # Convert the image to float to avoid overflow during multiplication
        image = image.astype(np.float32)

        # Increase brightness in the masked regions
        for c in range(3):  # Loop over color channels (excluding alpha channel)
            image[:, :, c] = np.where(mask > 0, image[:, :, c] * factor, image[:, :, c])

        # Clip values to valid range (0-255) and convert back to uint8
        return np.clip(image, 0, 255).astype(np.uint8)
    except Exception as e:
        log_to_json(f"increase_brightness: Exceptional error {e}", current_file_name)
        return None


def shrink_mask(mask, pixels=40):
    try:
    # Create a kernel for the erosion operation
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pixels, pixels))

        # Perform erosion to shrink the mask inward
        shrunken_mask = cv2.erode(mask, kernel, iterations=1)

        return shrunken_mask
    except Exception as e:
        log_to_json(f"increase_brightness: Exceptional error {e}", current_file_name)
        return None


def smooth_mask_edges(mask, kernel_size=15):
    # Apply Gaussian blur to the mask to smooth edges
    try:
        blurred_mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)

        # Threshold to convert blurred mask back to binary
        smoothed_mask = np.where(blurred_mask > 0, 255, 0).astype(np.uint8)

        return smoothed_mask
    except Exception as e:
        log_to_json(f"Exceptional error {e}", current_file_name)
        return None

# def rimpolishingfunc(image_file):
#     log_to_json("Got the imagefile in rimpolishing function.", current_file_name)

    


  
#         # Load the transparent image from the response content
#     log_to_json(f"rim polished the image", current_file_name)
#         # print(f"Remove.bg done")
#     image = Image.open(BytesIO(image_file.content)).convert("RGBA")
#     return {"image": image, "status": "success"}


def rimpolishingfunc( input_image_bytes , transparent_image):
    # Read the image from BytesIO
    try:
        input_image_bytes.seek(0)
        file_bytes = np.asarray(bytearray(input_image_bytes.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            log_to_json(f"rimpolishingfunc: Unable to read full image.", current_file_name)
            return None
        
        transparent_image.seek(0)
        file_bytes_transparent = np.asarray(bytearray(transparent_image.read()), dtype=np.uint8)
        img_transparent = cv2.imdecode(file_bytes_transparent, cv2.IMREAD_UNCHANGED)
        if img_transparent is None:
            log_to_json(f"rimpolishingfunc: Unable to read transparent image.", current_file_name)
            return None

        original_height, original_width = img.shape[:2]

        # Load the YOLO model for segmentation
        model = YOLO(model_path)

        # Run inference on the image
        results = model(img)

        if not results:
            log_to_json("No wheels found in car by the YOLO model.", current_file_name)
            return None

        # Create a blank mask with the same dimensions as the original image
        combined_mask = np.zeros((original_height, original_width), dtype=np.uint8)

        # Iterate over the detected results to combine each mask
        for result in results:
            if result.masks is None:
                log_to_json("No masks found for the detected object.", current_file_name)
                continue
            for mask in result.masks.data:
                # Move the mask to CPU if needed and scale to 0-255
                mask = (mask.cpu().numpy() * 255).astype('uint8')

                # Resize the mask to match the original image dimensions
                mask_resized = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

                # Combine the resized mask with the blank mask using bitwise OR
                combined_mask = cv2.bitwise_or(combined_mask, mask_resized)

        # Shrink the combined mask inward by 150px
        shrunken_mask = shrink_mask(combined_mask, pixels=50)

        # Smooth the edges of the shrunken mask
        smoothed_mask = smooth_mask_edges(shrunken_mask)

        # Increase brightness in the shrunken segmented areas
        brightened_image = increase_brightness(img_transparent, smoothed_mask)

        # Encode the brightened image to bytes
        _, encoded_image = cv2.imencode('.png', brightened_image)
        output_image_bytes = BytesIO(encoded_image.tobytes())
        output_image_bytes.seek(0)

        return output_image_bytes
    except FileNotFoundError as e:
        log_to_json(f"rimpolishingfunc : File not found error {e}", current_file_name)
        return None
    except ValueError as e:
        log_to_json(f"rimpolishingfunc : ValueError {e}", current_file_name)
        return None
    except Exception as e:
        log_to_json(f"rimpolishingfunc: Exceptional error {e}", current_file_name)
        return None