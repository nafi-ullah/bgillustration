from ultralytics import YOLO
import cv2
import os
from my_logging_script import log_to_json
current_file_name = "features/core/licenseplateoperations/blurfunc.py"

def blur_license_plate(img_path, output_path=None):
    # Check if the image path exists and is a file
    # img_path = './results/output-28-08-2024-0233AM.png'
    try:
        if not os.path.isfile(img_path):
            log_to_json(f"{img_path} does not exist or is not a valid file.", current_file_name)
            return img_path

        # Load the YOLOv8 model
        model = YOLO('./models/licenseplate/initialJune/initialJune.pt')

        # Load the image from file
        img = cv2.imread(img_path)

        if img is None:
            raise ValueError(f"Image at {img_path} could not be loaded. Please check the file format and content.")

        # Perform inference
        results = model.predict(source=img, save=False)  # Pass the image array directly to predict
        if not results:
            log_to_json("No licenseplate found in car by the YOLO model.", current_file_name)
            return img_path

        # Iterate over the detected results
        for result in results:
            if result.masks is None:
                log_to_json("No masks found for the detected object.", current_file_name)
                continue
            boxes = result.boxes  # Get the bounding boxes

            # Process each detected bounding box
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]  # Extract coordinates
                
                # Ensure coordinates are integers
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Extract the region of interest (ROI) corresponding to the license plate
                roi = img[y1:y2, x1:x2]

                # Apply a Gaussian blur to the ROI
                roi_blurred = cv2.GaussianBlur(roi, (51, 51), 0)

                # Replace the original ROI with the blurred one
                img[y1:y2, x1:x2] = roi_blurred

        # Save the output image
        cv2.imwrite(img_path, img)
        log_to_json(f"License plate area blurred and saved as {img_path}")
    except Exception as e:
        log_to_json(f"License plate area not blurred for: {e}")
        return img_path
    


