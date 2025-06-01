import cv2
import numpy as np
from ultralytics import YOLO
from io import BytesIO
from my_logging_script import log_to_json
from PIL import Image
model_path = 'models/full-car-1800images/full-car-1800images-best.pt'

current_file_name = "features/detections/fullcarcoordinates.py"

def get_car_4_coordinates( image_bytes):
    try:
        # Read the image from BytesIO
        file_bytes = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            print("Error: Unable to decode image from bytes.")
            return None

        original_height, original_width = img.shape[:2]

        # Load the YOLO model for segmentation
        try:
            # Fix for PyTorch 2.6+ issue: Load with weights_only=False
            model = YOLO(model_path)
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            return None

        # Run inference on the image
        results = model(img)
        if not results:
            log_to_json("get_car_4_coordinates-- No car found  by the YOLO model.", current_file_name)
            return None

        largest_area = 0
        largest_car_coords = None

        # Iterate over detected objects to find the largest car
        for result in results:
            if result.masks is None:
                log_to_json("get_car_4_coordinates-- No car found for the detected object.", current_file_name)
                continue
            for mask in result.masks.data:
                # Move the mask to CPU and convert it to a binary mask
                mask = (mask.cpu().numpy() * 255).astype('uint8')

                # Resize mask to match the original image dimensions
                mask_resized = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

                # Calculate the area of the current mask
                area = cv2.countNonZero(mask_resized)

                if area > largest_area:
                    largest_area = area
                    # Get the bounding box of the largest segmented area
                    contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        # Find the largest contour
                        largest_contour = max(contours, key=cv2.contourArea)
                        x, y, w, h = cv2.boundingRect(largest_contour)

                        # Extract coordinates
                        top_most = (x + w // 2, y)
                        bottom_most = (x + w // 2, y + h)
                        left_most = (x, y + h // 2)
                        right_most = (x + w, y + h // 2)


                        is_success, buffer = cv2.imencode('.jpg', img)
                        if not is_success:
                            log_to_json("Failed to encode image into BytesIO format.",current_file_name)

                        image_bytes_io = BytesIO(buffer)
                        image_bytes_io.seek(0)

                        # Save these coordinates
                        largest_car_coords = {
                            "top_most": top_most,
                            "bottom_most": bottom_most,
                            "left_most": left_most,
                            "right_most": right_most,
                            "bytesio_image": image_bytes_io
                        }

        # Return the largest car's coordinates
        return largest_car_coords

    except Exception as e:
        print(f"An error occurred: {e}")
        return None




def create_proxy_background_with_car(bg_image: BytesIO, car_image: BytesIO) -> BytesIO:
    try:
        # Open the background and car images from BytesIO
        bg = Image.open(bg_image)
        car = Image.open(car_image)

        # Get the sizes of the car image
        car_width, car_height = car.size

        # Calculate the new height for the background while maintaining the aspect ratio
        bg_aspect_ratio = bg.height / bg.width
        new_bg_height = int(car_width * bg_aspect_ratio)

        # Resize the background image
        resized_bg = bg.resize((car_width, new_bg_height), Image.LANCZOS)

        # Create a canvas with the size of the car image
        canvas = Image.new("RGBA", (car_width, car_height), (0, 0, 0, 255))  # Transparent background

        # Determine the position to paste the resized background
        bg_position = (0, car_height - new_bg_height)  # Align the background to the bottom of the canvas
        canvas.paste(resized_bg, bg_position)

        # Position the car image on the canvas
        car_position = (0, 0)  # Place car at the top of the canvas
        canvas.paste(car, car_position, car.convert('RGBA'))

        # Flatten the image into an RGB format
        flattened_image = Image.new("RGB", canvas.size, (0, 0, 0))
        flattened_image.paste(canvas, mask=canvas.split()[3])

        # Save the final image to a BytesIO object
        output = BytesIO()
        flattened_image.save(output, format="PNG")
        output.seek(0)

        return output
    except Exception as e:
        log_to_json(f"Error occurs for {e}", current_file_name)
        return BytesIO()

