import cv2
import numpy as np
import os
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
from my_logging_script import log_to_json
from features.core.licenseplateoperations.licensplatealgorithm import licensplate_coordinates
current_file_name = "features/core/licenseplateoperations/licenseplateblurfunc.py"

def make_the_licenseplate_blur(model_path, input_image, transparent_image ):
    # Load the input image
    try:
        input_image.seek(0)
        pil_image = Image.open(input_image)
        img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        if img is None:
            print(f"Error: Unable to read image {input_image}")
            return None
        
        transparent_image.seek(0)
        pil_image_transparent = Image.open(transparent_image).convert("RGBA")
        img_transparent = np.array(pil_image_transparent)

        if img_transparent is None:
            log_to_json(f"addLicensePlateImage: Unable to read transparent image.", current_file_name)
            return None

        original_height, original_width = img.shape[:2]

        top_left, bottom_left, top_right, bottom_right = licensplate_coordinates(
            model_path, input_image,  transparent_image
        )



        if top_left is None or bottom_left is None or top_right is None or bottom_right is None:
        # Load the YOLO model
            model = YOLO(model_path)

            # Run inference on the image
            results = model(img)
            if not results:
                log_to_json("No licenseplate found in car by the YOLO model.", current_file_name)
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
            top_left = tuple(largest_contour[np.argmin(largest_contour[:, 0, 0] + largest_contour[:, 0, 1])][0])
            top_right = tuple(largest_contour[np.argmax(largest_contour[:, 0, 0] - largest_contour[:, 0, 1])][0])
            bottom_left = tuple(largest_contour[np.argmin(largest_contour[:, 0, 0] - largest_contour[:, 0, 1])][0])
            bottom_right = tuple(largest_contour[np.argmax(largest_contour[:, 0, 0] + largest_contour[:, 0, 1])][0])

            from features.core.licenseplateoperations.license_angle import make_parallelogram
            adjusted = make_parallelogram(top_left, top_right, bottom_left, bottom_right)
            top_left = adjusted["top_left"]
            top_right = adjusted["top_right"]
            bottom_left = adjusted["bottom_left"]
            bottom_right = adjusted["bottom_right"]



        # Draw the quadrilateral on the image
        quadrilateral = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.int32)
        highlighted_image = img_transparent.copy()

        # Create a mask for the quadrilateral area
        mask = np.zeros((original_height, original_width, 4), dtype=np.uint8)
        cv2.fillPoly(mask, [quadrilateral], (255, 255, 255, 255))  # White with full alpha in the mask

        # Extract only the quadrilateral region from the transparent image
        quadrilateral_region = cv2.bitwise_and(img_transparent[:, :, :3], mask[:, :, :3])

        # Apply Gaussian blur to the quadrilateral region
        blurred_region = cv2.GaussianBlur(quadrilateral_region, (51, 51), 0)

        # Blend the blurred quadrilateral region with the transparent image
        alpha_mask = mask[:, :, 3] / 255.0  # Extract alpha channel
        for c in range(3):  # Iterate over RGB channels
            highlighted_image[:, :, c] = (
                alpha_mask * blurred_region[:, :, c] + (1 - alpha_mask) * highlighted_image[:, :, c]
            )
        highlighted_image[:, :, 3] = (alpha_mask * 255 + (1 - alpha_mask) * highlighted_image[:, :, 3]).astype(np.uint8)  # Update alpha channel

        # Convert the result back to PIL Image with RGBA
        result_image = Image.fromarray(highlighted_image, mode="RGBA")
        output_image = BytesIO()
        result_image.save(output_image, format="PNG")
        output_image.seek(0)

        return output_image

    except FileNotFoundError as e:
        log_to_json(f"File not found error: {e}", current_file_name)
        return None
    except ValueError as e:
        log_to_json(f"ValueError: {e}", current_file_name)
        return None
    except Exception as e:
        log_to_json(f"Exceptional error: {e}", current_file_name)
        return None

    # Save the result
    # cv2.imwrite(output_image_path, highlighted_image)
    # print(f"Largest segmented area highlighted and saved as '{output_image_path}'.")


# def process_all_images_in_folder(model_path, input_folder, output_folder):
#     # Ensure output folder exists
#     os.makedirs(output_folder, exist_ok=True)

#     # Iterate over all files in the input folder
#     for filename in os.listdir(input_folder):
#         input_image_path = os.path.join(input_folder, filename)
        
#         if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
#             continue  # Skip non-image files
        
#         # Construct output image path
#         output_image_path = os.path.join(output_folder, f"highlighted_{filename}")

#         # Call the function to process the image
#         find_largest_segment_and_draw(model_path, input_image_path, output_image_path)


# Define paths
# model_path = './models/licenseplate/augmented/best.pt'
# input_folder = './data/otherangles'
# output_folder = './output/v495images/straight/otherangles'

# Process all images in the input folder
#process_all_images_in_folder(model_path, input_folder, output_folder)
