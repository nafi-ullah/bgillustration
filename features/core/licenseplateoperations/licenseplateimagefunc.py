import cv2
import numpy as np
import os
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
from my_logging_script import log_to_json
current_file_name = "features/core/licenseplateoperations/licenseplateimagefunc.py"

def addLicensePlateImage(model_path, input_image, transparent_image=None, license_plate=None):
    try:
    # Load the input image
        input_image.seek(0)
        pil_image = Image.open(input_image)
        img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        if img is None:
            log_to_json(f"addLicensePlateImage: Unable to read withbg image.", current_file_name)
            return None
        
        transparent_image.seek(0)
        pil_image_transparent = Image.open(transparent_image).convert("RGBA")
        img_transparent = np.array(pil_image_transparent)


        # print width and heeight of the images
        # input image size
        print(f"input image size: {img.shape}")
        # transparent image size
        print(f"transparent image size: {img_transparent.shape}")
        # license plate image size


        if img_transparent is None:
            log_to_json(f"addLicensePlateImage: Unable to read transparent image.", current_file_name)
            return None
        # if img_transparent.shape[-1] != 4:
        #     log_to_json(f"Transparent image does not have an alpha channel.", current_file_name)
        #     return None


        original_height, original_width = img.shape[:2]

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
        
        log_to_json(f"licencese plate points are tl, tr, bl, br: {top_left} {top_right} {bottom_left} {bottom_right} ", current_file_name)
        from features.core.licenseplateoperations.license_angle import make_parallelogram
        adjusted = make_parallelogram(top_left, top_right, bottom_left, bottom_right)
        top_left = adjusted["top_left"]
        top_right = adjusted["top_right"]
        bottom_left = adjusted["bottom_left"]
        bottom_right = adjusted["bottom_right"]

        log_to_json(f"licencese plate adjusted points are tl, tr, bl, br: {top_left} {top_right} {bottom_left} {bottom_right} ", current_file_name)
        # Draw the quadrilateral on the image
        quadrilateral = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.int32)
        highlighted_image = img_transparent.copy()

        # Load the image to fill the quadrilateral
        license_plate.seek(0)  # Ensure we're at the start of the BytesIO object
        image_array = np.frombuffer(license_plate.read(), np.uint8)

        # Decode the image using OpenCV
        fill_image = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)

        # Convert BGR to RGB if the image has 3 channels
        if fill_image.shape[2] == 3:
            fill_image = cv2.cvtColor(fill_image, cv2.COLOR_BGR2RGB)
        elif fill_image.shape[2] == 4:
            fill_image = cv2.cvtColor(fill_image, cv2.COLOR_BGRA2RGBA)

        if fill_image is None:
            print(f"Error: Unable to read fill image from BytesIO.")
            return input_image
        

        print("Color mean:", np.mean(fill_image.reshape(-1, fill_image.shape[2]), axis=0))


        # Warp the fill image
        src_points = np.array([[0, 0], [fill_image.shape[1], 0], [fill_image.shape[1], fill_image.shape[0]], [0, fill_image.shape[0]]], dtype=np.float32)
        dst_points = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
        perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        warped_fill_image = cv2.warpPerspective(fill_image, perspective_matrix, (original_width, original_height))

        # Add white background behind warped_fill_image
        white_bg = np.ones_like(warped_fill_image, dtype=np.uint8) * 255
        if warped_fill_image.shape[2] == 4:
            alpha_fill = warped_fill_image[:, :, 3] / 255.0
            for c in range(3):
                warped_fill_image[:, :, c] = (
                    alpha_fill * warped_fill_image[:, :, c] + (1 - alpha_fill) * white_bg[:, :, c]
                ).astype(np.uint8)
            warped_fill_image[:, :, 3] = 255
        else:
            # If no alpha channel, use full opacity
            alpha_fill = np.ones((original_height, original_width), dtype=np.float32)
            for c in range(3):
                warped_fill_image[:, :, c] = (
                    alpha_fill * warped_fill_image[:, :, c] + (1 - alpha_fill) * white_bg[:, :, c]
                ).astype(np.uint8)
            # Add alpha channel manually
            alpha_channel = np.full((original_height, original_width), 255, dtype=np.uint8)
            warped_fill_image = np.dstack((warped_fill_image, alpha_channel))


        # Create mask for license plate area
        mask = np.zeros((original_height, original_width, 4), dtype=np.uint8)
        cv2.fillPoly(mask, [quadrilateral], (255, 255, 255, 255))

        # Blend onto the highlighted image
        # Ensure warped_fill_image matches the output size
        if warped_fill_image.shape[:2] != (original_height, original_width):
            warped_fill_image = cv2.resize(warped_fill_image, (original_width, original_height), interpolation=cv2.INTER_LINEAR)

        # Ensure alpha mask matches shape
        if mask.shape[:2] != (original_height, original_width):
            mask = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

        alpha_mask = mask[:, :, 3] / 255.0

        for c in range(3):
            highlighted_image[:, :, c] = (
                alpha_mask * warped_fill_image[:, :, c] + (1 - alpha_mask) * highlighted_image[:, :, c]
            ).astype(np.uint8)
        highlighted_image[:, :, 3] = (
            alpha_mask * 255 + (1 - alpha_mask) * highlighted_image[:, :, 3]
        ).astype(np.uint8)

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
    # # Save the result
    # cv2.imwrite(output_image_path, highlighted_image)
    # print(f"Largest segmented area filled with image and saved as '{output_image_path}'.")

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

        # Call the function to process the image
        #find_largest_segment_and_draw(model_path, input_image_path, output_image_path)


# Define paths
# model_path = './models/v495images/best.pt'
# input_folder = './data/otherangles'
# output_folder = './output/v495images/withlicense/otherangles'

# Process all images in the input folder
# process_all_images_in_folder(model_path, input_folder, output_folder)
