import cv2
import numpy as np
import os
import requests
from ultralytics import YOLO
from my_logging_script import log_to_json
from features.interiorprocess.interiorbg import save_18_19_image_to_s3
from io import BytesIO
from PIL import Image
from features.laravelconnection.processdoneapi import notify_error_status, notify_success_status
from dynamicfuncs import  save_image_with_timestamp, get_user_setting
from features.detections.fullcarexistence import is_full_car
wheel_model_path = 'models/wheels/wheeldetect3folder/wheel-3folder-best.pt'
headlight_model_path = 'models/headlights/v900/v900_best.pt'
zoomed_wheel_model_path = 'models/wheels/singlewheelzoomed/wheelaugmented_single.pt'
zoomed_hedlight_model_path = 'models/headlights/zoomedsingle/zoomednfull.pt'

current_file_name = "features/auto5_6angle/process_angle_5_6.py"



def process_image_with_denoising(image_bytesio: BytesIO) -> BytesIO:
    # Read the image from BytesIO into a NumPy array
    try:
        image_bytesio.seek(0)  # Ensure we're at the start
        file_bytes = np.asarray(bytearray(image_bytesio.read()), dtype=np.uint8)
        
        # Decode the image using OpenCV
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode image")

        # Apply fastNlMeansDenoisingColored
        denoised_image = cv2.fastNlMeansDenoisingColored(image, None, h=3, hColor=3, templateWindowSize=7, searchWindowSize=21)

        # Encode the processed image back into bytes
        _, encoded_image = cv2.imencode('.jpg', denoised_image)

        # Convert to BytesIO
        processed_bytesio = BytesIO(encoded_image.tobytes())
        processed_bytesio.seek(0)

        return processed_bytesio
    except Exception as e:
        log_to_json(f"Error in process_image_with_denoising: {str(e)}", current_file_name)
        return None

def extract_image_info(image_url: str):
    url_parts = image_url.split('/')    
    catalogue_id = url_parts[-3]  # Extract catalogue ID
    angle_id = url_parts[-2]      # Extract angle ID
    filename = url_parts[-1]      # Extract filename
    
    return catalogue_id, angle_id, filename


def process_wheel_headlight(image_url, angle_id_derived, userid, global_user_setting_data, catalogue_feature_list):
    log_to_json(f"process_wheel_headlight : started for {image_url}---{angle_id_derived}---------", current_file_name)
    try:
        from backgroundconfs import get_dynamic_basic_wall_images, get_dynamic_wall_images
        

        
        catalogue_id, angle_id, filename = extract_image_info(image_url)
        file_name, file_extension = os.path.splitext(filename)
        process_name = 'Wheel'

        # --------- get the orignal image and transparent image first -----------------
        # here is image single processed or derived from angle 2 is defined by angle id. if angle 2 then its dereived from angle 2 else 18/19 then it is clicked individually
        # image url may angle 2 or 18/19 and saved may angle2_temp or indangle. both image should be resized and add background
        # To identify is the image from angle 2 or it is single shot from angle 18/19 it should be checked from image url's angle id . 
        # if angle_id is 2 then it is derived from angle 2 else it is single shot from angle 18/19

        # angle_id_derived means request bodys angle id
        # angle_id means the image url angle id
        image_file, transparent_file = None, None
        folder_name = "angle2_temp" if angle_id == '2' else "indangle"


        orignal_file_path = f"config/assets/{folder_name}/{file_name}.png"
        with open(orignal_file_path, 'rb') as f:
            image_file = BytesIO(f.read())
        transparent_file_path = f"config/assets/{folder_name}/{file_name}_transparent.png"
        with open(transparent_file_path, 'rb') as f:
            transparent_file = BytesIO(f.read())


        # --------- background create -------------------------------

        set_bg_type = get_user_setting(global_user_setting_data, userid, "Background Type")
        set_def_background = get_user_setting(global_user_setting_data, userid, "Default Background")
        # will cross check later if these value exist or not
        
        # get wall images:
        if set_bg_type == 'photo_box':
            wallImages_bytes = get_dynamic_wall_images(set_def_background)
        elif set_bg_type == 'basic':
            wallImages_bytes = get_dynamic_basic_wall_images(set_def_background)
        else:
            wallImages_bytes = None

        floor_image = wallImages_bytes['floor_wall_img']
        left_wall_image = wallImages_bytes['left_wall_img']
        print(f"floor image type {type(floor_image)}")
        
        if angle_id_derived == 19:
            # points_list = [[(-2006,390),(-2526,-2564),(2440,2581),(1920,-373)]]
            points_list = [
                            [-2806, 1890], # Top-left
                            [2920, 1650],   # Top-right
                            [5840, 2340],   # Bottom-right
                            [-2526, 3064]  # Bottom-left
                            ]
            bg_image = warp_image_to_canvas(points_list, floor_image, angle_id_derived)
        else:
            points_list = [
                            [0, -500], # Top-left
                            [2500, 0],   # Top-right
                            [2500, 1440],   # Bottom-right
                            [0, 2564]  # Bottom-left
                            ]
            bg_image = warp_image_to_canvas(points_list, left_wall_image, angle_id_derived)
        
        
        
        log_to_json(f"bg image type {type(bg_image)} orignal image type {type(image_file)} transparent image type {type(transparent_file)}", current_file_name)
        


        # ---------------- detect and crop image -------------------------- it will process for bothe angle id case like angle 2 and 18/19
        if angle_id_derived == 19 :
            cropped_object = wheel_process(image_file, transparent_file)
    
        else:
            process_name = 'Headlight'
            cropped_object = headlight_process(image_file, transparent_file)
            



        if cropped_object is None and angle_id == '2':
            log_to_json(f"process_wheel_headlight: Cant Proceed because of not have clear image.", current_file_name)
            notify_error_status(catalogue_id, angle_id_derived, filename, f"{process_name} auto generation failed. {process_name} could not be detected.", current_file_name)
            return None
        # elif cropped_object is None and angle_id != '2':
        #     notify_error_status(catalogue_id, angle_id_derived, filename, f"{process_name} could not be detected. Processed your image with background", current_file_name, notify=1)
        #     log_to_json(f"process_wheel_headlight: Object couldnot  detected, setting background.", current_file_name)
        #     cropped_object = transparent_file

        
        processed_angle = combine_bg_fg(bg_image, cropped_object)

        print(f"processed_angle file type {type(processed_angle)}")
        
        processed_angle.seek(0)
        processed_angle = process_image_with_denoising(processed_angle)
        # save_image_with_timestamp(processed_angle, 'outputs/indtest/lombaprob', 'wheel_head.png')
        final_image_link = save_18_19_image_to_s3(processed_angle, filename, catalogue_id, angle_id_derived)
        if final_image_link:
            log_to_json(f"Processed and uploaded image: {final_image_link}", current_file_name)
                
                # Extract filename from final_image_link
            processed_filename = final_image_link.split('/')[-1]
            notify_response = notify_success_status(catalogue_id, angle_id_derived, processed_filename, current_file_name)

        else:
            log_to_json(f"Failed to upload the processed image to S3.", current_file_name)

        
        return final_image_link

    except FileNotFoundError as e:
        log_to_json(f"File not found error {e}", current_file_name)
        notify_error_status(catalogue_id, angle_id_derived, filename, f"System error occurred. It will be available soon.", current_file_name)
        return None
    except ValueError as e:
        log_to_json(f"ValueError {e}", current_file_name)
        notify_error_status(catalogue_id, angle_id_derived, filename, f"System error occurred. It will be available soon.", current_file_name)
        return None
    except Exception as e:
        log_to_json(f"Exceptional error {e}", current_file_name)
        notify_error_status(catalogue_id, angle_id_derived, filename, f"System error occurred. It will be available soon.", current_file_name)
        return  None




def warp_image_to_canvas(points_list, image_bytesio, angle_id_derived):
    try:
        # Open the image from BytesIO
        image = Image.open(image_bytesio)
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Adjust points so the minimum x and y are shifted to 0

        # Ensure minimum canvas size
        canvas_width = 4000
        canvas_height = 3000

        # Create a blank gray canvas (RGB: 128,128,128)
        canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 128  # Gray background

    
            # Ensure the points are in correct order (Top-Left, Top-Right, Bottom-Right, Bottom-Left)
        dst_pts = np.array(points_list, dtype=np.float32) 

            # Define fixed source points (original image coordinates)
        src_pts = np.array([
                [0, 0],       # Top-left
                [4000, 0],    # Top-right
                [4000, 3000], # Bottom-right
                [0, 3000]     # Bottom-left
            ], dtype=np.float32)

            # Compute the perspective transform matrix
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

            # Resize the input image to match the new canvas size before warping
        resized_image = cv2.resize(image_cv, (4000, 3000))

            # Perform the warp
        warped_image = cv2.warpPerspective(resized_image, matrix, (canvas_width, canvas_height))

            # Composite the warped image onto the gray canvas (only non-background areas)
        mask = (warped_image != 128).any(axis=-1)  # Avoid replacing gray background
        canvas[mask] = warped_image[mask]

        # Convert canvas back to PIL
        final_image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))

        # Crop to 1920x1440 from the center if needed
        if canvas_width > 1920 or canvas_height > 1440:
            left = 0
            top = 1900

            bottom = 2540

            if angle_id_derived == 18:
                top = 0
                bottom = 1440

            # Ensure correct aspect ratio (1920x1440)
            aspect_ratio = 1920 / 1440  
            right = left + int((bottom - top) * aspect_ratio)

            # Crop & Resize correctly
            final_image = final_image.crop((left, top, right, bottom))
            final_image = final_image.resize((1920, 1440))



        # Save to BytesIO
        output_bytesio = BytesIO()
        final_image.save(output_bytesio, format="PNG")
        output_bytesio.seek(0)

        return output_bytesio
    except Exception as e:
        print(f"warp_image_to_canvas: Exceptional error: {e}")
        return None

# Example input



def combine_bg_fg(background_bytesio, foreground_bytesio):
    try:
        # Canvas size
        canvas_width, canvas_height = 1920, 1440

        background = Image.open(background_bytesio).convert("RGB")
        foreground = Image.open(foreground_bytesio)

        # Resize background to fill the entire canvas (1920x1440)
        background_resized = background.resize((canvas_width, canvas_height), Image.LANCZOS)

        # Create blank RGBA canvas
        canvas = Image.new('RGBA', (canvas_width, canvas_height), (255, 255, 255, 255))

        # Paste resized background onto canvas
        canvas.paste(background_resized, (0, 0))

        # Check if foreground has an alpha channel
        if foreground.mode != 'RGBA':
            log_to_json("combine_bg_fg: Warning: Foreground image is not transparent (no alpha channel).", current_file_name)
            # Convert to RGBA so it can be pasted
            foreground = foreground.convert('RGBA')

        # Paste foreground with transparency
        canvas.paste(foreground, (0, 0), foreground)

        # Save with transparency (PNG format)
        output_bytesio = BytesIO()
        canvas.save(output_bytesio, format="PNG")
        output_bytesio.seek(0)

        return output_bytesio

    except Exception as e:
        log_to_json(f"combine_bg_fg: Exceptional error {e}", current_file_name)
        return None

from typing import Union

def optimized_cropped(img: np.ndarray) -> tuple[np.ndarray, Union[tuple[int, int, int, int], None]]:
    """Crop the image with fixed dimensions: 400px from top, 200px from bottom, 300px from left & right."""
    print("Cropping image")

    # Get image dimensions
    h, w = img.shape[:2]

    # Ensure cropping does not exceed image size
    top_crop = min(400, h)
    bottom_crop = min(200, h - top_crop)
    left_crop = min(300, w)
    right_crop = min(300, w - left_crop)

    # Compute safe cropping coordinates
    y1, y2 = top_crop, h - bottom_crop
    x1, x2 = left_crop, w - right_crop

    # Validate crop dimensions
    if y1 >= y2 or x1 >= x2:
        print("Error: Invalid crop dimensions, returning original image")
        return img, None  # Return original if crop is invalid

    cropped_img = img[y1:y2, x1:x2]
    return cropped_img, (x1, y1, x2, y2)



def wheel_headlight_process(image_bytes: BytesIO, angle_type: str) -> Union[BytesIO, None]:
    log_to_json("wheel_headlight_process: Got the file in wheel_headlight_process function", current_file_name)

    try:
        # Load the input image
        image_array = np.frombuffer(image_bytes.read(), np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if img is None:
            log_to_json("wheel_headlight_process: Unable to read image.", current_file_name)
            return None

        original_height, original_width = img.shape[:2]

        # Load the YOLO model
        print(f"Angle type is {angle_type}")
        model = YOLO(wheel_model_path) if angle_type == 'wheel' else YOLO(headlight_model_path)
        

        def detect_objects(img: np.ndarray, first_attempt: bool = True, crop_coords: tuple[int, int, int, int] | None = None) -> np.ndarray:
            """Runs YOLO detection and processes the results"""
            log_to_json("~~/\~~~/\~~Running object detection~~~~", current_file_name)
            results = model(img)
            has_valid_masks = False
            combined_mask = np.zeros((original_height, original_width), dtype=np.uint8)

            for result in results:
                if result.masks is None:
                    continue  # No mask found, check other results
                has_valid_masks = True  # At least one valid mask found
                for mask in result.masks.data:
                    mask = (mask.cpu().numpy() * 255).astype('uint8')

                    if crop_coords:
                        x1, y1, x2, y2 = crop_coords
                    else:
                        x1, y1, x2, y2 = 0, 0, original_width, original_height

                    mask_resized = cv2.resize(mask, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
                    combined_mask[y1:y2, x1:x2] = mask_resized

            # If no masks found, try optimized_cropped (one-time recursion)
            if not has_valid_masks and first_attempt:
                log_to_json("wheel_headlight_process: No masks found, trying optimized cropping.", current_file_name)
                cropped_img, crop_coords = optimized_cropped(img)
                if crop_coords is not None:
                    return detect_objects(cropped_img, first_attempt=False, crop_coords=crop_coords)

            return combined_mask

        # Run object detection and get mask
        combined_mask = detect_objects(img, first_attempt=(angle_type == 'headlight'))

        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            log_to_json("wheel_headlight_process: No segmented areas found.", current_file_name)
            return None

        # Find the largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)

        # Find extreme points of the largest contour
        top_left = tuple(largest_contour[np.argmin(largest_contour[:, 0, 0] + largest_contour[:, 0, 1])][0])
        top_right = tuple(largest_contour[np.argmax(largest_contour[:, 0, 0] - largest_contour[:, 0, 1])][0])
        bottom_left = tuple(largest_contour[np.argmin(largest_contour[:, 0, 0] - largest_contour[:, 0, 1])][0])
        bottom_right = tuple(largest_contour[np.argmax(largest_contour[:, 0, 0] + largest_contour[:, 0, 1])][0])

        # Calculate the midpoint of the trapezium
        midpoint_x = (top_left[0] + top_right[0] + bottom_left[0] + bottom_right[0]) // 4
        midpoint_y = (top_left[1] + top_right[1] + bottom_left[1] + bottom_right[1]) // 4

        # Calculate new points for cropping
        if angle_type == 'wheel':
            cropped_width, cropped_height = 752, 564  # Aspect ratio 1920x1440
        else:
            cropped_width, cropped_height = 640, 480

        x_half, y_half = cropped_width // 2, cropped_height // 2
        x1, x2 = midpoint_x - x_half, midpoint_x + x_half
        y1, y2 = midpoint_y - y_half, midpoint_y + y_half

        # Clamp the points to the image boundaries
        x1, x2 = max(0, x1), min(original_width, x2)
        y1, y2 = max(0, y1), min(original_height, y2)

        print(f"Cropping region: x1={x1}, x2={x2}, y1={y1}, y2={y2}")

        # Ensure valid dimensions after clamping
        if x1 >= x2 or y1 >= y2:
            log_to_json(f"Error: Invalid cropping dimensions. x1={x1}, y1={y1}, x2={x2}, y2={y2}", current_file_name)
            return None

        # Crop the area defined by the new points
        cropped_img = img[y1:y2, x1:x2]

        # Validate the cropped image dimensions
        if cropped_img.shape[0] == 0 or cropped_img.shape[1] == 0:
            log_to_json("Error: Cropped region is empty.", current_file_name)
            return None

        # Resize the cropped image
        new_width = 1920
        aspect_ratio = cropped_img.shape[0] / cropped_img.shape[1]  # height / width
        new_height = int(new_width * aspect_ratio)

        resized_img = cv2.resize(cropped_img, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Convert the image to PIL format for saving
        pil_img = Image.fromarray(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB))

        # Save the image to a BytesIO object
        output = BytesIO()
        pil_img.save(output, format="PNG")
        output.seek(0)

        return output

    except FileNotFoundError as e:
        log_to_json(f"wheel_headlight_process: File not found error {e}", current_file_name)
        return None
    except ValueError as e:
        log_to_json(f"wheel_headlight_process: ValueError {e}", current_file_name)
        return None
    except Exception as e:
        log_to_json(f"wheel_headlight_process: Exceptional error {e}", current_file_name)
        return None
    # # Save the result
    # cv2.imwrite(output_image_path, resized_img)
    # print(f"Cropped and resized image saved as '{output_image_path}'.")



def wheel_process(image_bytes: BytesIO, transparent_image: BytesIO) -> BytesIO:
    log_to_json("Got the file in wheel_headlight_process funcion", current_file_name)
    # Load the input image
    try:
        image_array = np.frombuffer(image_bytes.read(), np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        transparent_image_array = np.frombuffer(transparent_image.read(), np.uint8)
        transparent_img = cv2.imdecode(transparent_image_array, cv2.IMREAD_UNCHANGED)



        if img is None:
            log_to_json(f"wheel_process: Error: Unable to read image.", current_file_name)
            return None
        
        original_height, original_width = img.shape[:2]


        model = YOLO(wheel_model_path)
        model2 = YOLO(zoomed_wheel_model_path)



        # Run inference on the image
        results = model(img)
        if not results or all(result.masks is None for result in results):
            log_to_json("wheel_process: No wheel found by model, trying model2.", current_file_name)
            results = model2(img)

            if not results or all(result.masks is None for result in results):
                log_to_json("wheel_process: No wheel found by model2 either.", current_file_name)
                return None

        # Combine all masks into a single binary mask
        combined_mask = np.zeros((original_height, original_width), dtype=np.uint8)
        for result in results:
            if result.masks is None:
                log_to_json("wheel_process: No masks found for the detected object.", current_file_name)
                continue
            for mask in result.masks.data:
                mask = (mask.cpu().numpy() * 255).astype('uint8')
                mask_resized = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
                combined_mask = cv2.bitwise_or(combined_mask, mask_resized)

        # Find contours in the combined mask
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            log_to_json("wheel_process: No segmented areas found.",current_file_name)
            return None

        # Find the largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)

        # Find extreme points of the largest contour
        # Find extreme directional points of the largest contour
        top_most = tuple(largest_contour[largest_contour[:, :, 1].argmin()][0])
        bottom_most = tuple(largest_contour[largest_contour[:, :, 1].argmax()][0])
        left_most = tuple(largest_contour[largest_contour[:, :, 0].argmin()][0])
        right_most = tuple(largest_contour[largest_contour[:, :, 0].argmax()][0])


        # Calculate the midpoint of the trapezium
        midpoint_x, midpoint_y = int((left_most[0]+right_most[0])/2.3), int((top_most[1] + bottom_most[1]) / 2)

        # Calculate new points for cropping
        bottom_space = original_height - bottom_most[1]
        wheel_height = bottom_most[1] - top_most[1]
        top_padding = int(wheel_height * 0.22) # 220/1000 = top_pad/wheel_height equation ratio
        crop_y = original_height - bottom_space - wheel_height - top_padding
        crop_width = int(1.9 * wheel_height)  # 1920/1000 = crop_width/wheel_height equation ratio
        crop_x = midpoint_x - int(crop_width / 2)
        crop_height = wheel_height + top_padding + bottom_space
        
        
        crop_y = max(0, crop_y)
        crop_x = max(0, crop_x)
        crop_width = min(crop_width, original_width - crop_x)
        crop_height = min(crop_height, original_height - crop_y)

        # Crop the transparent image
        # Save original transparent image for debugging
        # cv2.imwrite("outputs/sizedtest/transparent_before_crop.png", transparent_img)

        cropped_transparent_img = transparent_img[crop_y:crop_y+crop_height, crop_x:crop_x+crop_width]

        
        # cv2.imwrite("outputs/sizedtest/transparent_after_crop.png", cropped_transparent_img)

        
 


        # Resize the cropped transparent image with transparency preserved
        new_width = 1920
        aspect_ratio = crop_height / crop_width  # height / width
        new_height = int(new_width * aspect_ratio)
        resized_img = cv2.resize(cropped_transparent_img, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # cv2.imwrite("outputs/sizedtest/resized_transparent.png", resized_img)

        # Convert to PIL image with transparency
        if resized_img.shape[2] == 4:  # Check if it has an alpha channel
            pil_img = Image.fromarray(cv2.cvtColor(resized_img, cv2.COLOR_BGRA2RGBA))
        else:
            log_to_json("wheel_process: Warning - Resized image lost transparency (no alpha channel).", current_file_name)
            pil_img = Image.fromarray(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB))

        # Save as PNG with transparency
        output = BytesIO()
        pil_img.save(output, format="PNG")
        output.seek(0)

        return output
    except FileNotFoundError as e:
        log_to_json(f"wheel_process: File not found error {e}", current_file_name)
        return None
    except ValueError as e:
        log_to_json(f"wheel_process: ValueError {e}", current_file_name)
        return None
    except Exception as e:
        log_to_json(f"wheel_process: Exceptional error {e}", current_file_name)
        return  None
    

def headlight_process(image_bytes: BytesIO, transparent_image: BytesIO) -> BytesIO:
    log_to_json("Got the file in headlight_process funcion", current_file_name)
    # Load the input image
    try:
        image_array = np.frombuffer(image_bytes.read(), np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        transparent_image_array = np.frombuffer(transparent_image.read(), np.uint8)
        transparent_img = cv2.imdecode(transparent_image_array, cv2.IMREAD_UNCHANGED)



        if img is None:
            log_to_json(f"wheel_process: Error: Unable to read image.", current_file_name)
            return None
        
        original_height, original_width = img.shape[:2]


        model = YOLO(headlight_model_path)
        model2 = YOLO(zoomed_hedlight_model_path)


        # Run inference on the image
        results = model(img)
        if not results or all(result.masks is None for result in results):
            log_to_json("headlight_process: No headlights found by model, trying model2.", current_file_name)
            results = model2(img)

            if not results or all(result.masks is None for result in results):
                log_to_json("headlight_process: No headlights found by model2 either.", current_file_name)
                return None

        # Combine all masks into a single binary mask
        combined_mask = np.zeros((original_height, original_width), dtype=np.uint8)
        for result in results:
            if result.masks is None:
                log_to_json("wheel_process: No masks found for the detected object.", current_file_name)
                continue
            for mask in result.masks.data:
                mask = (mask.cpu().numpy() * 255).astype('uint8')
                mask_resized = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
                combined_mask = cv2.bitwise_or(combined_mask, mask_resized)

        # Find contours in the combined mask
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            log_to_json("wheel_process: No segmented areas found.",current_file_name)
            return None

        # Find the largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)

        # Find extreme points of the largest contour
        # Find extreme directional points of the largest contour
        top_most = tuple(largest_contour[largest_contour[:, :, 1].argmin()][0])
        bottom_most = tuple(largest_contour[largest_contour[:, :, 1].argmax()][0])
        left_most = tuple(largest_contour[largest_contour[:, :, 0].argmin()][0])
        right_most = tuple(largest_contour[largest_contour[:, :, 0].argmax()][0])



        # Calculate the midpoint of the trapezium
        midpoint_x, midpoint_y = int((left_most[0] + right_most[0])/2.2), int((top_most[1] + bottom_most[1]) / 2.2)
        log_to_json(f"top_most: {top_most}, bottom_most: {bottom_most}, left_most: {left_most}, right_most: {right_most} mid {midpoint_x},{midpoint_y}", current_file_name)

        # Calculate new points for cropping
        bottom_space = original_height - bottom_most[1]
        headlight_width = abs(right_most[0] - left_most[0])
        crop_width = int(headlight_width * 1.25) # 1920/1400 = top_pad/wheel_height equation ratio
        crop_height = int((1440/1920) * crop_width)  # 1440/1920 = crop_height/crop_width equation ratio
        
        
        crop_y = midpoint_y - int(crop_height / 2) 
        crop_x = midpoint_x - int(crop_width / 2)
        
        
        
        crop_y = max(0, crop_y)
        crop_x = max(0, crop_x)
        crop_width = min(crop_width, original_width - crop_x)
        crop_height = min(crop_height, original_height - crop_y)

        # Crop the transparent image
        # Save original transparent image for debugging
        # cv2.imwrite("outputs/sizedtest/transparent_before_crop.png", transparent_img)
        log_to_json(f"headlight_process: Crop coordinates: x={crop_x}, y={crop_y}, width={crop_width}, height={crop_height}", current_file_name)

        cropped_transparent_img = transparent_img[crop_y:crop_y+crop_height, crop_x:crop_x+crop_width]

        if cropped_transparent_img is None or cropped_transparent_img.size == 0:
            log_to_json("headlight_process: Cropped transparent image is empty.", current_file_name)
            return None


        
        # cv2.imwrite("outputs/sizedtest/transparent_after_crop.png", cropped_transparent_img)

        
 


        # Resize the cropped transparent image with transparency preserved
        new_width = 1920
        aspect_ratio = crop_height / crop_width  # height / width
        new_height = int(new_width * aspect_ratio)
        resized_img = cv2.resize(cropped_transparent_img, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # cv2.imwrite("outputs/sizedtest/resized_transparent.png", resized_img)

        # Convert to PIL image with transparency
        if resized_img.shape[2] == 4:  # Check if it has an alpha channel
            pil_img = Image.fromarray(cv2.cvtColor(resized_img, cv2.COLOR_BGRA2RGBA))
        else:
            log_to_json("wheel_process: Warning - Resized image lost transparency (no alpha channel).", current_file_name)
            pil_img = Image.fromarray(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB))

        # Save as PNG with transparency
        output = BytesIO()
        pil_img.save(output, format="PNG")
        output.seek(0)

        return output
    except FileNotFoundError as e:
        log_to_json(f"headlight_process: File not found error {e}", current_file_name)
        return None
    except ValueError as e:
        log_to_json(f"headlight_process: ValueError {e}", current_file_name)
        return None
    except Exception as e:
        log_to_json(f"headlight_process: Exceptional error {e}", current_file_name)
        return  None