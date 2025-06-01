import cv2
import os
import numpy as np
from main import get_unique_filename
from rembg import remove
from features.core.licenseplateoperations.blurfunc import blur_license_plate
from my_logging_script import log_to_json
from datetime import datetime
from PIL import Image, ImageOps
import requests
from io import BytesIO
import boto3

from botocore.client import Config
# Load the image with alpha channel (transparency)
access_key = '4IUMPGRCHXNLAJ5EUUGP'
secret_key = 'LakMU0aI4zTtEd3dzxr2LUD5R9EgvEMtBTgf7ukd'
region = 'eu-de'
bucket_name = 'vroomview'
endpoint_url = 'https://obs.eu-de.otc.t-systems.com'
get_url = 'https://vroomview.obs.eu-de.otc.t-systems.com'
current_file_name = "reverse.py"
s3_client = boto3.client(
    's3',
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key,
    region_name=region,
    endpoint_url=endpoint_url,
    config=Config(s3={'addressing_style': 'path'})
)

current_file_name = "reverse.py"

def crop_image(image, left_pixels=100, right_pixels=200, top_pixels=0, bottom_pixels=0):
  # left_pixels=80, right_pixels=150,
    width, height = image.size

    # Calculate the cropping box
    left = left_pixels
    right = width - right_pixels
    top = top_pixels
    bottom = height - bottom_pixels

    # Crop the image using the calculated box
    cropped_image = image.crop((left, top, right, bottom))

    return cropped_image

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

from PIL import Image
import numpy as np
import cv2
from io import BytesIO

def process_image_return_byte(image_path, direction, isPremium, filename, catalogue_id, angle_id):
    # Try to read the image
    log_to_json(f"got the save image path in process_image_return_byte function {image_path}", current_file_name)
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    # Check if the image was successfully loaded
    if image is None:
        log_to_json(f"Error: Unable to load image at {image_path}", current_file_name)
        return None  # Or handle it in a way that fits your use case
    
    # Check if the image has an alpha channel (transparency)
    if image.shape[2] == 4:
        # Separate the alpha channel from the image
        alpha_channel = image[:, :, 3]
        
        # Find the bounding box of the non-transparent object
        _, thresh = cv2.threshold(alpha_channel, 0, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            # Find the largest contour which corresponds to the main object
            main_contour = max(contours, key=cv2.contourArea)
            
            # Get the bounding box of the main object
            x, y, w, h = cv2.boundingRect(main_contour)
            
            # Define the left and right boundaries with an extra 100px margin
            left_bound = max(x - 150, 0)
            right_bound = min(x + w + 200, image.shape[1])
            height = image.shape[0]
            reverse_target_height = 744
            # Crop the image to keep the main object with a 100px margin on the sides
            if direction == 1:
                cropped_image = image[:height - 300, :, :]
            elif direction == 2:
                cropped_image = image[:height - 200, :, :]
            
            # You can add further processing for the cropped_image here
            
    
    # Crop 200px from the bottom
            cropped_image_rotate_portion = image[:height - 10, :, :]

            #cropped_image_rotate_portion = image[:y + int(1.2 * h), left_bound:right_bound, :]
          
            # Rotate the cropped image
            if direction == 1:
                rotated_reflection = rotate_image_with_transparency(cropped_image_rotate_portion, -30)
                alpha = 0.1
            elif direction == 2:
                rotated_reflection = rotate_image_with_transparency(cropped_image_rotate_portion, 33)
                alpha = 0.05
            else:
                raise ValueError("Invalid direction. Use 1 for -30 degrees or 2 for 30 degrees.")
            
            # Create a vertical reflection of the rotated image
            reflected_image = cv2.flip(rotated_reflection, 0)
            
            # Apply a strong blur to the reflection
            blurred_reflection = cv2.GaussianBlur(reflected_image, (25, 25), 20)
            
            # Adjust the opacity of the reflection by manipulating the alpha channel
              # Set the opacity of the reflection
            blurred_reflection[:, :, 3] = (blurred_reflection[:, :, 3] * alpha).astype(np.uint8)
            
            # Create a canvas to combine the original cropped image with its reflection
            reflection_height = blurred_reflection.shape[0]
            
            if direction == 1:
                # No need to increase the canvas size
                combined_image = np.zeros((cropped_image.shape[0] + reflection_height, cropped_image.shape[1] + 0, 4), dtype=np.uint8)
                # Place the original cropped image on top of the combined image
                combined_image[:cropped_image.shape[0], :cropped_image.shape[1]] = cropped_image
                # Place the blurred and low-opacity reflection below the cropped image, shifted 120px to the right
                combined_image[cropped_image.shape[0]:, 0:0 + blurred_reflection.shape[1]] = blurred_reflection
                for i in range(reflection_height):
                    alpha_fade = 1.0 - (i / float(reflection_height))
                    combined_image[cropped_image.shape[0] + i, :, 3] = (combined_image[cropped_image.shape[0] + i, :, 3] * alpha_fade).astype(np.uint8)

            elif direction == 2:
                # Increase the canvas width by 200px to the left
                combined_height = cropped_image.shape[0] + reflection_height
                combined_width = cropped_image.shape[1]   # Add 200px to the left for shifting
                canvas = np.zeros((combined_height, combined_width, 4), dtype=np.uint8)
                
                # Place the original cropped image on the canvas (shifted 200px to the right)
                canvas[:cropped_image.shape[0], 0:0 + cropped_image.shape[1], :] = cropped_image
                
                # Place the reflected image shifted 200px to the left
                canvas[cropped_image.shape[0]:, :blurred_reflection.shape[1], :] = blurred_reflection
                
                combined_image = canvas
            
            # Save the resulting image with transparency to the specified path
            ensure_directory_exists('./output_reversed/')
            combined_filename = get_unique_filename('./output_reversed/', 'reflected', 'png')
            cv2.imwrite(f'./output_reversed/{combined_filename}', combined_image)


            
            log_to_json("Image with reflection and transparent background saved successfully.")
        else:
            log_to_json("No non-transparent object detected in the image.")
    else:
        log_to_json("The image does not have a transparency channel, assuming no transparent object.")


    foreground_top = Image.open(image_path)
    
    if direction == 1:
        background = Image.open('./backgrounds/universalcarbg.png')
        position = (0,0)
    elif direction == 2:
        background = Image.open('./backgrounds/universalcarbg.png')
        position = (
            0, 0)
    else:
        raise ValueError("Invalid direction. Use 1 for -30 degrees or 2 for 30 degrees.")
    #background = Image.open('./backgrounds/backright.png')
    foreground = Image.open(f'./output_reversed/{combined_filename}')
    


    foreground.paste(foreground_top, position, foreground_top)
    #foreground = crop_image(foreground)
    # Resize the foreground to maintain aspect ratio if necessary
    foreground_ratio = foreground.width / foreground.height
    background_ratio = background.width / background.height
    
    new_width = 1530 # background.width
    target_height = 1200
    aspect_ratio = foreground.height / foreground.width
    new_height = int(new_width * aspect_ratio)
    # foreground_resized = foreground.resize((new_width, new_height), Image.LANCZOS)
    foreground_resized_cropped = process_foreground(foreground, new_width, target_height)

    # Calculate the position to center the foreground on the background
     
    #the background image is 1600px. the forground's mid should position at 1200px of the image.
    if height > 728 :
        upValue = 150
    elif height < 728 : 
        upValue = 250

    if direction == 1:
      position = (70, upValue) #280
    elif direction == 2:
        newupValue = upValue + 50
        position = (45,newupValue)
    # Paste the foreground onto the background, considering the alpha channel
    background.paste(foreground_resized_cropped, position, foreground_resized_cropped)
    results_directory = './results/'
    ensure_directory_exists(results_directory)
    output_filename = get_unique_filename(results_directory, 'output', 'png')
    # Save the final combined image
    print(output_filename)
    background.save(f'./results/{output_filename}')
    outputfile = blur_license_plate(f'./results/{output_filename}')
    result_string = f'./results/{output_filename}' # return this string
    
    return result_string



def rotate_image_with_transparency(image, angle):
    # Separate the color and alpha channels
    if image.shape[2] == 4:
        bgr = image[:, :, :3]
        alpha = image[:, :, 3]

        # Rotate the color channels
        center = (bgr.shape[1] // 2, bgr.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_bgr = cv2.warpAffine(bgr, rotation_matrix, (bgr.shape[1], bgr.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)

        # Rotate the alpha channel
        rotated_alpha = cv2.warpAffine(alpha, rotation_matrix, (alpha.shape[1], alpha.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)

        # Merge the rotated color channels with the rotated alpha channel
        rotated_image = np.dstack([rotated_bgr, rotated_alpha])
    else:
        # If there's no alpha channel, simply rotate the image
        center = (image.shape[1] // 2, image.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
    
    image_no_bg = remove(rotated_image)
    return image_no_bg

def process_image_with_reflection(image_path, direction, isPremium):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # Check if the image has an alpha channel (transparency)
    if image.shape[2] == 4:
        # Separate the alpha channel from the image
        alpha_channel = image[:, :, 3]
        
        # Find the bounding box of the non-transparent object
        _, thresh = cv2.threshold(alpha_channel, 0, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            # Find the largest contour which corresponds to the main object
            main_contour = max(contours, key=cv2.contourArea)
            
            # Get the bounding box of the main object
            x, y, w, h = cv2.boundingRect(main_contour)
            
            # Define the left and right boundaries with an extra 100px margin
            left_bound = max(x - 150, 0)
            right_bound = min(x + w + 200, image.shape[1])
            height = image.shape[0]
            reverse_target_height = 744
            #Crop the image to keep the main object with a 100px margin on the sides
            if direction == 1:
                cropped_image = image[:height - 300, :, :]
            elif direction == 2:
                cropped_image = image[:height - 200, :, :]
            
    
    # Crop 200px from the bottom
            cropped_image_rotate_portion = image[:height - 10, :, :]

            #cropped_image_rotate_portion = image[:y + int(1.2 * h), left_bound:right_bound, :]
          
            # Rotate the cropped image
            if direction == 1:
                rotated_reflection = rotate_image_with_transparency(cropped_image_rotate_portion, -30)
                alpha = 0.1
            elif direction == 2:
                rotated_reflection = rotate_image_with_transparency(cropped_image_rotate_portion, 33)
                alpha = 0.05
            else:
                raise ValueError("Invalid direction. Use 1 for -30 degrees or 2 for 30 degrees.")
            
            # Create a vertical reflection of the rotated image
            reflected_image = cv2.flip(rotated_reflection, 0)
            
            # Apply a strong blur to the reflection
            blurred_reflection = cv2.GaussianBlur(reflected_image, (25, 25), 20)
            
            # Adjust the opacity of the reflection by manipulating the alpha channel
              # Set the opacity of the reflection
            blurred_reflection[:, :, 3] = (blurred_reflection[:, :, 3] * alpha).astype(np.uint8)
            
            # Create a canvas to combine the original cropped image with its reflection
            reflection_height = blurred_reflection.shape[0]
            
            if direction == 1:
                # No need to increase the canvas size
                combined_image = np.zeros((cropped_image.shape[0] + reflection_height, cropped_image.shape[1] + 0, 4), dtype=np.uint8)
                # Place the original cropped image on top of the combined image
                combined_image[:cropped_image.shape[0], :cropped_image.shape[1]] = cropped_image
                # Place the blurred and low-opacity reflection below the cropped image, shifted 120px to the right
                combined_image[cropped_image.shape[0]:, 0:0 + blurred_reflection.shape[1]] = blurred_reflection
                for i in range(reflection_height):
                    alpha_fade = 1.0 - (i / float(reflection_height))
                    combined_image[cropped_image.shape[0] + i, :, 3] = (combined_image[cropped_image.shape[0] + i, :, 3] * alpha_fade).astype(np.uint8)

            elif direction == 2:
                # Increase the canvas width by 200px to the left
                combined_height = cropped_image.shape[0] + reflection_height
                combined_width = cropped_image.shape[1]   # Add 200px to the left for shifting
                canvas = np.zeros((combined_height, combined_width, 4), dtype=np.uint8)
                
                # Place the original cropped image on the canvas (shifted 200px to the right)
                canvas[:cropped_image.shape[0], 0:0 + cropped_image.shape[1], :] = cropped_image
                
                # Place the reflected image shifted 200px to the left
                canvas[cropped_image.shape[0]:, :blurred_reflection.shape[1], :] = blurred_reflection
                
                combined_image = canvas
            
            # Save the resulting image with transparency to the specified path
            ensure_directory_exists('./outputs/output_reversed/')
            combined_filename = get_unique_filename('./outputs/output_reversed/', 'reflected', 'png')
            cv2.imwrite(f'./outputs/output_reversed/{combined_filename}', combined_image)


            
            log_to_json("Image with reflection and transparent background saved successfully.", current_file_name)
        else:
            log_to_json("No non-transparent object detected in the image.", current_file_name)
    else:
        log_to_json("The image does not have a transparency channel, assuming no transparent object.", current_file_name)


    foreground_top = Image.open(image_path)
    
    if direction == 1:
        background = Image.open('./backgrounds/bluishwhitebg.png')
        position = (0,0)
    elif direction == 2:
        background = Image.open('./backgrounds/bluishwhitebg.png')
        position = (
            0, 0)
    else:
        log_to_json("Invalid direction. Use 1 for -30 degrees or 2 for 30 degrees.", current_file_name)
    
    #background = Image.open('./backgrounds/backright.png')
    foreground = Image.open(f'./output_reversed/{combined_filename}')
    


    foreground.paste(foreground_top, position, foreground_top)
    #foreground = crop_image(foreground)
    # Resize the foreground to maintain aspect ratio if necessary
    foreground_ratio = foreground.width / foreground.height
    background_ratio = background.width / background.height
    
    new_width = 1530 # background.width
    target_height = 1200
    aspect_ratio = foreground.height / foreground.width
    new_height = int(new_width * aspect_ratio)
    # foreground_resized = foreground.resize((new_width, new_height), Image.LANCZOS)
    foreground_resized_cropped = process_foreground(foreground, new_width, target_height)

    # Calculate the position to center the foreground on the background
     
    #the background image is 1600px. the forground's mid should position at 1200px of the image.
    if height > 728 :
        upValue = 150
    elif height < 728 : 
        upValue = 250

    if direction == 1:
      position = (70, upValue) #280
    elif direction == 2:
        newupValue = upValue + 50
        position = (45,newupValue)
    # Paste the foreground onto the background, considering the alpha channel
    background.paste(foreground_resized_cropped, position, foreground_resized_cropped)
    results_directory = './results/'
    ensure_directory_exists(results_directory)
    output_filename = get_unique_filename(results_directory, 'output', 'png')
    # Save the final combined image
    print(output_filename)
    log_to_json(f"saved combined image {output_filename}", current_file_name)
    background.save(f'./results/{output_filename}')
    outputfile = blur_license_plate(f'./results/{output_filename}')
    result_string = f'./results/{output_filename}' # return this string
    return output_filename


def rotate_image_with_transparency(image, angle):
    # Separate the color and alpha channels
    if image.shape[2] == 4:
        bgr = image[:, :, :3]
        alpha = image[:, :, 3]

        # Rotate the color channels
        center = (bgr.shape[1] // 2, bgr.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_bgr = cv2.warpAffine(bgr, rotation_matrix, (bgr.shape[1], bgr.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)

        # Rotate the alpha channel
        rotated_alpha = cv2.warpAffine(alpha, rotation_matrix, (alpha.shape[1], alpha.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)

        # Merge the rotated color channels with the rotated alpha channel
        rotated_image = np.dstack([rotated_bgr, rotated_alpha])
    else:
        # If there's no alpha channel, simply rotate the image
        center = (image.shape[1] // 2, image.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
    
    image_no_bg = remove(rotated_image)
    return image_no_bg


def process_foreground(foreground, new_width, target_height):
    # Resize the foreground to maintain aspect ratio
    aspect_ratio = foreground.height / foreground.width
    new_height = int(new_width * aspect_ratio)
    foreground_resized = foreground.resize((new_width, new_height), Image.LANCZOS)

    # Now crop the resized image if it's taller than target_height (1000px in this case)
    if new_height > target_height:
        bottom_crop = target_height  # Keep the top part and crop from the bottom
        foreground_cropped = foreground_resized.crop((0, 0, new_width, bottom_crop))
    else:
        # If the height is already less than or equal to target_height, no cropping is necessary
        foreground_cropped = foreground_resized

    return foreground_cropped