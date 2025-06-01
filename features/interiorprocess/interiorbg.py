import os
from flask import Flask, request, jsonify
from datetime import datetime
from PIL import Image, ImageOps
import requests
import tempfile
import imageio.v3 as iio
from io import BytesIO
import boto3
from botocore.client import Config
import cv2
import numpy as np
from rembg import remove
from features.laravelconnection.processdoneapi import notify_error_status, notify_success_status, notify_background_success_status, notify_background_error_status
from features.functionalites.backgroundoperations.addbackground.addwallinitiate import addBackgroundInitiate, addBasicBackgroundInitiate
from dynamicfuncs import save_image_with_timestamp
from backgroundconfs import get_wall_images_from_urls, get_logo_image , get_wall_coordinates, get_basic_wall_coordinates, get_basic_wall_images_from_urls
from my_logging_script import log_to_json
from config.configfunc import read_api_key
from dynamicfuncs import is_feature_enabled, get_user_setting, save_image_with_timestamp, getLicensePlateImage, get_blur_intensity
from config.configfunc import read_config, write_config, read_config_bool, write_api_key, read_api_key
from features.functionalites.addBackground import add_gray_background_solo, dynamic_interior_images
from features.laravelconnection.uploadImageBucket import upload_to_s3
from features.functionalites.imageoperations.typeconversion import convert_image_to_bytesio, convert_image_to_png
from features.functionalites.imageoperations.basic import resize_image_bytesio, paste_foreground_on_background_bytesio, blur_image_bytesio, resize_1920_image, get_image_dimensions
UPLOAD_FOLDER = './outputs/uploaded_images/'
OUTPUT_DIR = './outputs/bgremoved/'
RESULT_DIR = './outputs/results/'
# REMOVE_BG_API_KEY = 'cyQmYASY66Q6r8ZwooTHJykQ'
REMOVE_BG_API_KEY = read_api_key()
# backup key: VzvXU5f1pGK8KWJeenw2LVnt
access_key = '4IUMPGRCHXNLAJ5EUUGP'
secret_key = 'LakMU0aI4zTtEd3dzxr2LUD5R9EgvEMtBTgf7ukd'
region = 'eu-de'
bucket_name = 'vroomview'
endpoint_url = 'https://obs.eu-de.otc.t-systems.com'
get_url = 'https://vroomview.obs.eu-de.otc.t-systems.com'

current_file_name = "interiorbg.py"

s3_client = boto3.client(
    's3',
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key,
    region_name=region,
    endpoint_url=endpoint_url,
    config=Config(s3={'addressing_style': 'path'})
)

def save_image(image_file):
    """Saves the uploaded image and returns the file path."""
    timestamp = datetime.now().strftime("%d-%m-%Y-%I%M%p")
    file_ext = os.path.splitext(image_file.filename)[1]
    new_filename = f"{os.path.splitext(image_file.filename)[0]}_{timestamp}{file_ext}"
    image_path = os.path.join(UPLOAD_FOLDER, new_filename)
    image_file.save(image_path)
    return image_path


global_user_setting_data= []


def remove_bg_add_bg(image_url, userid, global_user_setting_data, catalogue_feature_list):
    set_interior_crop_type = get_user_setting(global_user_setting_data, userid, "Interior Crop Type")
    set_blur_intesity = get_blur_intensity(get_user_setting(global_user_setting_data, userid, "Blur"))


    try:
    # Download the image
        response = requests.get(image_url)
        url_parts = image_url.split('/')
        catalogue_id = url_parts[-3]  # Example: 1001
        angle_id = url_parts[-2]      # Example: 7
        filename = url_parts[-1]  
        if response.status_code != 200:
            notify_error_status(catalogue_id, angle_id, filename, f"Failed to get the selected image.", current_file_name)
            return None


        # Open the image from the downloaded content
        log_to_json(f"Got the image by downloading: {image_url}", current_file_name)
        image_file = BytesIO(response.content)

        # -------------Remove the background from the image-------------
        #result = remove_background_rembg(image_file)
        rem_config = read_config_bool()
        if rem_config == True:
            result = remove_background_premium(image_file, for_type='auto')
            print(f"Remove car bg premium {rem_config}")
        else: 
            result = remove_background(image_file)
            print(f"Remove car bg python package {rem_config}")
        
        if result["status"] == "failed":
            # Notify failure
            # log_to_json(f"Fix the following error: {result["error"]}", current_file_name)
            # notify_error_status(catalogue_id, angle_id, filename, "Foreground object not detected." , current_file_name, notify=1)
            resized_transparent = image_file
        else:
            transparent_image = convert_image_to_bytesio(result["image"])
            print(type(transparent_image))
            resized_transparent = resize_1920_image(transparent_image)

        #----- blur background--------
        if set_interior_crop_type == 'blur_crop':
            blurred_bg = blur_image_bytesio(image_file, set_blur_intesity)
        else:
            blurred_bg = None


        #------------------- Background removal was successful; proceed with adding gray background and upload
        
        added_bg_image = dynamic_interior_images(blurred_bg, resized_transparent)
        print(f"tyoe if save image {type(added_bg_image)}")
        #final_image_link = add_gray_background(transparent_image, filename, catalogue_id, angle_id)
        final_image_link = save_image_to_s3(added_bg_image, filename, catalogue_id, angle_id)
        
        if final_image_link:
            log_to_json(f"Processed and uploaded image: {final_image_link}", current_file_name)

            # Extract filename from final_image_link
            processed_filename = final_image_link.split('/')[-1]
            notify_success_status(catalogue_id, angle_id, processed_filename, current_file_name)

        # else:
        #     # Notify failure if upload to S3 fails
        #     notify_error_status(catalogue_id, angle_id, filename, "Upload to bucket failed", current_file_name)

        return final_image_link

    except FileNotFoundError as e:
        log_to_json(f"File not found error {e}", current_file_name)
        notify_error_status(catalogue_id, angle_id, filename, f"System error occurred. It will be available soon.", current_file_name)
        return f'Sorry the image is not processed for: File not found: {str(e)}'
    except ValueError as e:
        log_to_json(f"ValueError {e}", current_file_name)
        notify_error_status(catalogue_id, angle_id, filename, f"System error occurred. It will be available soon.", current_file_name)
        return f'Sorry the image is not processed for: Value error: {str(e)}'
    except Exception as e:
        log_to_json(f"Exceptional error {e}", current_file_name)
        notify_error_status(catalogue_id, angle_id, filename, f"System error occurred. It will be available soon.", current_file_name)
        return  f'Sorry the image is not processed for:  An unexpected error occurred: {str(e)}'




def addExtraSizebottom(image_bytes_io: BytesIO, extra_pixel: int) -> BytesIO:
    print("addExtraSizebottom has received the image")
    try:
        # Open the image from BytesIO
        image = Image.open(image_bytes_io).convert("RGBA")
        
        # Get image dimensions
        original_width, original_height = image.size
        
        # Create a new transparent canvas
        new_height = original_height + extra_pixel
        canvas = Image.new("RGBA", (original_width, new_height), (255, 255, 255, 0))
        
        # Paste the original image onto the canvas at (0, 0)
        canvas.paste(image, (0, 0))
        
        # Save the resulting image to BytesIO
        result_bytes_io = BytesIO()
        canvas.save(result_bytes_io, format="PNG")
        result_bytes_io.seek(0)
        
        return result_bytes_io
    
    except Exception as e:
        log_to_json(f"Image resize is not possible in addExtraSizebottom function because {e}")
        return image_bytes_io 

import requests
import base64
from io import BytesIO
from PIL import Image

def remove_background_premium(image_file, for_type='car'):
    log_to_json("Got the image file in remove_background_premium function. image type {type(image_file)}", current_file_name)
    
    
    try:
        REMOVE_BG_API_KEY = read_api_key()
        response = requests.post(
            'https://api.remove.bg/v1.0/removebg',
            files={'image_file': image_file},
            data={'type': for_type, 'size': 'auto', 'shadow_type': 'car', 'crop': 'true'},
            headers={
                'X-Api-Key': REMOVE_BG_API_KEY,
                'Accept': 'application/json'
            },
        )

        if response.status_code == 200:
            # Parse the JSON response.
            json_response = response.json()
            data = json_response.get("data", {})

            # Get the foreground values
            foreground_top = data.get("foreground_top", 0)
            foreground_left = data.get("foreground_left", 0)
            foreground_width = data.get("foreground_width", 0)
            foreground_height = data.get("foreground_height", 0)

            # Print the foreground coordinates
            log_to_json("Foreground Coordinates: top: {}, left: {}, width: {}, height: {}".format(
                foreground_top, foreground_left, foreground_width, foreground_height
            ))
            log_to_json(
                f"remove_background_premium: Remove.bg done. Crop Coordinates -> Top: {foreground_top}, Left: {foreground_left}, Width: {foreground_width}, Height: {foreground_height}",
                current_file_name
            )

            # Get the base64 encoded image result and decode it
            result_b64 = data.get("result_b64")
            if result_b64:
                image_bytes = base64.b64decode(result_b64)
                bg_removed_image = Image.open(BytesIO(image_bytes)).convert("RGBA")
            else:
                raise ValueError("No result_b64 found in response.")

            # Optional: Resize image to match original dimensions
            input_image = Image.open(image_file)
            original_width, original_height = input_image.size
            aspect_ratio = bg_removed_image.height / bg_removed_image.width
            new_height = int(original_width * aspect_ratio)
            # Optionally, perform resizing here if desired:
            # bg_removed_image = bg_removed_image.resize((original_width, new_height), Image.LANCZOS)

            # Return both the processed image and the foreground details.
            return {
                "image": bg_removed_image,
                "status": "success",
                "foreground": { 
                    "top": foreground_top,
                    "left": foreground_left,
                    "width": foreground_width,
                    "height": foreground_height
                }
            }

        elif response.status_code in {400, 402, 403, 429}:
            error_details = response.json().get("errors", [])
            error_message = ", ".join([error.get("title", "Unknown error") for error in error_details])
            log_to_json(f"Remove.bg API failed: {REMOVE_BG_API_KEY} {response.status_code} - {error_message}", current_file_name)
            return {"status": "failed", "error": f"{response.status_code}: {error_message}"}

        else:
            log_to_json(f"Remove.bg API failed: {REMOVE_BG_API_KEY} {response.status_code} - Unexpected error", current_file_name)
            return {"status": "failed", "error": f"Unexpected error: {response.status_code}"}

    except requests.RequestException as e:
        log_to_json(f"Remove.bg API failed: {REMOVE_BG_API_KEY} Network error - {str(e)}", current_file_name)
        return {"status": "failed", "error": "Network error: " + str(e)}

    except Exception as e:
        log_to_json(f"Remove.bg API failed: {REMOVE_BG_API_KEY} Unknown error - {str(e)}", current_file_name)
        return {"status": "failed", "error": "Unknown error: " + str(e)}

    


def add_extra_size_bottom(image_file: BytesIO, extra_pixel: int) -> BytesIO:
  
    try:
        # Open the original image
        image = Image.open(image_file).convert("RGBA")
        
        # Get original dimensions
        original_width, original_height = image.size
        
        # Calculate the new dimensions for the resized image
        new_image_height = original_height - extra_pixel
        aspect_ratio = original_width / original_height
        new_image_width = int(new_image_height * aspect_ratio)
        
        # Resize the image
        resized_image = image.resize((new_image_width, new_image_height), Image.Resampling.LANCZOS)
        
        # Create a new canvas with the original dimensions and a white background
        canvas = Image.new("RGBA", (original_width, original_height), (255, 255, 255, 255))
        
        # Center the resized image on the canvas
        x_offset = (original_width - new_image_width) // 2
        y_offset = (original_height - new_image_height) // 2
        canvas.paste(resized_image, (x_offset, 0), mask=resized_image)  # Use alpha mask
        
        # Save the modified image to a new BytesIO object
        result_image = BytesIO()
        canvas.save(result_image, format="PNG")
        result_image.seek(0)  # Reset stream position
        return result_image
    
    except Exception as e:
        log_to_json(f"Error resizing image: {e}")
        raise e



def remove_background_premium_extrasizing(image_file: BytesIO):
    REMOVE_BG_API_KEY = read_api_key()
    log_to_json("Received the image file in remove background function.", current_file_name)
    print(f"Remove background premium received the image type: {type(image_file)}")

    # Add extra size to the bottom
    try:
        image_with_extra_size = add_extra_size_bottom(image_file, 150)

        # Ensure the stream is reset
        image_with_extra_size.seek(0)

        # Send the modified image to the Remove.bg API
        response = requests.post(
            "https://api.remove.bg/v1.0/removebg",
            files={"image_file": ("image.png", image_with_extra_size, "image/png")},
            data={"size": "auto", "shadow_type": "car"},
            headers={"X-Api-Key": REMOVE_BG_API_KEY},
        )

        # Handle the API response
        if response.status_code == 200:
            log_to_json("Remove.bg API call succeeded.", current_file_name)

            # Load and return the processed image
            bg_removed_image = Image.open(BytesIO(response.content)).convert("RGBA")
            return {"image": bg_removed_image, "status": "success"}

        else:
            # Handle specific error responses
            error_details = response.json().get("errors", [])
            error_message = ", ".join([error.get("title", "Unknown error") for error in error_details])
            log_to_json(f"Remove.bg API failed: {REMOVE_BG_API_KEY} {response.status_code} - {error_message}", current_file_name)
            return {"status": "failed", "error": f"{response.status_code}: {error_message}"}

    except requests.RequestException as e:
        log_to_json(f"Remove.bg API failed: {REMOVE_BG_API_KEY} Network error - {e}", current_file_name)
        return {"status": "failed", "error": f"Network error: {e}"}

    except Exception as e:
        log_to_json(f"Remove.bg API failed: {REMOVE_BG_API_KEY} Unknown error - {e}", current_file_name)
        return {"status": "failed", "error": f"Unknown error: {e}"}
    

def remove_background(image_file):
    try:
        log_to_json("Got the image file in removebackground function.", current_file_name)

        # Read the image content and process it with rembg
        image_bytes = image_file.read()
        output_bytes = remove(image_bytes)

        # Load the transparent image from the processed bytes
        image = Image.open(BytesIO(output_bytes)).convert("RGBA")

        log_to_json("Background removed successfully using rembg.", current_file_name)

        return {"image": image, "status": "success"}
    except Exception as e:
        log_to_json(f"rembg python processing failed: {str(e)}", current_file_name)
        return {"status": "failed", "error": str(e)}



def add_gray_background(image, original_filename, catalogue_id, angle_id):
    """Adds a gray background to the transparent image and uploads it to the S3 bucket."""
    base_filename = os.path.splitext(original_filename)[0].replace("_original", "")
    result_filename = f"{base_filename}_processed.png"
    # Set gray background color
    gray_color = (184, 183, 179)
    
    # Create a new image with the gray background
    bg_image = Image.new("RGBA", image.size, gray_color)
    bg_image.paste(image, (0, 0), image)
    
    # Convert image to RGB (without alpha) for uploading to S3
    rgb_image = bg_image.convert("RGB")
    
    # Create in-memory file
    in_memory_file = BytesIO()
    rgb_image.save(in_memory_file, format="PNG")
    in_memory_file.seek(0)  # Reset file pointer to start

    # Format the S3 object path
    timestamp = datetime.now().strftime("%d-%m-%Y-%I%M%p")
    # result_filename = f"{os.path.splitext(original_filename)[0]}_processed.png"
    s3_object_name = f"uploads/{catalogue_id}/{angle_id}/{result_filename}"

    # Upload to S3
    try:
        s3_client.upload_fileobj(in_memory_file, bucket_name, s3_object_name)
        final_image_link = f"{get_url}/{s3_object_name}"
        # print(f"Uploaded processed image to S3: {final_image_link}")
        log_to_json(f"Uploaded angleid {angle_id} processed image to S3: {final_image_link}", current_file_name)
        return final_image_link
    except Exception as e:
        #print("Error uploading file to S3:", e)
        log_to_json(f"Error uploading file to S3:: {e}", current_file_name)
        # Notify failure via API
        notify_response = notify_error_status(catalogue_id, angle_id, original_filename, f"Upload to bucket failed", current_file_name)
   
    

    

def only_save(image, original_filename, catalogue_id, angle_id):
    """Uploads the given image directly to the S3 bucket."""
    
    # Remove "_original" from filename if it exists and add "_processed"
    base_filename = os.path.splitext(original_filename)[0].replace("_original", "")
    result_filename = f"{base_filename}_processed.png"

    # Create in-memory file
    in_memory_file = BytesIO()
    image.save(in_memory_file, format="PNG")
    in_memory_file.seek(0)  # Reset file pointer to start

    # Format the S3 object path
    s3_object_name = f"uploads/{catalogue_id}/{angle_id}/{result_filename}"

    # Upload to S3
    try:
        s3_client.upload_fileobj(in_memory_file, bucket_name, s3_object_name)
        final_image_link = f"{get_url}/{s3_object_name}"
        print(f"Uploaded image to S3: {final_image_link}")
        return final_image_link
    except Exception as e:
        print("Error uploading file to S3:", e)
        return None
    
def no_process_and_save(image, original_filename, catalogue_id, angle_id):
    
    try:
        final_image_link = f"{get_url}"
        print(f"Uploaded image to S3: {final_image_link}")
        return final_image_link
    except Exception as e:
        print("Error uploading file to S3:", e)
        return None
    


def save_image_to_s3_image_path(image_path, filename, catalogue_id, angle_id):
    # Open the image from the local path
    base_filename = os.path.splitext(filename)[0].replace("_original", "")
    result_filename = f"{base_filename}_processed.png"
    try:
        with open(image_path, 'rb') as image_file:
            img_byte_arr = BytesIO(image_file.read())  # Read the image into memory as a byte array

        # S3 object name format
        s3_object_name = f"uploads/{catalogue_id}/{angle_id}/{result_filename}"

        # Upload to S3
        s3_client.upload_fileobj(img_byte_arr, bucket_name, s3_object_name)
        final_image_link = f"{get_url}/{s3_object_name}"
        # print(f"Uploaded processed image to S3: {final_image_link}")
        log_to_json(f"Uploaded processed image to S3: {final_image_link}", current_file_name)
        return final_image_link
    except Exception as e:
        # print(f"Error uploading file to S3: {e}")
        log_to_json(f"Error uploading file to S3: {e}", current_file_name)
        notify_response = notify_error_status(catalogue_id, angle_id, filename, f"Upload to bucket failed", current_file_name)

    return None


def compress_image(image_bytesio, target_size_kb=1000, initial_quality=85, step=10):

    # Open the image from BytesIO
    image = Image.open(image_bytesio)

    # Create a BytesIO object to store the compressed image
    output_bytesio = BytesIO()

    # Initialize quality
    quality = initial_quality

    while True:
        # Clear the BytesIO object for each iteration
        output_bytesio.seek(0)
        output_bytesio.truncate(0)

        # Save the image with the current quality using imageio
        iio.imwrite(output_bytesio, image, format='JPEG', quality=quality)

        # Check the size of the compressed image
        size_kb = output_bytesio.tell() / 1024
        print(f"size == {size_kb}")  # Debugging: Print the current size

        if size_kb <= target_size_kb:
            break  # Exit the loop if the target size is achieved

        # Reduce the quality for the next iteration
        quality -= step

        # Stop if quality drops below a reasonable threshold
        if quality <= 10:
            break

    # Seek to the beginning of the BytesIO object
    output_bytesio.seek(0)
    # save_image_with_timestamp(output_bytesio, 'outputs/dumtest/outputdebug/finalimages', 'compressed_image.png')
    return output_bytesio






def save_image_to_s3(image_bytes_io, filename, catalogue_id, angle_id):

    # Prepare the processed filename
    image_bytes_io = compress_image(image_bytes_io)
    
    base_filename = os.path.splitext(filename)[0].replace("_original", "")
    result_filename = f"{base_filename}_processed.jpg"

    try:
        # Convert BytesIO image to a normal image using Pillow
        # image = Image.open(image_bytes_io).convert("RGBA")
        temp_image_path = f"/tmp/{result_filename}"
        # image.save(temp_image_path, format="PNG")  # Save the image temporarily
        with open(temp_image_path, "wb") as output_file:
            output_file.write(image_bytes_io.getvalue())

        # Define the S3 object name
        s3_object_name = f"uploads/{catalogue_id}/{angle_id}/{result_filename}"

        # Upload the image to S3
        s3_client.upload_file(temp_image_path, bucket_name, s3_object_name)

        # Remove the temporary file after upload
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

        # Construct the final image link
        final_image_link = f"{get_url}/{s3_object_name}"

        # Log the success
        log_to_json(f"Uploaded processed image to S3: {final_image_link}", current_file_name)

        return final_image_link

    except Exception as e:
        # Log the error
        log_to_json(f"Error uploading file to S3: {e}", current_file_name)

        # Notify about the error
        notify_error_status(catalogue_id, angle_id, filename, f"Upload to bucket failed", current_file_name)

    return None

def save_18_19_image_to_s3(image_bytes_io, filename, catalogue_id, angle_id):
    # image_bytes_io = compress_image(image_bytes_io)
    # Prepare the processed filename
    base_filename = os.path.splitext(filename)[0].replace("_original", "_processed")
    result_filename = f"{base_filename}.jpg"

    try:
        # Convert BytesIO image to a normal image using Pillow
        image = Image.open(image_bytes_io).convert("RGBA")
        temp_image_path = f"/tmp/{result_filename}"
        # image.save(temp_image_path, format="PNG")  # Save the image temporarily
        with open(temp_image_path, "wb") as output_file:
            output_file.write(image_bytes_io.getvalue())

        # Define the S3 object name
        s3_object_name = f"uploads/{catalogue_id}/{angle_id}/{result_filename}"

        # Upload the image to S3
        s3_client.upload_file(temp_image_path, bucket_name, s3_object_name)

        # Remove the temporary file after upload
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

        # Construct the final image link
        final_image_link = f"{get_url}/{s3_object_name}"

        # Log the success
        log_to_json(f"Uploaded processed image to S3: {final_image_link}", current_file_name)

        return final_image_link

    except Exception as e:
        # Log the error
        log_to_json(f"Error uploading file to S3: {e}", current_file_name)

        # Notify about the error
        notify_error_status(catalogue_id, angle_id, filename, f"Upload to bucket failed", current_file_name)

    return None

def save_background_image_to_s3(image_bytes_io, filename, basepath):

    # Prepare the processed filename
    # base_filename = os.path.splitext(filename)[0].replace("_original", "")
    # result_filename = f"{base_filename}_processed.png"

    try:
        # Convert BytesIO image to a normal image using Pillow
        image = Image.open(image_bytes_io).convert("RGBA")
        temp_image_path = f"/tmp/{filename}"
        # image.save(temp_image_path, format="PNG")  # Save the image temporarily
        with open(temp_image_path, "wb") as output_file:
            output_file.write(image_bytes_io.getvalue())

        # Define the S3 object name
        s3_object_name =  f"{basepath}processed/{filename}" #f"uploads/{catalogue_id}/{angle_id}/{result_filename}"

        # Upload the image to S3
        s3_client.upload_file(temp_image_path, bucket_name, s3_object_name)

        # Remove the temporary file after upload
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

        # Construct the final image link
        final_image_link = f"{get_url}/{s3_object_name}"

        # Log the success
        log_to_json(f"Uploaded processed image to S3: {final_image_link}", current_file_name)

        return final_image_link

    except Exception as e:
        # Log the error
        log_to_json(f"Error uploading file to S3: {e}", current_file_name)

        # Notify about the error
        # notify_error_status(catalogue_id, angle_id, filename, f"Upload to bucket failed", current_file_name)

    return None


def process_images(image_urls, bg_type):
    # Simulate processing of images
    try:
        print(f"processing images {image_urls} and bgtype {bg_type}")
        logo_bytesio = None
        if bg_type == 'photo_box':
            floor_coordinates, left_wall_coordinates, right_wall_coordinates, ceiling_coordinates = get_wall_coordinates()
            wallImages_bytes = get_wall_images_from_urls(image_urls)
            processed_bg = addBackgroundInitiate(wallImages_bytes, logo_bytesio, floor_coordinates, left_wall_coordinates, right_wall_coordinates, ceiling_coordinates, angle='normal')
        elif bg_type == 'basic':
            coord = {
                "top_x": 0,
                "top_y": 850
            }
            floor_coordinates, wall_coordinates = get_basic_wall_coordinates(coord)
            wallImages_bytes = get_basic_wall_images_from_urls(image_urls)
            processed_bg = addBasicBackgroundInitiate(wallImages_bytes, logo_bytesio, floor_coordinates, wall_coordinates)

        
        
        
        #save_image_with_timestamp(processed_bg, 'outputs/dumtest/outputdebug/finalimages', 'bg_image.png')


        return processed_bg
    except Exception as e:
        log_to_json(f"Exceptional error {e}", current_file_name)
        return  f'Sorry : An unexpected error occurred: {str(e)}'

def handle_configuration(user_code, bg_type, counter):
    bg_file_name = None 
    try:
        base_path = f"user/{user_code}/bg_type/{bg_type}/"
        print(f"Got the req in handle configuration: usercode {user_code} bg_type: {bg_type} counter: {counter}")
        
        # Determine required files based on bg_type
        if bg_type == "photo_box":
            required_files = {
                f"lw_{counter}",
                f"rw_{counter}",
                f"floor_{counter}",
                f"ceiling_{counter}"
            }
        elif bg_type == "basic":
            required_files = {
                f"lw_{counter}",
                f"floor_{counter}"
            }
        else:
            return {"error": f"Unsupported bg_type: {bg_type}"}, 400

        # List objects in the S3 bucket
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=base_path)
        # print(response)
        if 'Contents' not in response:
            return {"error": "No files found in the specified path"}, 404

        # Filter for specific files based on the counter
        files = [obj['Key'] for obj in response['Contents']]
        found_files = {}
        for file in files:
            filename = os.path.basename(file)
            name, ext = os.path.splitext(filename)

            # Prioritize JPG files if both JPG and PNG exist
            if name in required_files:
                if name not in found_files or ext.lower() == ".jpg":
                    found_files[name] = ext.strip(".")

        # Check if all required files are found
        if len(found_files) != len(required_files):
            return {"error": "Incomplete set of images for processing"}, 400

        # Construct URLs for the found files
        images = {
            name: f"{get_url}/{base_path}{name}.{ext}"
            for name, ext in found_files.items()
        }
        print(images)

        # Process the images and get the resulting filename
        result_file_bytesio = process_images(images, bg_type)
        bg_file_name = f"{bg_type}_{counter}.jpg"
        full_path = f"{base_path}processed/{bg_file_name}"
        final_image_link = save_background_image_to_s3(result_file_bytesio, bg_file_name, base_path)
        log_to_json(f"Done bg processed, save the bg in {final_image_link}", current_file_name)
        notify_background_success_status(user_code, counter, bg_type, full_path, current_file_name)
        # Create and return the response
        response_body = {
            "pictures": user_code,
            "bg_type": bg_type,
            "counter": counter,
            "final_image_link": final_image_link
        }
        return response_body, 200

    except Exception as e:
        notify_background_error_status(user_code, counter, bg_type, full_path, f"error: {e}",current_file_name)
        return {"error": str(e)}, 500





