from my_logging_script import log_to_json
import requests
current_file_name = "addBackground.py"
from io import BytesIO
from PIL import Image, ImageOps
import os
from datetime import datetime
import boto3
from botocore.client import Config


UPLOAD_FOLDER = './outputs/uploaded_images/'
OUTPUT_DIR = './outputs/bgremoved/'
RESULT_DIR = './outputs/results/'
REMOVE_BG_API_KEY = 'cyQmYASY66Q6r8ZwooTHJykQ'
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

def addBackgroundImagefunc(image_file):
    log_to_json("Got the imagefile in add background function.", current_file_name)

    


  
        # Load the transparent image from the response content
    log_to_json(f"added background in the license plate", current_file_name)
        # print(f"Remove.bg done")
    image = Image.open(BytesIO(image_file.content)).convert("RGBA")
    return {"image": image, "status": "success"}

def save_canvas_state(canvas: Image.Image, filename: str, save_path: str = "outputs/canvas_states"):
    """Saves the current state of the canvas to a local folder."""
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_path = os.path.join(save_path, filename)
    canvas.save(file_path, format="PNG")
    print(f"Canvas state saved at {file_path}")

def dynamic_interior_images(background_image: BytesIO = None, foreground_image: BytesIO = None) -> BytesIO:
    print(f"Got the image in dyunamic interior image function {type(background_image)}  {type(foreground_image)}")
    try:
        # Create a 1920x1440 canvas with the specified gray color
        gray_color = (184, 183, 179)
        
        # Open the foreground image
        fg = Image.open(foreground_image)
        fg_width, fg_height = fg.size
        
        # Resize the foreground image to fit within 1920x1440 while maintaining aspect ratio
        aspect_ratio = fg_width / fg_height
        new_width = 1920
        new_height = int(new_width / aspect_ratio)
        
        # Adjust height if it exceeds the canvas height
        if new_height > 1440:
            new_height = 1440
            new_width = int(new_height * aspect_ratio)

        canvas = Image.new("RGB", (new_width, new_height), gray_color)

        # save_canvas_state(canvas, "after_gray.png")
        # If a background image is provided, open it and paste it onto the canvas
        if background_image:
            bg = Image.open(background_image)
            bg = bg.resize((new_width, new_height), Image.LANCZOS)
            canvas.paste(bg, (0, 0))
            # save_canvas_state(canvas, "after_addbg.png")
        
        
        
        fg_resized = fg.resize((new_width, new_height), Image.LANCZOS)
        
        # Center the resized foreground image on the canvas
        fg_position = (0,0)
        canvas.paste(fg_resized, fg_position, fg_resized.convert('RGBA'))  # Paste with transparency if applicable

        # Save the final image to a BytesIO object
        output = BytesIO()
        canvas.save(output, format="PNG")  # Save image to BytesIO
        output.seek(0)  # Reset the cursor to the beginning of the stream
        print("Dynamic bg added")
        return output

    except Exception as e:
        print(f"Error: {str(e)}")
        return None 


def add_gray_background_solo(image):

    # Set gray background color
    gray_color = (184, 183, 179)

    # Create a new image with the gray background
    bg_image = Image.new("RGBA", image.size, gray_color)
    bg_image.paste(image, (0, 0), image)

    # Convert image to RGB (without alpha) for further processing
    rgb_image = bg_image.convert("RGB")
    return rgb_image

def add_gray_background(image, original_filename, catalogue_id, angle_id):
    try:
        base_filename = os.path.splitext(original_filename)[0].replace("_original", "")
        result_filename = f"{base_filename}_processed.png"
        # Set gray background color
        gray_color = (184, 183, 179)
        
        # Create a new image with the gray background
        bg_image = Image.new("RGBA", image.size, gray_color)
        bg_image.paste(image, (0, 0), image)
        
        # Convert image to RGB (without alpha) for uploading to S3
        rgb_image = bg_image.convert("RGB")

        #-----------
        
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
            try:
                notify_response = requests.post(
                    'http://80.158.3.229:5054/api/process/done',
                    json={
                        "catalogue_id": catalogue_id,
                        "angle_id": angle_id,
                        "filename": original_filename,
                        "status": 0,
                        "message": "Upload to bucket failed"
                    }
                )
                if notify_response.status_code == 200 :
                    # print("Failure notification API called successfully.")
                    log_to_json(f"Failure notification API called successfully.", current_file_name)
                else:
                    # print(f"Failed to notify API: {notify_response.status_code} - {notify_response.text}")
                    log_to_json(f"Failed to notify API: {notify_response.status_code} - {notify_response.text}", current_file_name)
            except Exception as notify_exception:
                # print("Error calling failure notification API:", notify_exception)
                log_to_json(f"Error calling failure notification API:", current_file_name)
            
            return None
        
    except FileNotFoundError as e:
        log_to_json(f"File not found error {e}", current_file_name)
        return f'Sorry the image is not processed for: File not found: {str(e)}'
    except ValueError as e:
        log_to_json(f"ValueError {e}", current_file_name)
        return f'Sorry the image is not processed for: Value error: {str(e)}'
    except Exception as e:
        log_to_json(f"Exceptional error {e}", current_file_name)
        return  f'Sorry : An unexpected error occurred: {str(e)}'