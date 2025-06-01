from my_logging_script import log_to_json
import requests
REMOVE_BG_API_KEY = 'cyQmYASY66Q6r8ZwooTHJykQ'
from PIL import Image, ImageOps
from io import BytesIO
from rembg import remove

current_file_name = "functionalites/removeBackground.py"

def remove_background(image_file):
    log_to_json("Got the imagefile in removebackground function.", current_file_name)
    response = requests.post(
        'https://api.remove.bg/v1.0/removebg',
        files={'image_file': image_file},
        data={'size': 'auto', 'shadow_type': 'car'},
        headers={'X-Api-Key': REMOVE_BG_API_KEY},
    )

    if response.status_code == 200:
        # Load the transparent image from the response content
        log_to_json(f"Remove.bg done", current_file_name)
        # print(f"Remove.bg done")
        image = Image.open(BytesIO(response.content)).convert("RGBA")
        return {"image": image, "status": "success"}
    else:
        # print(f"Remove.bg API failed: {response.status_code} - {response.text}")
        log_to_json(f"Remove.bg API failed: {response.status_code} - {response.text}", current_file_name)
        return {"status": "failed", "error": f"Remove.bg API failed: {response.status_code} - {response.text}"}
    

def remove_background_rembg(image_file):

    log_to_json("Got the image file in remove_background function.", current_file_name)
    
    try:
        # Read the input image
        #input_image = Image.open(image_file).convert("RGBA")
        # Remove background
        output_data = remove(image_file.read())
        output_image = Image.open(BytesIO(output_data)).convert("RGBA")
        
        log_to_json("Background removal using rembg completed successfully.", current_file_name)
        return {"image": output_image, "status": "success"}
    
    except Exception as e:
        log_to_json(f"Background removal failed: {e}", current_file_name)
        return {"status": "failed", "error": str(e)}
