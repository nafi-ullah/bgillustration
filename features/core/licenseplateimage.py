from my_logging_script import log_to_json
import requests
current_file_name = "licenseplateimage.py"
from io import BytesIO
from PIL import Image, ImageOps


def licenseplateimage(image_file):
    log_to_json("Got the imagefile in licenseplateimage function.", current_file_name)

    


  
        # Load the transparent image from the response content
    log_to_json(f"Image added in the license plate", current_file_name)
        # print(f"Remove.bg done")
    image = Image.open(BytesIO(image_file.content)).convert("RGBA")
    return {"image": image, "status": "success"}