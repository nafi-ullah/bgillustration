from my_logging_script import log_to_json
import requests
current_file_name = "polishing.py"
from io import BytesIO
from PIL import Image, ImageOps, ImageEnhance


def democar(image_file):
    log_to_json("Got the imagefile in polishingcar function.", current_file_name)

  
        # Load the transparent image from the response content
    log_to_json(f"Polishing Car Done", current_file_name)
        # print(f"Remove.bg done")
    image = Image.open(BytesIO(image_file.content)).convert("RGBA")
    return {"image": image, "status": "success"}




def polish_car(image_input, brightness_factor=1.1):
    try:
    # Ensure the image is in RGBA format (to handle transparency)
        if isinstance(image_input, BytesIO):
            image_input.seek(0)  # Ensure the stream is at the start
            image = Image.open(image_input)
        elif isinstance(image_input, Image.Image):
            image = image_input
        else:
            raise ValueError("Unsupported input type. Provide a PIL.Image.Image or BytesIO object.")

        # Ensure the image is in RGBA format (to handle transparency)
        image = image.convert("RGBA")
        
        # Split the alpha channel if transparency exists
        r, g, b, alpha = image.split()
        
        # Adjust the brightness for RGB channels only
        enhancer = ImageEnhance.Brightness(Image.merge("RGB", (r, g, b)))
        brightened_image = enhancer.enhance(brightness_factor)
        
        # Merge back the alpha channel to maintain transparency
        final_image = Image.merge("RGBA", (*brightened_image.split(), alpha))
        
        # Save the result to BytesIO
        img_byte_arr = BytesIO()
        final_image.save(img_byte_arr, format="PNG")  # Save as PNG
        img_byte_arr.seek(0) # Reset file pointer to the start
        
        return img_byte_arr
    except Exception as e:
        print("Error uploading file to S3:", e)
        return None
