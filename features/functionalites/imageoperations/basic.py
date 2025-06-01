from io import BytesIO
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps

def resize_image_bytesio(resizable_image, resize_type, size):
    try:
        # Read the image bytes from the BytesIO stream
        img_array = np.frombuffer(resizable_image.read(), dtype=np.uint8)
        
        # Use IMREAD_UNCHANGED to load all channels (including alpha channel if present)
        img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)

        # Check if the image was loaded successfully
        if img is None:
            raise ValueError("Could not load image from BytesIO.")

        # Get original dimensions
        original_height, original_width = img.shape[:2]

        # Calculate new dimensions based on the resize_type
        if resize_type == "width":
            new_width = size
            aspect_ratio = original_height / original_width
            new_height = int(new_width * aspect_ratio)
        elif resize_type == "height":
            new_height = size
            aspect_ratio = original_width / original_height
            new_width = int(new_height * aspect_ratio)
        else:
            raise ValueError("Invalid resize_type. Must be 'width' or 'height'.")

        # Resize the image (all channels including alpha will be resized)
        resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

        # Optionally log if the image contains transparency
        if len(resized_img.shape) == 3 and resized_img.shape[2] == 4:
            print("Resized image retains transparency (alpha channel present).")
        else:
            print("Resized image does not have an alpha channel.")

        # Encode the resized image to a PNG format (which supports transparency)
        is_success, buffer = cv2.imencode(".png", resized_img)
        if not is_success:
            raise ValueError("Error encoding resized image to buffer.")

        image_bytes_io = BytesIO(buffer)
        image_bytes_io.seek(0)  # Reset the BytesIO position
        return {"retruned_image": image_bytes_io, "returnend_height": new_height}

    except Exception as e:
        print(f"Error during resizing: {e}")
        return None


def add_border_to_image(image_bytesio, sides, width, color="#000000"):
   
    # Load the image from BytesIO
    image = Image.open(image_bytesio)

    # Initialize border dictionary with 0 values
    border = {'left': 0, 'right': 0, 'top': 0, 'bottom': 0}

    # Map sides to their respective border values
    side_keys = ['right', 'left', 'top', 'bottom']
    for i, side in enumerate(sides):
        border[side_keys[i]] = width if side else 0

    # Add border to the image using ImageOps.expand
    bordered_image = ImageOps.expand(image, border=(border['left'], border['top'], border['right'], border['bottom']), fill=color)

    # Save the result back to a BytesIO object
    output = BytesIO()
    bordered_image.save(output, format=image.format)
    output.seek(0)

    return output

    # Add borders: 10px on right and left sides, no top or bottom, red color
    # sides = [1, 1, 0, 0]  # right, left, top, bottom
    # border_width = 10
    # border_color = "#FF0000"  # Red color in hex

    # result_image = add_border_to_image(img_bytes, sides, border_width, border_color)



def get_image_dimensions(image_bytes: BytesIO, message: str) -> tuple:
 
    try:
        # Open the image from the BytesIO object
        image = Image.open(image_bytes)
        # Get width and height
        width, height = image.size
        print(f"{message}: {width}, {height}")
        return width, height
    except Exception as e:
        raise ValueError(f"Unable to process the image. Error: {e}")

def blur_image_bytesio(image_bytes_io: BytesIO, blur_intensity: int) -> BytesIO:
 
    try:
        # Ensure blur_intensity is an odd number >= 3
        if blur_intensity < 3 or blur_intensity % 2 == 0:
            raise ValueError("Blur intensity must be an odd integer greater than or equal to 3.")

        # Convert BytesIO to PIL Image
        image = Image.open(image_bytes_io).convert("RGBA")

        # Apply Gaussian blur using ImageFilter
        blurred_image = image.filter(ImageFilter.GaussianBlur(blur_intensity // 2))

        # Save the blurred image to BytesIO
        blurred_image_bytes_io = BytesIO()
        blurred_image.save(blurred_image_bytes_io, format="PNG")
        blurred_image_bytes_io.seek(0)  # Reset the BytesIO object position

        return blurred_image_bytes_io

    except Exception as e:
        print(f"Error during blurring: {e}")
        return None


def paste_foreground_on_background_bytesio(background_bytesio: BytesIO, foreground_bytesio: BytesIO, x: int, y: int) -> BytesIO:
    try:
    # Load images from BytesIO
        background_image = Image.open(background_bytesio).convert("RGBA")
        foreground_image = Image.open(foreground_bytesio).convert("RGBA")

        # Create a blank canvas with the same size as the background
        combined_image = Image.new("RGBA", background_image.size)

        # Paste the background onto the blank canvas
        combined_image.paste(background_image, (0, 0))

        # Paste the foreground image onto the background at the specified position
        combined_image.paste(foreground_image, (x, y), mask=foreground_image)

        # Save the resulting image to BytesIO
        result_bytesio = BytesIO()
        combined_image.save(result_bytesio, format="PNG")
        result_bytesio.seek(0)  # Reset the stream position

        return result_bytesio
    except Exception as e:
        print(f"Error during blurring: {e}")
        return None

def resize_1920_image(image_bytesio: BytesIO) -> BytesIO:
    try:
        print("Resized image to 1920 function got it")

        # Read image data from BytesIO
        img_array = np.frombuffer(image_bytesio.read(), dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)  # Use IMREAD_UNCHANGED to keep alpha channel

        if image is None:
            raise ValueError("Failed to decode the image. Ensure it's a valid image format.")

        # Check if the image has an alpha channel (transparency)
        has_alpha = image.shape[2] == 4 if len(image.shape) == 3 else False

        # Get original dimensions
        original_height, original_width = image.shape[:2]

        # Resize the width to 1920 and maintain aspect ratio for height
        width = 1920
        height = int((original_height / original_width) * width)

        # If the height after resizing is less than 1440, adjust the height to 1440 and width accordingly
        if height < 1440:
            height = 1440
            width = int((original_width / original_height) * height)

        # Resize the image using cv2
        resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

        # If the image has transparency (alpha channel), ensure the alpha channel is preserved
        if has_alpha:
            # Ensure the image stays in RGBA mode if transparency exists
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2BGRA)

        # Encode the resized image back to BytesIO
        _, buffer = cv2.imencode('.png', resized_image)
        output = BytesIO(buffer.tobytes())  # Store the image data in a BytesIO object
        output.seek(0)  # Reset the cursor to the beginning of the stream

        print("Done resizing")
        return output

    except Exception as e:
        print("Sorry, the image could not be resized.")
        error_message = f"Sorry, the image is not processed due to: {str(e)}"
        return BytesIO(error_message.encode())

