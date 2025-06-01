from my_logging_script import log_to_json
import requests
from PIL import Image, ImageEnhance, ImageFilter, ImageChops
from io import BytesIO
from rembg import remove
import cv2
import numpy as np


current_file_name = "functionalites/carbasicoperations.py"
def image_rotate(bg_image: BytesIO, degree: int) -> BytesIO:
    try:
        # Open the background image from BytesIO
        bg = Image.open(bg_image)

        # Rotate the image by the specified degree
        bg = bg.rotate(degree, expand=True)

        # Save the rotated image to a BytesIO object
        output = BytesIO()
        bg.save(output, format="PNG")
        output.seek(0)

        return output
    except Exception as e:
        log_to_json(f"Error occurs for {e}", current_file_name)
        return None



def crop_image_by_coordinates( image_bytes_io, top, left, width, height):

    try:
        
        # Open the image from the BytesIO stream
        img = Image.open(image_bytes_io)
    except Exception as e:
        log_to_json(f"Error opening image type {type(image_bytes_io)} from BytesIO: {e}")
        return None

    log_to_json(f"crop_image_by_coordinates: got for crop top {top}, left {left}, width {width}, height {height}", current_file_name)
    # Define the box to crop: (left, top, right, bottom)
    box = (left, top, left + width, top + height)
    
    try:
        # Crop the image using the box defined above
        cropped_img = img.crop(box)
    except Exception as e:
        print(f"Error during cropping: {e}")
        return None

    # Save the cropped image to a BytesIO stream, preserving transparency by using the PNG format
    output_bytes = BytesIO()
    try:
        cropped_img.save(output_bytes, format="PNG")
        output_bytes.seek(0)  # Reset the stream position to the beginning
    except Exception as e:
        log_to_json(f"crop_image_by_coordinates: Error saving cropped image: {e}", current_file_name)
        return None

    return output_bytes


def add_padding_to_image(image_bytesio, top, left, right, bottom):
    log_to_json(f"add_padding_to_image: got for padding top {top}, left {left}, right {right}, bottom {bottom}", current_file_name)
    # Open the image
    image = Image.open(image_bytesio).convert("RGBA")

    # Original image size
    original_width, original_height = image.size

    # Calculate resized width and height
    resized_width = original_width - left - right
    if resized_width <= 0:
        raise ValueError("Padding is too large: resulting image width is non-positive.")
    
    aspect_ratio = original_height / original_width
    resized_height = int(resized_width * aspect_ratio)

    # Calculate final canvas size
    canvas_width = resized_width + left + right
    canvas_height = resized_height + top + bottom

    # Resize the image
    image = image.resize((resized_width, resized_height), Image.LANCZOS)

    # Create transparent canvas
    canvas = Image.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 0))

    # Paste resized image onto canvas with padding
    canvas.paste(image, (left, top))

    # Save to BytesIO
    output = BytesIO()
    canvas.save(output, format="PNG")
    output.seek(0)

    return output


def image_flip(bg_image: BytesIO, flip_type: str) -> BytesIO:
    try:
    # Open the background image from BytesIO
        bg = Image.open(bg_image)

        # Apply the requested flip
        if flip_type == "horizontal":
            bg = bg.transpose(Image.FLIP_LEFT_RIGHT)
        elif flip_type == "vertical":
            bg = bg.transpose(Image.FLIP_TOP_BOTTOM)
        else:
            raise ValueError(f"Invalid flip_type: {flip_type}. Use 'horizontal' or 'vertical'.")

        # Save the flipped image to a BytesIO object
        output = BytesIO()
        bg.save(output, format="PNG")
        output.seek(0)

        return output
    except Exception as e:
        log_to_json(f"Image flip failed: {e}", current_file_name)
        return {"status": "failed", "error": str(e)}
    
def right_image_operation(bg_image: BytesIO) -> BytesIO:
    try:
        # Open the background image from BytesIO
        bg = Image.open(bg_image)

        # Rotate the image by the specified degree
        bg = bg.rotate(270, expand=True)
        bg = bg.transpose(Image.FLIP_LEFT_RIGHT)
        # Save the rotated image to a BytesIO object
        output = BytesIO()
        bg.save(output, format="PNG")
        output.seek(0)

        return output 
    except Exception as e:
        log_to_json(f"Error occurs for {e}", current_file_name)
        return None
    
def left_reversewall_image_operation(bg_image: BytesIO) -> BytesIO:
    try:
        # Open the background image from BytesIO
        bg = Image.open(bg_image)

        # Rotate the image by the specified degree
        #bg = bg.rotate(180, expand=True)
        bg = bg.transpose(Image.FLIP_LEFT_RIGHT)
        # Save the rotated image to a BytesIO object
        output = BytesIO()
        bg.save(output, format="PNG")
        output.seek(0)

        return output 
    except Exception as e:
        log_to_json(f"Error occurs for {e}", current_file_name)
        return None

def basic_floor_image_operation(bg_image: BytesIO) -> BytesIO:
    try:
        # Open the background image from BytesIO
        bg = Image.open(bg_image)

        # Rotate the image by the specified degree
        bg = bg.rotate(90, expand=True)
        # bg = bg.transpose(Image.FLIP_LEFT_RIGHT)
        # Save the rotated image to a BytesIO object
        output = BytesIO()
        bg.save(output, format="PNG")
        output.seek(0)

        return output 
    except Exception as e:
        log_to_json(f"Error occurs for {e}", current_file_name)
        return None


def remove_background_rembg(image_file):

    log_to_json("Got the image file in remove_background function.", current_file_name)
    
    try:
        # Read the input image
        image_file.seek(0)
        #input_image = Image.open(image_file).convert("RGBA")
        # Remove background
        output_data = remove(image_file.read())
        output_image = BytesIO(output_data)
        
        log_to_json("Background removal using rembg completed successfully.", current_file_name)
        return {"image": output_image, "status": "success"}
    
    except Exception as e:
        log_to_json(f"Background removal failed: {e}", current_file_name)
        return {"status": "failed", "error": str(e)}
    

def rotate_image_by_angle(image_bytes_io, angle, direction='clockwise'):
    try:
    # Load the image from BytesIO
        image_bytes_io.seek(0)
        img_array = np.frombuffer(image_bytes_io.read(), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            print("Error: Unable to load image from BytesIO.")
            return None

        # Check if the direction is clockwise or counter-clockwise
        if direction == 'counter-clockwise':
            angle = -angle  # Make the angle negative for counter-clockwise rotation

        # Get the image dimensions
        height, width = img.shape[:2]

        # Get the rotation matrix for the specified angle
        rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)

        # Perform the rotation
        rotated_img = cv2.warpAffine(img, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR)

        # Convert the rotated image back to BytesIO format
        _, rotated_img_bytes = cv2.imencode('.png', rotated_img)
        rotated_img_bytes_io = BytesIO(rotated_img_bytes)

        return rotated_img_bytes_io
    except Exception as e:
        log_to_json(f"Error occurs for {e}", current_file_name)
        return None



def create_reflection_effect(car_image_bytesio, opacity_intensity, gaussian_blur_intensity, brightness_intensity):
  

    # Load the car image from BytesIO
    car = Image.open(car_image_bytesio)

    # Convert to RGBA for alpha blending
    car = car.convert('RGBA')

    # Apply opacity reduction
    alpha = car.split()[3]  # Extract alpha channel
    if opacity_intensity < 1:
        # Scale the alpha channel to match the desired opacity intensity
        alpha = ImageEnhance.Brightness(alpha).enhance(opacity_intensity)
        car.putalpha(alpha)

    # Apply gradient mask
    gradient = Image.new('L', (1, car.height), color=255)  # Start with full opacity
    for y in range(car.height):
        gradient.putpixel((0, y), int(255 * (1 - y / car.height)))  # Linear gradient
    gradient = gradient.resize(car.size)

    # Combine gradient with the current alpha channel
    combined_alpha = ImageChops.multiply(car.split()[3], gradient)
    car.putalpha(combined_alpha)

    # Apply Gaussian blur to the reflection
    if gaussian_blur_intensity > 0:
        kernel_size = (3, 3)  # Slight blur
        car_np = np.array(car)
        blurred = cv2.GaussianBlur(car_np, kernel_size, sigmaX=gaussian_blur_intensity)
        car = Image.fromarray(blurred, 'RGBA')

    # Adjust brightness
    if brightness_intensity != 1:
        enhancer_brightness = ImageEnhance.Brightness(car)
        car = enhancer_brightness.enhance(brightness_intensity)

    # Return the processed reflection image as BytesIO
    output_bytesio = BytesIO()
    car.save(output_bytesio, format='PNG')
    output_bytesio.seek(0)
    return output_bytesio



def apply_blur_and_opacity(image_bytes_io, 
                           blur_intensity=0.5, 
                           opacity_intensity=0.9,  # Default to 90% opacity
                           brightness_intensity=0.0):
    try:
        # Read the image from BytesIO using OpenCV
        image_bytes_io.seek(0)
        file_bytes = np.asarray(bytearray(image_bytes_io.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)

        if image is None:
            raise ValueError("Unable to decode the image. Please check the input format.")

        # Ensure image has an alpha channel for opacity control
        if image.shape[2] != 4:  # If not RGBA, add an alpha channel
            alpha_channel = np.full((image.shape[0], image.shape[1]), 255, dtype=np.uint8)  # Full opacity
            image = cv2.merge((image, alpha_channel))

        # Apply Gaussian blur
        if blur_intensity > 0:
            blur_kernel_size = max(1, int(blur_intensity * 30))
            blur_kernel_size = blur_kernel_size if blur_kernel_size % 2 == 1 else blur_kernel_size + 1
            image = cv2.GaussianBlur(image, (blur_kernel_size, blur_kernel_size), 0)

        # Adjust opacity
        if 0 <= opacity_intensity <= 1:
            image[:, :, 3] = (image[:, :, 3] * opacity_intensity).astype(np.uint8)

        # Adjust brightness
        if brightness_intensity != 0:
            brightness_adjustment = int(brightness_intensity * 255)
            rgb_image = image[:, :, :3].astype(np.float32)  # Extract RGB channels
            rgb_image = np.clip(rgb_image + brightness_adjustment, 0, 255).astype(np.uint8)  # Adjust brightness
            image = cv2.merge((rgb_image, image[:, :, 3]))  # Merge back with alpha channel

        # Convert the processed image back to BytesIO format
        _, processed_image_array = cv2.imencode('.png', image)
        processed_image_io = BytesIO(processed_image_array.tobytes())

        return processed_image_io

    except FileNotFoundError as e:
        return f"Error: File not found. Details: {str(e)}"
    except ValueError as e:
        return f"Error: Value error. Details: {str(e)}"
    except Exception as e:
        return f"Error: Unexpected error occurred. Details: {str(e)}"
    

#def add_background_image(bg_image_path, car_image)
    


#  def flip_transparent_image(image_path)
    

def apply_distortion_and_effects(image_bytes_io, 
                                 distortion_intensity=0.1, 
                                 gradient_intensity=0.5, 
                                 color_adjustment_intensity=0.1, 
                                 noise_intensity=0.05):
    try:
    # Load the image from BytesIO
        image_bytes_io.seek(0)
        img_array = np.frombuffer(image_bytes_io.read(), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)

        if img is None:
            print("Error: Unable to load image from BytesIO.")
            return None

        # Apply distortion/warping (Perspective Transform)
        rows, cols = img.shape[:2]
        src_points = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])
        dst_points = src_points + np.float32([[distortion_intensity * cols, distortion_intensity * rows], 
                                            [-distortion_intensity * cols, distortion_intensity * rows],
                                            [distortion_intensity * cols, -distortion_intensity * rows],
                                            [-distortion_intensity * cols, -distortion_intensity * rows]])

        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        warped_img = cv2.warpPerspective(img, matrix, (cols, rows))

        # Apply gradient fade (alpha blending)
        if warped_img.shape[2] != 4:
            alpha_channel = np.full((rows, cols), 255, dtype=np.uint8)
            warped_img = cv2.merge((warped_img, alpha_channel))
        
        # Create a gradient mask
        gradient_mask = np.tile(np.linspace(0, 255, cols, dtype=np.uint8), (rows, 1))
        gradient_mask = (gradient_mask * gradient_intensity).astype(np.uint8)
        warped_img[:, :, 3] = cv2.addWeighted(warped_img[:, :, 3], 1 - gradient_intensity, gradient_mask, gradient_intensity, 0)

        # Apply color adjustment (brightness or color shift)
        hsv_img = cv2.cvtColor(warped_img[:, :, :3], cv2.COLOR_BGR2HSV)
        hsv_img[:, :, 1] = cv2.add(hsv_img[:, :, 1], int(255 * color_adjustment_intensity))
        color_adjusted_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

        # Add noise (Gaussian noise)
        noise = np.random.normal(0, noise_intensity * 255, color_adjusted_img.shape).astype(np.uint8)
        noisy_img = cv2.add(color_adjusted_img, noise)

        # Merge with alpha channel
        final_img = cv2.merge((noisy_img, warped_img[:, :, 3]))

        # Convert the processed image back to BytesIO format
        _, processed_image_array = cv2.imencode('.png', final_img)
        processed_image_io = BytesIO(processed_image_array.tobytes())

        return processed_image_io
    except Exception as e:
        log_to_json(f"Error occurs for {e}", current_file_name)
        return None
