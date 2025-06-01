from PIL import Image, ImageDraw
import numpy as np
from io import BytesIO
import os
import cv2
from datetime import datetime
from features.functionalites.backgroundoperations.addbackground.logo_maker import logo_maker_initiate, combine_logo_with_bg, combine_logo_with_bg_Reverse, basic_combine_logo_with_bg
from my_logging_script import log_to_json
from features.functionalites.backgroundoperations.carbasicoperations import  left_reversewall_image_operation, right_image_operation, basic_floor_image_operation
current_file_name = "features/functionalites/backgroundoperations/addbackground/addwallinitiate.py"

def apply_vanishing_point(image, polygon, canvas_size):
    try:
        if len(polygon) != 4:
            raise ValueError(f"Polygon must have exactly 4 points. Found: {len(polygon)}")

        width, height = image.size
        src = np.array([
            [0, 0],
            [width, 0],
            [width, height],
            [0, height]
        ], dtype=np.float32)

        # Convert polygon to float32 numpy array
        dst = np.array(polygon, dtype=np.float32)

        # Compute the perspective transformation matrix
        matrix = cv2.getPerspectiveTransform(src, dst)

        # Warp the image to fit the polygon
        transformed = cv2.warpPerspective(np.array(image), matrix, canvas_size)
        return Image.fromarray(transformed)
    except FileNotFoundError as e:
        log_to_json(f"apply_vanishing_point-- File not found error {e}", current_file_name)
        return f'Sorry the image is not processed for: File not found: {str(e)}'
    except ValueError as e:
        log_to_json(f"apply_vanishing_point-- ValueError {e}", current_file_name)
        return f'Sorry the image is not processed for: Value error: {str(e)}'
    except Exception as e:
        log_to_json(f"apply_vanishing_point-- Exceptional error {e}", current_file_name)
        return  f'Sorry : An unexpected error occurred: {str(e)}'

def resize_texture_to_polygon(texture, polygon):
    # Calculate the width and height of the polygon
    def distance(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    # Calculate the width and height of the floor polygon
    width = int(distance(polygon[0], polygon[1]))  # Distance between left and right points
    height = int(distance(polygon[1], polygon[2]))  # Distance between right and bottom points

    # Resize the texture to match the polygon aspect ratio
    resized_texture = texture.resize((width, height), Image.Resampling.LANCZOS)
    return resized_texture

def createBackground(wallImages=None, wallCoordinates=None, wall_logo=None, logo_position=None):
    try:
        canvas_size = (1920, 1440)
        canvas = Image.new("RGB", canvas_size, (255, 255, 255))

        # shouldAddLogo = wall_logo is not None and logo_position is not None
        #save_image_with_timestamp(wall_logo, "./addbackground/backgrounds", "wall_logo.png")

        # First, render the floor behind all other layers
        for wall, coordinates in wallCoordinates.items():
            image_bytes = wallImages[f"{wall}_img"]  # Now, we get the BytesIO object
            
            
            if wall == "left_wall":
                image_bytes = combine_logo_with_bg(wallImages[f"{wall}_img"], logo_position, wall_logo)

            if wall == "right_wall":
                image_bytes = right_image_operation(wallImages[f"{wall}_img"], )

            texture = Image.open(image_bytes)

            for polygon in coordinates:
                # Ensure the polygon has 4 points
                if len(polygon) != 4:
                    raise ValueError(f"Each polygon must have 4 points. Found: {len(polygon)}")

                # Resize texture to fit polygon aspect ratio
                resized_texture = texture

                # Apply vanishing point transformation
                transformed_texture = apply_vanishing_point(resized_texture, polygon, canvas_size)
        
                # Create a mask for the polygon
                mask = Image.new("L", canvas_size, 0)
                ImageDraw.Draw(mask).polygon(polygon, fill=255)

                # Paste the transformed texture onto the canvas using the mask
                canvas.paste(transformed_texture, (0, 0), mask)
                # output_path = os.path.join("outputs/dumtest/textures", f"{wall}_texture.png")
                # transformed_texture.save(output_path)

        # Save the final image and return it as BytesIO
        # output_path = './addbackground/backgrounds/dynamic_bg.png'
        # os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # canvas.save(output_path)

        output = BytesIO()
        canvas.save(output, format="PNG")
        output.seek(0)
        return output
    except FileNotFoundError as e:
        log_to_json(f"createBackground-- File not found error {e}", current_file_name)
        return f'Sorry the image is not processed for: File not found: {str(e)}'
    except ValueError as e:
        log_to_json(f"createBackground-- ValueError {e}", current_file_name)
        return f'Sorry the image is not processed for: Value error: {str(e)}'
    except Exception as e:
        log_to_json(f"createBackground-- Exceptional error {e}", current_file_name)
        return  f'Sorry : An unexpected error occurred: {str(e)}'

def createBackgroundReverse(wallImages=None, wallCoordinates=None, wall_logo=None, logo_position=None):
    try:
        print(wallCoordinates)
        canvas_size = (1920, 1440)
        canvas = Image.new("RGB", canvas_size, (255, 255, 255))

        #save_image_with_timestamp(wall_logo, "./addbackground/backgrounds", "wall_logo.png")

        # First, render the floor behind all other layers
        for wall, coordinates in wallCoordinates.items():
            image_bytes = wallImages[f"{wall}_img"]  # Now, we get the BytesIO object
            
            if wall == "right_wall":
                image_bytes = combine_logo_with_bg_Reverse(wallImages[f"{wall}_img"], logo_position, wall_logo)
            if wall == "left_wall":
                
                image_bytes = left_reversewall_image_operation(wallImages[f"{wall}_img"], )
               

            texture = Image.open(image_bytes)

            for polygon in coordinates:
                # Ensure the polygon has 4 points
                if len(polygon) != 4:
                    raise ValueError(f"Each polygon must have 4 points. Found: {len(polygon)}")

                # Resize texture to fit polygon aspect ratio
                resized_texture = texture

                # Apply vanishing point transformation
                transformed_texture = apply_vanishing_point(resized_texture, polygon, canvas_size)

                # Create a mask for the polygon
                mask = Image.new("L", canvas_size, 0)
                ImageDraw.Draw(mask).polygon(polygon, fill=255)

                # Paste the transformed texture onto the canvas using the mask
                canvas.paste(transformed_texture, (0, 0), mask)

        # Save the final image and return it as BytesIO
        # output_path = './addbackground/backgrounds/dynamic_bg.png'
        # os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # canvas.save(output_path)

        output = BytesIO()
        canvas.save(output, format="PNG")
        output.seek(0)
        return output
    except FileNotFoundError as e:
        log_to_json(f"createBackgroundReverse-- File not found error {e}", current_file_name)
        return f'Sorry the image is not processed for: File not found: {str(e)}'
    except ValueError as e:
        log_to_json(f"createBackgroundReverse-- ValueError {e}", current_file_name)
        return f'Sorry the image is not processed for: Value error: {str(e)}'
    except Exception as e:
        log_to_json(f"createBackgroundReverse -- Exceptional error {e}", current_file_name)
        return  f'Sorry : An unexpected error occurred: {str(e)}'

def createBasicBackground(wallImages=None, wallCoordinates=None, wall_logo=None, logo_position=None):
    canvas_size = (1920, 1440)
    canvas = Image.new("RGB", canvas_size, (255, 255, 255))

    #save_image_with_timestamp(wall_logo, "./addbackground/backgrounds", "wall_logo.png")

    # First, render the floor behind all other layers
    for wall, coordinates in wallCoordinates.items():
        image_bytes = wallImages[f"{wall}_img"]  # Now, we get the BytesIO object
        
        if wall == "left_wall":
            image_bytes = basic_combine_logo_with_bg(wallImages[f"{wall}_img"], logo_position, wall_logo)
        if wall == "floor_wall":
            image_bytes = basic_floor_image_operation(wallImages[f"{wall}_img"], )

        texture = Image.open(image_bytes)

        for polygon in coordinates:
            # Ensure the polygon has 4 points
            if len(polygon) != 4:
                raise ValueError(f"Each polygon must have 4 points. Found: {len(polygon)}")

            # Resize texture to fit polygon aspect ratio
            resized_texture = texture

            # Apply vanishing point transformation
            transformed_texture = apply_vanishing_point(resized_texture, polygon, canvas_size)

            # Create a mask for the polygon
            mask = Image.new("L", canvas_size, 0)
            ImageDraw.Draw(mask).polygon(polygon, fill=255)

            # Paste the transformed texture onto the canvas using the mask
            canvas.paste(transformed_texture, (0, 0), mask)

    # Save the final image and return it as BytesIO
    # output_path = './addbackground/backgrounds/dynamic_bg.png'
    # os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # canvas.save(output_path)

    output = BytesIO()
    canvas.save(output, format="PNG")
    output.seek(0)
    return output



def addBackgroundInitiate(
    wallImages_bytes, 
    logo_bytes=None, 
    floor_coordinates=None, 
    left_wall_coordinates=None, 
    right_wall_coordinates=None, 
    ceiling_coordinates=None, 
    angle=None, 
    logo_position=None
):
    # Use the coordinates passed as arguments
    print(f"--------hey its initiated-----------")
    print(f"wallimages {type(wallImages_bytes)}")
    print(f"logo_bytes {type(logo_bytes)}")
    print(f"floor_coordinates {type(floor_coordinates)}")
    print(f"left_wall_coordinates {type(left_wall_coordinates)}")
    print(f"right_wall_coordinates {type(right_wall_coordinates)}")
    print(f"ceiling_coordinates {type(ceiling_coordinates)}")
    print(f"angle {type(angle)}")

    # if isinstance(wallImages_bytes, dict):
    #     print("wallImages_bytes is a dictionary")
    # for key, value in wallImages_bytes.items():
    #     print(f"Key: {key}, Type of value: {type(value)}")

    try:
        floor = [floor_coordinates]
        left_wall = [left_wall_coordinates]
        right_wall = [right_wall_coordinates]
        ceiling = [ceiling_coordinates]

        # Wall coordinates
        wallCoordinates = {
            "floor_wall": floor,
            "left_wall": left_wall,
            "right_wall": right_wall,
            "ceiling_wall": ceiling
        }

        # Convert the wall image file paths to BytesIO
        # modified_logo = None
        # if logo_bytes is not None:
        #     modified_logo = logo_maker_initiate(logo_bytes)
    

        if angle == 'reverse':
            background_bytesio = createBackgroundReverse(wallImages_bytes, wallCoordinates, logo_bytes ,logo_position)
        else:
            background_bytesio = createBackground(wallImages_bytes, wallCoordinates, logo_bytes ,logo_position)
        return background_bytesio
    except FileNotFoundError as e:
        log_to_json(f"addBackgroundInitiate-- File not found error {e}", current_file_name)
        return f'Sorry the image is not processed for: File not found: {str(e)}'
    except ValueError as e:
        log_to_json(f"addBackgroundInitiate-- ValueError {e}", current_file_name)
        return f'Sorry the image is not processed for: Value error: {str(e)}'
    except Exception as e:
        log_to_json(f"addBackgroundInitiate-- Exceptional error {e}", current_file_name)
        return  f'Sorry : An unexpected error occurred: {str(e)}'
    
def addBasicBackgroundInitiate(wallImages_bytes=None, logo_bytes=None, floor_coordinates=None, wall_coordinates=None, logo_position=None):
    # Use the coordinates passed as arguments
    modified_logo = None
    try:
        floor = [floor_coordinates]
        left_wall = [wall_coordinates]

        # Wall coordinates
        wallCoordinates = {
            "floor_wall": floor,
            "left_wall": left_wall
        }

        # Convert the wall image file paths to BytesIO
        if logo_bytes is not None:
            modified_logo = logo_maker_initiate(logo_bytes)
        
        background_bytesio = createBasicBackground(wallImages=wallImages_bytes, wallCoordinates=wallCoordinates, wall_logo=modified_logo , logo_position=logo_position)
        
        return background_bytesio
    except FileNotFoundError as e:
        log_to_json(f"addBasicBackgroundInitiate-- File not found error {e}", current_file_name)
        return f'Sorry the image is not processed for: File not found: {str(e)}'
    except ValueError as e:
        log_to_json(f"addBasicBackgroundInitiate-- ValueError {e}", current_file_name)
        return f'Sorry the image is not processed for: Value error: {str(e)}'
    except Exception as e:
        log_to_json(f"addBasicBackgroundInitiate-- Exceptional error {e}", current_file_name)
        return  f'Sorry : An unexpected error occurred: {str(e)}'


def save_image_with_timestamp(IMAGE_PATH, file_location: str, file_name: str):

    # Ensure the folder exists
    os.makedirs(file_location, exist_ok=True)

    # Reset the file-like object to the start
    IMAGE_PATH.seek(0)

    # Open the image using Pillow
    image = Image.open(IMAGE_PATH)

    # Extract the file name and extension
    name, extension = os.path.splitext(file_name)

    # Add a timestamp to the file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_file_name = f"{name}_{timestamp}{extension}"

    # Create the full file path
    full_path = os.path.join(file_location, new_file_name)

    # Save the image in the appropriate format
    image.save(full_path)

    return full_path

# Example Usage
# if __name__ == "__main__":
#     background_bytes = addBackgroundInitiate()
#     print("Background created and saved successfully.")
