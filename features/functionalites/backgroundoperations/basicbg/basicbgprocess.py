from PIL import Image, ImageDraw, ImageOps
from io import BytesIO
import numpy as np
from dynamicfuncs import is_feature_enabled, save_image_with_timestamp
import cv2
from features.functionalites.backgroundoperations.basicbg.findtoppoint import wheel_coordinate_for_wall
from backgroundconfs import get_basic_wall_coordinates
from features.functionalites.backgroundoperations.addbackground.addwallinitiate import addBasicBackgroundInitiate
from my_logging_script import log_to_json

current_file_name = "features/functionalites/backgroundoperations/basicbg/basicbgprocess.py"

def process_image_with_canvas(new_image: BytesIO, transparent_image: BytesIO) -> dict:
    try:
        # Open the input image
        image = Image.open(new_image)
        imagetr = Image.open(transparent_image)

        # Calculate new dimensions for resizing
        new_width = 1700
        aspect_ratio = image.height / image.width
        new_height = int(new_width * aspect_ratio)

        # Resize the image
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        resized_image_tr = imagetr.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Create two canvas images (1920x1440)
        canvas_size = (1920, 1440)
        canvas_transparent = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
        canvas_green = Image.new("RGBA", canvas_size, (0, 255, 0, 255))

        # Position the resized image
        x_offset = 100
        y_offset = canvas_size[1] - 140 - resized_image.height

        # Place the resized image on the canvases
        canvas_transparent.paste(resized_image_tr, (x_offset, y_offset), resized_image if resized_image.mode == "RGBA" else None)
        canvas_green.paste(resized_image, (x_offset, y_offset), resized_image if resized_image.mode == "RGBA" else None)

        # Convert canvases to BytesIO objects
        canvas_transparent_bytes = BytesIO()
        canvas_green_bytes = BytesIO()

        canvas_transparent.save(canvas_transparent_bytes, format="PNG")
        canvas_green.save(canvas_green_bytes, format="PNG")

        canvas_transparent_bytes.seek(0)
        canvas_green_bytes.seek(0)

        # Return both canvases as JSON data
        return {
            "canvasgreen": canvas_green_bytes,
            "canvastransparent": canvas_transparent_bytes
        }

    except Exception as e:
        log_to_json(f"process_image_with_canvas: An error occurred: {e}", current_file_name)
        return {}
    

def process_straight_image_with_canvas(new_image: BytesIO, transparent_image: BytesIO, angle_id=str) -> dict:
    try:
        # Open the input image
        image = Image.open(new_image)
        imagetr = Image.open(transparent_image)
        print(f"main image size ===> : {image.width}x {image.height} and transparent one: {imagetr.width} x {imagetr.width}")
        # Calculate new dimensions for resizing
        
        new_height = image.height
        if angle_id == "3" or angle_id == "5":
            new_height = image.height
        
        aspect_ratio = image.width / image.height
        new_width = int(new_height * aspect_ratio)
        print(f"updated image size ===> : {new_width}x {new_height} ")
        # Resize the image
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        resized_image_tr = imagetr.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Create two canvas images (1920x1440)
        canvas_size = (1920, 1440)
        canvas_transparent = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
        canvas_green = Image.new("RGBA", canvas_size, (0, 0, 0, 255))

        # Position the resized image
        x_offset = 0 #int(1920/2) - int(resized_image.width/2)
        y_offset = canvas_size[1] - 200 - resized_image.height

        print(f"offset x: {x_offset} y: {y_offset}")

        # Place the resized image on the canvases
        canvas_transparent.paste(resized_image_tr, (x_offset, y_offset), resized_image if resized_image.mode == "RGBA" else None)
        canvas_green.paste(resized_image, (x_offset, y_offset), resized_image if resized_image.mode == "RGBA" else None)

        # Convert canvases to BytesIO objects
        canvas_transparent_bytes = BytesIO()
        canvas_green_bytes = BytesIO()

        canvas_transparent.save(canvas_transparent_bytes, format="PNG")
        canvas_green.save(canvas_green_bytes, format="PNG")

        canvas_transparent_bytes.seek(0)
        canvas_green_bytes.seek(0)

        # Return both canvases as JSON data
        return {
            "canvasgreen": canvas_green_bytes,
            "canvastransparent": canvas_transparent_bytes
        }

    except Exception as e:
        log_to_json(f"process_straight_image_with_canvas: An error occurred: {e}", current_file_name)
        return {}
    
def process_topbottom_image_with_canvas(new_image: BytesIO, transparent_image: BytesIO, angle_id=str) -> dict:
    try:
        # Open the input image
        image = Image.open(new_image)
        imagetr = Image.open(transparent_image)
        print(f"main image size ===> : {image.width}x {image.height} and transparent one: {imagetr.width} x {imagetr.width}")
        # Calculate new dimensions for resizing
        
        # new_width = 1920
        if image.height > 920:
            new_height = 920
        else:
            new_height = image.height
        
        aspect_ratio = image.width / image.height
        new_width = int(new_height * aspect_ratio)

        image_size_decreased = new_height / image.height #times

        new_transparent_height = int(imagetr.height * image_size_decreased)
        aspect_ratio_transparent = imagetr.width / imagetr.height
        new_transparent_width = int(new_transparent_height * aspect_ratio_transparent)
        # print(f"updated image size ===> : {new_width}x {new_height} ")
        # # Resize the image
        resized_image =  image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        resized_image_tr =  imagetr.resize((new_transparent_width, new_transparent_height), Image.Resampling.LANCZOS)

        # Create two canvas images (1920x1440)
        canvas_size = (1920, 1440)
        canvas_transparent = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
        canvas_green = Image.new("RGBA", canvas_size, (0, 255, 0, 255))

        # Position the resized image
        
            
        x_offset = int(1920/2) - int(resized_image.width/2)
        y_offset = canvas_size[1] - 200 - resized_image.height

        # Place the resized image on the canvases
        canvas_transparent.paste(resized_image_tr, (x_offset, y_offset), resized_image if resized_image.mode == "RGBA" else None)
        canvas_green.paste(resized_image, (x_offset, y_offset), resized_image if resized_image.mode == "RGBA" else None)

        # Convert canvases to BytesIO objects
        canvas_transparent_bytes = BytesIO()
        canvas_green_bytes = BytesIO()

        canvas_transparent.save(canvas_transparent_bytes, format="PNG")
        canvas_green.save(canvas_green_bytes, format="PNG")

        canvas_transparent_bytes.seek(0)
        canvas_green_bytes.seek(0)

        # Return both canvases as JSON data
        return {
            "canvasgreen": canvas_green_bytes,
            "canvastransparent": canvas_transparent_bytes
        }

    except Exception as e:
        log_to_json(f"process_straight_image_with_canvas: An error occurred: {e}", current_file_name)
        return {}
    

def add_basic_bg(wallImagesBytes, logo_bytes: BytesIO = None, new_image: BytesIO = None, transparent_image: BytesIO = None, logo_position: str = None) -> dict:
    print("Got the image in add basic bg")
    try:
        canvases = process_image_with_canvas(new_image, transparent_image)

        canvasgreen = canvases['canvasgreen']
        canvastransparent = canvases['canvastransparent']

        get_points = wheel_coordinate_for_wall(canvasgreen)

        red_dot_point = (get_points['top_x'], get_points['top_y'])  
        floor_coordinates, wall_coordinates = get_basic_wall_coordinates(get_points)

        print(f"wall coordinates floor {floor_coordinates} wall {wall_coordinates}")
        getbg = addBasicBackgroundInitiate(
            wallImages_bytes=wallImagesBytes, 
            logo_bytes=logo_bytes, 
            floor_coordinates=floor_coordinates, 
            wall_coordinates=wall_coordinates, 
            logo_position=logo_position)

        main_image = final_combine_bg(canvastransparent, getbg)

        if isinstance(main_image, BytesIO):
            img = Image.open(main_image)
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            result_io = BytesIO()
            img.save(result_io, format='JPEG')  # or 'PNG' if you want transparency
            result_io.seek(0)
            main_image = result_io

        return main_image

    except Exception as e:
        log_to_json(f"add_basic_bg: An error occurred: {e}", current_file_name)
        return None


    


def add_straight_angle_basic_bg(wallImagesBytes, logo_bytes: BytesIO = None, new_image: BytesIO = None, transparent_image: BytesIO = None, logo_position: str = None, angle_id: str = None) -> dict:
    print("Got the image in add_straight_angle_basic_bg")
    try:
        # Open the input image
        log_to_json(f"images types are new_image: {type(new_image)} transparent_image: {type(transparent_image)}", current_file_name)
        if angle_id in ["9", "10", "11", "12", "13", "14", "15", "16"]:
            canvases = process_topbottom_image_with_canvas(new_image, transparent_image, angle_id)
        else:
            canvases = process_straight_image_with_canvas(new_image, transparent_image, angle_id)

        canvasgreen =  canvases['canvasgreen']
        canvastransparent =  canvases['canvastransparent']

        green = save_image_with_timestamp(canvasgreen, 'outputs/ind', 'greencanvas.png')
        transparent = save_image_with_timestamp(canvastransparent, 'outputs/ind', 'canvstransparent.png')
        origina_image_open = Image.open(canvastransparent)
        width, height = origina_image_open.size
        origin_y = int(height/2) 

        if angle_id == "3" or angle_id == "5":
            get_points = wheel_coordinate_for_wall(canvasgreen)
            if get_points == {}:
                get_points = {"top_x": 0, "top_y": 850} 
            logo_bytes = None   # make the logo invisivle for side  left right
        else:
            get_points = {"top_x": 0, "top_y": origin_y } # (0, origin_y)

        # result_image_bytes = add_red_dot( , red_dot_point)
        #save_image_with_timestamp(result_image_bytes, 'outputs/dumtest/basicbg', 'dotpoint.png')
        print(f"topx and topy point: {get_points}")

        floor_coordinates, wall_coordinates = get_basic_wall_coordinates(get_points)
        # Return both canvases as JSON data
        print(f"wall coordinates floor {floor_coordinates} wall {wall_coordinates}")
        getbg = addBasicBackgroundInitiate(
            wallImages_bytes=wallImagesBytes, 
            logo_bytes=logo_bytes, 
            floor_coordinates=floor_coordinates, 
            wall_coordinates=wall_coordinates, 
            logo_position=logo_position)
        main_image = final_combine_bg(canvastransparent, getbg)
        #save_image_with_timestamp(main_image, 'outputs/dumtest/basicbg', 'finalimage.png')
        # Ensure main_image is a PIL Image
        if isinstance(main_image, BytesIO):
            img = Image.open(main_image)
        else:
            img = main_image  # assume it's already a PIL Image

        # Convert RGBA to RGB if needed
        if img.mode == 'RGBA':
            img = img.convert('RGB')

        # Save as JPEG
        result_io = BytesIO()
        img.save(result_io, format='JPEG')
        result_io.seek(0)
        main_image = result_io



  
        return main_image
    except Exception as e:
        log_to_json(f"add_straight_angle_basic_bg : An error occurred: {e}", current_file_name)
        return None
    

def add_red_dot(test_image: BytesIO, point: tuple) -> BytesIO:
    try:
        # Read the input image using OpenCV
        file_bytes = np.asarray(bytearray(test_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Error: Unable to read the image from BytesIO.")

        # Draw a red dot at the specified point
        cv2.circle(image, point, radius=5, color=(0, 0, 255), thickness=-1)

        # Convert the image back to BytesIO
        result_image_bytes = BytesIO()
        _, buffer = cv2.imencode('.png', image)
        result_image_bytes.write(buffer)
        result_image_bytes.seek(0)

        return result_image_bytes

    except Exception as e:
        print(f"add_red_dot: An error occurred while adding a red dot: {e}")
        return BytesIO()
    

def final_combine_bg(foreground_bytes, background_bytes):
    # Load the background and foreground images from the BytesIO objects
    try:
        background_image = Image.open(background_bytes)
        foreground_image = Image.open(foreground_bytes)
        
        # Create a new canvas (1920x1440)
        canvas = Image.new("RGBA", (1920, 1440), (0, 0, 0, 0))  # Transparent background canvas
        
        # Resize the background to fit the canvas size
        background_image = background_image.resize((1920, 1440))
        
        # Paste the background image onto the canvas
        canvas.paste(background_image, (0, 0))
        
        # Resize foreground image if needed to fit the canvas (optional)
        # foreground_image = foreground_image.resize((1920, 1440), Image.LANCZOS)
        
        # Paste the foreground image on top of the background image at position (0, 0)
        canvas.paste(foreground_image, (0, 0), foreground_image.convert("RGBA").split()[3])  # Using alpha channel for transparency if any
        
        # Save the result to a BytesIO object and seek back to the beginning
        output_bytes = BytesIO()
        canvas.save(output_bytes, format="PNG")
        output_bytes.seek(0)  # Rewind the BytesIO object to the beginning
        
        return output_bytes
    except Exception as e:
        print(f"add_red_dot: An error occurred while adding a red dot: {e}")
        return BytesIO()
