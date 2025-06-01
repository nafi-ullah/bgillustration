from flask import Flask, request, jsonify, send_from_directory
import requests
import json
import os
from queue import Queue
from threading import Thread
import time
from PIL import Image
from rembg import remove
from io import BytesIO
from datetime import datetime
from my_logging_script import log_to_json
import numpy as np
import cv2


from features.functionalites.backgroundoperations.coordinateOperations import apply_blur_and_opacity
from features.functionalites.backgroundoperations.roundedsegmented import detect_wheels_and_annotate,detect_left_right_wheels, headlight_points,  detect_cars_coordinates, detect_cars_top_bottom_certainx
from features.functionalites.backgroundoperations.coordinateOperations import splitImage, find_perspective_points
from features.functionalites.backgroundoperations.utlis import calculate_six_points,  modify_perspective_points, calculate_shadow_points, save_image_with_timestamp, remove_all_images
from features.functionalites.backgroundoperations.finalwrapreflection import process_split_and_perspective_warp, create_canvas_with_perspective
from features.functionalites.backgroundoperations.shadowmaker import create_shadow_shape
from features.functionalites.backgroundoperations.carbasicoperations import rotate_image_by_angle, apply_blur_and_opacity, create_reflection_effect
from features.exterior_process.reflectionwrap import transform_image_with_tps, overlay_car_on_background


current_file_name = "features/functionalites/backgroundoperations/reflection.py"
imagePath = "./data/all/image01-original(01).jpg"
output_folder = "./outputdebug"

def addReflection(image_bytes_io, transparent_car_bytes_io,angle_id, angle_view="normal"):
    # ---------detect wheel coordinates------------
    print(f"GOt the function in addReflection =====> angle view is {angle_view}")
    #split_result = {"leftPart": transparent_car_bytes_io, "rightPart": transparent_car_bytes_io}
    main_image_io = transparent_car_bytes_io
    try:
        carCoordinates = detect_cars_coordinates(image_bytes_io)
        #headlight_mid = headlight_points(image_bytes_io)
        #topmost, bottommost = detect_cars_top_bottom_certainx(image_bytes_io, headlight_mid["headlight_mid_x"])
        #print(f"headlight borabor points top: {topmost}  bottom: {bottommost}  headlght: {headlight_mid}" )

        if angle_view == 'normal':
            # save_image_with_timestamp(transparent_car_bytes_io, 'outputs/dumtest/outputdebug/finalimages', 'hackathon_carimage.png')
            wheelCoordinates = detect_wheels_and_annotate(image_bytes_io, 'outputdebug')
            if angle_id == "7":
                split_result = splitImage(transparent_car_bytes_io, wheelCoordinates)
                calculated_points = calculate_six_points(wheelCoordinates, carCoordinates)
             
        else:
            wheelCoordinates = detect_left_right_wheels(image_bytes_io)
            perspective_points = find_perspective_points(transparent_car_bytes_io, wheelCoordinates)
            modified_points = modify_perspective_points(perspective_points)
            # calculated_points = calculate_reverse_six_points(wheelCoordinates, carCoordinates)
        print(f"On the function of addReflection: wheelCoordinates:  {wheelCoordinates} carCoordinates: {carCoordinates}")  
        # ---------final wrap------------
        # Assuming process_split_and_perspective_warp returns a BytesIO object
        if angle_view == 'normal':
            # -----------Get the shadow points

            # Open the image from the BytesIO stream
            origina_image_open = Image.open(transparent_car_bytes_io)

            # Get the width and height
            width, height = origina_image_open.size


            if angle_id == "7":
                final_car_output_stream = process_split_and_perspective_warp(
                transparent_car_bytes_io, calculated_points, split_result
                )
            else: 
                origin_y = wheelCoordinates["front_bottom_y"] - (height - wheelCoordinates["front_bottom_y"]) + 20
                origin = (0, origin_y)
                flipped_image = apply_blur_and_opacity(flip_image_vertically(transparent_car_bytes_io), blur_intensity=0.8, opacity_intensity=0.15) 
                the_image_to_be_reflected = create_canvas_with_image(flipped_image, origin)
                calculated_points = calcualte_8_source_points(image_bytes_io, wheelCoordinates, carCoordinates)
                if calculated_points == None:
                    log_to_json("Points couldnot calculated during unexpected issues", current_file_name)
                    return None
                reversed_points, destination_points = calculated_points
                reversed_points = calculate_points_as_array(reversed_points)
                destination_points = calculate_points_as_array(destination_points)

                reflected_curved = transform_image_with_tps(reversed_points, destination_points,  the_image_to_be_reflected)
                if reflected_curved is None:
                    return None
                
                
                final_car_output_stream = overlay_car_on_background(reflected_curved, main_image_io)

        elif angle_view == 'reverse':
            reflected_image = apply_blur_and_opacity(transparent_car_bytes_io, blur_intensity=0.7, opacity_intensity=0.3)
            # reflected_image = create_reflection_effect(transparent_car_bytes_io, opacity_intensity=1, gaussian_blur_intensity=1.6, brightness_intensity=1.0)
            final_car_output_stream = create_canvas_with_perspective(
               modified_points,
               transparent_car_bytes_io,
               reflected_image
            )

        # Convert the returned BytesIO stream into a Pillow Image object
        final_car_image = Image.open(final_car_output_stream)
        output_stream = BytesIO()
        final_car_image.save(output_stream, format="PNG")
        output_stream.seek(0)

        return output_stream
    except FileNotFoundError as e:
        log_to_json(f"File not found error {e}", current_file_name)
        return None
    except ValueError as e:
        log_to_json(f"ValueError {e}", current_file_name)
        return None
    except Exception as e:
        log_to_json(f"Exceptional error {e}", current_file_name)
        return None

      

def calcualte_8_source_points(image_bytes_io, wheelCoordinates, carCoordinates, angle_id="2"):
    try:
        # Extract wheel and car coordinates
        fbx, fby = wheelCoordinates["front_bottom_x"], wheelCoordinates["front_bottom_y"]
        bbx, bby = wheelCoordinates["back_bottom_x"], wheelCoordinates["back_bottom_y"]
        cbx, cby = carCoordinates["cars_bottom_left_x"], carCoordinates["cars_bottom_left_y"]
        ctx, cty = carCoordinates["cars_top_left_x"], carCoordinates["cars_top_left_y"]
        crbx, crby = carCoordinates["cars_bottom_right_x"], carCoordinates["cars_bottom_right_y"]

        # First pair
        fpfx, fpfy = cbx, cby
        get_coords = detect_cars_top_bottom_certainx(image_bytes_io, cbx + 120)
        fpsx, fpsy = get_coords["topmost_coord"]


        # Second pair
        headlight_mid = headlight_points(image_bytes_io)
        get_coords_headlights = detect_cars_top_bottom_certainx(image_bytes_io, headlight_mid["headlight_mid_x"])
        spfx, spfy = get_coords_headlights["bottommost_coord"]
        spsx, spsy = get_coords_headlights["topmost_coord"]

        # Third pair
        tpfx, tpfy = fbx, fby
        get_coords_third = detect_cars_top_bottom_certainx(image_bytes_io, tpfx)
        tpsx, tpsy = get_coords_third["topmost_coord"]

        # Fourth pair
        fofx, fofy = int(bbx + ((crbx - bbx) / 2)), bby - 10
        get_coords_fourth = detect_cars_top_bottom_certainx(image_bytes_io, fofx)
        fosx, fosy = get_coords_fourth["topmost_coord"]

        # Format all points into a JSON object
        threshold_up = 30
        fpsy = fpsy - threshold_up
        spsy = spsy - threshold_up
        tpsy = tpsy - threshold_up
        fosy = fosy - threshold_up

        fofy = fofy - threshold_up

        points = {
            "first_pair": {"first": (fpfx, fpfy), "second": (fpsx, fpsy )},
            "second_pair": {"first": (spfx, spfy), "second": (spsx, spsy )},
            "third_pair": {"first": (tpfx, tpfy), "second": (tpsx, tpsy )},
            "fourth_pair": {"first": (fofx, fofy), "second": (fosx, fosy )}
        }

        car_height = fby

        rfpsy = 2*car_height - fpsy
        rspsy = 2*car_height - spsy 
        rtpsy = 2*car_height - tpsy 
        rfosy = 2*car_height - fosy 

        rfpfy = 2*car_height - fpfy
        rspfy = 2*car_height - spfy
        rtpfy = 2*car_height - tpfy
        rfofy = 2*car_height - fofy 
        
        reversed_points = {
            "first_pair": {"first": (fpfx, rfpfy), "second": (fpsx, rfpsy )},
            "second_pair": {"first": (spfx, rspfy), "second": (spsx, rspsy )},
            "third_pair": {"first": (tpfx, rtpfy), "second": (tpsx, rtpsy )},
            "fourth_pair": {"first": (fofx, rfofy), "second": (fosx, rfosy )}
        }
        # y values
        dfofy = fofy 
        
        threshold_parameter = (rfofy - fofy)
        dfosy = rfosy - threshold_parameter

        
        dfpfy = rfpfy - int(threshold_parameter / 4)
        dfpsy = rfpsy 
        
        dspfy = rspfy + int(threshold_parameter / 4)
        dspsy = rspsy 


        dtpfy = rtpfy
        dtpsy = rtpsy 



        # x values
        dfpfx = fpfx
        dfpsx = fpsx - int(threshold_parameter /4)

        dspfx = spfx
        dspsx = spsx - int(threshold_parameter /8)

        dtpfx = tpfx
        dtpsx = tpsx - int(threshold_parameter /8)

        dfofx = fofx
        dfosx = fosx - int(threshold_parameter / 8)



        destination_points = {
            "first_pair": {"first": (fpfx, dfpfy), "second": (dfpsx, dfpsy )},
            "second_pair": {"first": (spfx, dspfy), "second": (dspsx, dspsy )},
            "third_pair": {"first": (tpfx, dtpfy), "second": (dtpsx, dtpsy )},
            "fourth_pair": {"first": (fofx, dfofy), "second": (dfosx , dfosy )}
        }

        # Print points for debugging
        print(f"main car ponts {points}")
        print(f"calculated poinst {reversed_points}")
        print(f"destination poinst {destination_points}")
        

        return reversed_points, destination_points

    except Exception as e:
        # Log the error and return an error message
        log_to_json("Promlem to detect coordinate points", current_file_name)     
        return None


def calcualte_8_source_points_nafi_algo(image_bytes_io, wheelCoordinates, carCoordinates, angle_id="2"):
    try:
        # Extract wheel and car coordinates
        fbx, fby = wheelCoordinates["front_bottom_x"], wheelCoordinates["front_bottom_y"]
        bbx, bby = wheelCoordinates["back_bottom_x"], wheelCoordinates["back_bottom_y"]
        cbx, cby = carCoordinates["cars_bottom_left_x"], carCoordinates["cars_bottom_left_y"]
        ctx, cty = carCoordinates["cars_top_left_x"], carCoordinates["cars_top_left_y"]
        crbx, crby = carCoordinates["cars_bottom_right_x"], carCoordinates["cars_bottom_right_y"]

        # First pair
        fpfx, fpfy = cbx, cby
        get_coords = detect_cars_top_bottom_certainx(image_bytes_io, cbx + 120)
        fpsx, fpsy = get_coords["topmost_coord"]


        # Second pair
        headlight_mid = headlight_points(image_bytes_io)
        get_coords_headlights = detect_cars_top_bottom_certainx(image_bytes_io, headlight_mid["headlight_mid_x"])
        spfx, spfy = get_coords_headlights["bottommost_coord"]
        spsx, spsy = get_coords_headlights["topmost_coord"]

        # Third pair
        tpfx, tpfy = fbx, fby
        get_coords_third = detect_cars_top_bottom_certainx(image_bytes_io, tpfx)
        tpsx, tpsy = get_coords_third["topmost_coord"]

        # Fourth pair
        fofx, fofy = int(bbx + ((crbx - bbx) / 2)), bby - 10
        get_coords_fourth = detect_cars_top_bottom_certainx(image_bytes_io, fofx)
        fosx, fosy = get_coords_fourth["topmost_coord"]

        # Format all points into a JSON object
        threshold_up = 30
        fpsy = fpsy - threshold_up
        spsy = spsy - threshold_up
        tpsy = tpsy - threshold_up
        fosy = fosy - threshold_up

        fofy = fofy - threshold_up

        points = {
            "first_pair": {"first": (fpfx, fpfy), "second": (fpsx, fpsy )},
            "second_pair": {"first": (spfx, spfy), "second": (spsx, spsy )},
            "third_pair": {"first": (tpfx, tpfy), "second": (tpsx, tpsy )},
            "fourth_pair": {"first": (fofx, fofy), "second": (fosx, fosy )}
        }

        car_height = fby

        rfpsy = 2*car_height - fpsy
        rspsy = 2*car_height - spsy 
        rtpsy = 2*car_height - tpsy 
        rfosy = 2*car_height - fosy 

        rfpfy = 2*car_height - fpfy
        rspfy = 2*car_height - spfy
        rtpfy = 2*car_height - tpfy
        rfofy = 2*car_height - fofy 
        
        reversed_points = {
            "first_pair": {"first": (fpfx, rfpfy), "second": (fpsx, rfpsy )},
            "second_pair": {"first": (spfx, rspfy), "second": (spsx, rspsy )},
            "third_pair": {"first": (tpfx, rtpfy), "second": (tpsx, rtpsy )},
            "fourth_pair": {"first": (fofx, rfofy), "second": (fosx, rfosy )}
        }


        dfpfy = rfpfy - int(((rfpfy - fpfy)/3)) + 20  
        dfpsy = rfpsy - int(((rfpfy - fpfy)/3)) + 30
        
        dspfy = rspfy + 50
        dspsy = rspsy + 30

        dtpfy = rtpfy
        dtpsx = tpsx + 70
        dtpsy = rtpsy - 100

        dfofy = fofy + 10
        dfosy = rfofy - (dfofy - fofy)



        destination_points = {
            "first_pair": {"first": (fpfx, dfpfy), "second": (fpsx, dfpsy )},
            "second_pair": {"first": (spfx, dspfy), "second": (spsx, dspsy )},
            "third_pair": {"first": (tpfx, dtpfy), "second": (dtpsx, dtpsy )},
            "fourth_pair": {"first": (fofx, dfofy), "second": (fosx + 10, dfosy )}
        }

        # Print points for debugging
        print(f"calculated poinst {reversed_points}")
        print(f"destination poinst {destination_points}")
        

        return reversed_points, destination_points

    except Exception as e:
        # Log the error and return an error message
        log_to_json("Promlem to detect coordinate points", current_file_name)     
        return None
    

def addFlipAndMergeReflection(image_bytes_io, transparent_car_bytes_io,angle_id, isActive=None):
    # ---------detect wheel coordinates------------
    print("addFlipAndMergeReflection for reflection got it")
    try:
        origina_image_open = Image.open(transparent_car_bytes_io)

            # Get the width and height
        width, height = origina_image_open.size
        height_up = 300
        if angle_id == "3" or angle_id == "5":
            height_up = 270

        origin_y = height - height_up
        origin = (0, origin_y)
        if isActive == True:
            opacity_val = 0.35
        else:
            opacity_val = 0
        flipped_image = apply_blur_and_opacity(flip_image_vertically(transparent_car_bytes_io), blur_intensity=0.8, opacity_intensity=opacity_val)
        the_image_to_be_reflected = create_canvas_with_image(flipped_image, origin)
        final_car_output_stream = overlay_car_on_background(the_image_to_be_reflected, transparent_car_bytes_io)


        # Convert the returned BytesIO stream into a Pillow Image object
        final_car_image = Image.open(final_car_output_stream)
        output_stream = BytesIO()
        final_car_image.save(output_stream, format="PNG")
        output_stream.seek(0)

        return output_stream
    except FileNotFoundError as e:
        log_to_json(f"File not found error {e}", current_file_name)
        return None
    except ValueError as e:
        log_to_json(f"ValueError {e}", current_file_name)
        return None
    except Exception as e:
        log_to_json(f"Exceptional error {e}", current_file_name)
        return None


def calculate_points_as_array(points_dict):
    try:
        points = []
        for pair in points_dict.values():
            # Extract 'first' and 'second' points from the pair
            first = pair['first']
            second = pair['second']
            points.extend([first, second])

        # Convert to NumPy array with dtype=np.float32
        return np.array(points, dtype=np.float32)
    except Exception as e:
        # Log the error and return an error message
        log_to_json("Promlem to detect coordinate points", current_file_name)     
        return None

def flip_image_vertically(image_bytesio):
    try:
    # Open the image from BytesIO
        image = Image.open(image_bytesio)

        # Flip the image vertically
        flipped_image = image.transpose(Image.FLIP_TOP_BOTTOM)

        # Save the flipped image to BytesIO
        output_bytesio = BytesIO()
        flipped_image.save(output_bytesio, format="PNG")
        output_bytesio.seek(0)

        return output_bytesio
    except Exception as e:
        # Log the error and return an error message
        log_to_json("Promlem to detect coordinate points", current_file_name)     
        return None
# imageProcessInitiate(imagePath)

def create_canvas_with_image(image_bytesio, coordinate):
    try:
    # Open the input image
        image = Image.open(image_bytesio)
        width, height = image.size

        # Create a canvas with 2x the height of the image
        canvas_height = 2 * height
        canvas = Image.new("RGBA", (width, canvas_height), (0, 0, 0, 0))

        # Paste the image onto the canvas at the specified coordinate
        canvas.paste(image, coordinate, mask=image if image.mode == 'RGBA' else None)

        # Save the result to a BytesIO object
        output_bytesio = BytesIO()
        canvas.save(output_bytesio, format="PNG")
        output_bytesio.seek(0)

        return output_bytesio
    except Exception as e:
        # Log the error and return an error message
        log_to_json("Promlem to detect coordinate points", current_file_name)     
        return None



# def process_all_images(folder_path: str, process_function):


#     if not os.path.exists(folder_path):
#         print(f"The folder '{folder_path}' does not exist.")
#         return
    
#     # Iterate through all files in the folder
#     for file_name in os.listdir(folder_path):
#         # Full path to the image
#         file_path = os.path.join(folder_path, file_name)
        
#         # Check if the file is an image (basic extension check)
#         if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
#             try:
#                 # Call the provided function on the image
#                 process_function(file_path)
#                 print(f"Processed: {file_path}")
#             except Exception as e:
#                 print(f"Error processing {file_path}: {e}")
#         else:
#             print(f"Skipping non-image file: {file_name}")



# folder = "data/image01"
# process_all_images(folder, imageProcessInitiate)