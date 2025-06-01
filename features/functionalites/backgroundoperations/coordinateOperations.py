from PIL import Image, ImageChops
import os
from io import BytesIO
from features.functionalites.backgroundoperations.utlis import find_line_angle
from features.functionalites.backgroundoperations.finalwrapreflection import flip_image_vertically
from features.functionalites.backgroundoperations.carbasicoperations import rotate_image_by_angle,apply_blur_and_opacity, apply_distortion_and_effects
from features.functionalites.backgroundoperations.utlis import calculate_distance, find_point_by_angle_and_distance, calculate_six_points, calculate_shadow_points, save_image_with_timestamp
from backgroundconfs import calculate_angle
from my_logging_script import log_to_json

current_file_name = "features/functionalites/backgroundoperations/coordinateOperations.py"


def splitImage(output_stream, wheelCoordinates):

    output_folder = "outputdebug"
    os.makedirs(output_folder, exist_ok=True)

    try:
        # Load the image from the output_stream
        image = Image.open(output_stream).convert("RGBA")
        image_width, image_height = image.size

        # Extract wheel coordinates safely
        try:
            x1, y1 = wheelCoordinates["front_bottom_x"], wheelCoordinates["front_bottom_y"]
            x2, y2 = wheelCoordinates["back_bottom_x"], wheelCoordinates["back_bottom_y"]
        except KeyError:
            print("Invalid wheel coordinates. Processing the entire image as is.")
            return {
                "leftPart": flip_image_vertically(apply_blur_and_opacity(output_stream, blur_intensity=0.6, opacity_intensity=0.5, brightness_intensity=-0.8)),
                "rightPart": flip_image_vertically(apply_blur_and_opacity(output_stream, blur_intensity=0.7, opacity_intensity=0.5))
            }

        # Calculate the rotation angle
        angle = find_line_angle(x1, y1, x2, y2)
        print(f"Calculated rotation angle: { angle} degrees")

        # Rotate the image counterclockwise to align the split line vertically
        rotated_image = image.rotate((angle), resample=Image.BICUBIC, expand=True)

        # Update the split point coordinates
        rotated_split_x = int(x1 - 100 * (rotated_image.width / image_width)) - 35

        # Ensure rotated_split_x is within valid bounds
        if rotated_split_x < 0 or rotated_split_x > rotated_image.width:
            print(f"Invalid split coordinate: {rotated_split_x}. Processing the entire image as is.")
            return {
                "leftPart": flip_image_vertically(apply_blur_and_opacity(output_stream, blur_intensity=0.6, opacity_intensity=0.5, brightness_intensity=-0.8)),
                "rightPart": flip_image_vertically(apply_blur_and_opacity(output_stream, blur_intensity=0.6, opacity_intensity=0.15))
            }

        # Split the rotated image into left and right parts
        left_part = rotated_image.crop((0, 0, rotated_split_x, rotated_image.height))
        right_part = rotated_image.crop((rotated_split_x, 0, rotated_image.width, rotated_image.height))

        # Crop transparent areas from both parts
        left_part_cropped = crop_transparent_area(left_part)
        right_part_cropped = crop_transparent_area(right_part)

        # Convert the left and right parts to BytesIO
        left_part_stream = BytesIO()
        left_part_cropped.save(left_part_stream, format="PNG")
        left_part_stream.seek(0)

        right_part_stream = BytesIO()
        right_part_cropped.save(right_part_stream, format="PNG")
        right_part_stream.seek(0)

        # Return processed left and right parts
        return {
            "leftPart": flip_image_vertically(apply_blur_and_opacity(left_part_stream, blur_intensity=0.6, opacity_intensity=0.5, brightness_intensity=-0.1)),
            "rightPart": flip_image_vertically(apply_blur_and_opacity(right_part_stream,blur_intensity=1.2, opacity_intensity=0.3))
        }

    except Exception as e:
        print(f"Error during splitImage execution: {e}. Processing the entire image as is.")
        # Return the original image processed as leftPart and rightPart
        output_stream.seek(0)  # Ensure the stream is reset
        return {
            "leftPart": flip_image_vertically(apply_blur_and_opacity(output_stream, blur_intensity=0.2, opacity_intensity=0.15, brightness_intensity=-0.8)),
            "rightPart": flip_image_vertically(apply_blur_and_opacity(output_stream, blur_intensity=0.3, opacity_intensity=0.15))
        }


def splitImageReverse(output_stream, wheelCoordinates):
    print(f"Got the image in spitiimage reverse , wheel coordinates are: {wheelCoordinates}")
    output_folder = "outputdebug"
    os.makedirs(output_folder, exist_ok=True)

    try:
        # Load the image from the output_stream
        image = Image.open(output_stream).convert("RGBA")
        image_width, image_height = image.size

        # Extract wheel coordinates safely
        try:
            x1, y1 = wheelCoordinates["front_bottom_x"], wheelCoordinates["front_bottom_y"]
            x2, y2 = wheelCoordinates["back_bottom_x"], wheelCoordinates["back_bottom_y"]
        except KeyError:
            print("Invalid wheel coordinates. Processing the entire image as is.")
            return {
                "leftPart": flip_image_vertically(apply_blur_and_opacity(output_stream, blur_intensity=0.15, opacity_intensity=0.01, brightness_intensity=-0.8)),
                "rightPart": flip_image_vertically(apply_blur_and_opacity(output_stream, blur_intensity=0.15, opacity_intensity=0.01))
            }

        # Calculate the rotation angle
        wheel_left = (x1,y1)
        wheel_right = (x2, y2)
        angle_values = calculate_angle(wheel_left, wheel_right) 
        if isinstance(angle_values, dict) and "degree" in angle_values:
            angle_values["degree"] += 180

        angle = angle_values["degree"]
        print(f"Calculated rotation angle: { angle} degrees")

        # Rotate the image counterclockwise to align the split line vertically
        rotated_image =  image.rotate((angle), resample=Image.BICUBIC, expand=True)
        


        # Update the split point coordinates
        rotated_split_x = int(x1 - 100 * (rotated_image.width / image_width)) - 35

        # Ensure rotated_split_x is within valid bounds
        if rotated_split_x < 0 or rotated_split_x > rotated_image.width:
            print(f"Invalid split coordinate: {rotated_split_x}. Processing the entire image as is.")
            return {
                "leftPart": flip_image_vertically(apply_blur_and_opacity(output_stream, blur_intensity=0.6, opacity_intensity=0.5, brightness_intensity=-0.8)),
                "rightPart": flip_image_vertically(apply_blur_and_opacity(output_stream, blur_intensity=0.6, opacity_intensity=0.15))
            }

        # Split the rotated image into left and right parts
        left_part =  rotated_image # rotated_image.crop((0, 0, rotated_split_x, rotated_image.height))
        right_part =  rotated_image # rotated_image.crop((rotated_split_x, 0, rotated_image.width, rotated_image.height))

        # Crop transparent areas from both parts
        left_part_cropped = crop_transparent_area(left_part)
        right_part_cropped = crop_transparent_area(right_part)

        # Convert the left and right parts to BytesIO
        left_part_stream = BytesIO()
        left_part_cropped.save(left_part_stream, format="PNG")
        left_part_stream.seek(0)

        right_part_stream = BytesIO()
        right_part_cropped.save(right_part_stream, format="PNG")
        right_part_stream.seek(0)

        # Return processed left and right parts
        return {
            "leftPart": flip_image_vertically(apply_blur_and_opacity(left_part_stream, blur_intensity=0.1, opacity_intensity=0.02, brightness_intensity=-0.1)),
            "rightPart": flip_image_vertically(apply_blur_and_opacity(right_part_stream, blur_intensity=0.1, opacity_intensity=0.02, brightness_intensity=-0.1))
        }

    except Exception as e:
        print(f"Error during splitImage execution: {e}. Processing the entire image as is.")
        # Return the original image processed as leftPart and rightPart
        output_stream.seek(0)  # Ensure the stream is reset
        return {
            "leftPart": flip_image_vertically(apply_blur_and_opacity(output_stream, blur_intensity=0.2, opacity_intensity=0.4, brightness_intensity=-0.8)),
            "rightPart": flip_image_vertically(apply_blur_and_opacity(output_stream, blur_intensity=0.3, opacity_intensity=0.15))
        }



def find_perspective_points(output_stream, wheelCoordinates):
    print(f"Got the image in find_perspective_points func , wheel coordinates are: {wheelCoordinates}")
    output_folder = "outputdebug"
    os.makedirs(output_folder, exist_ok=True)

    try:
        # Load the image from the output_stream
        image = Image.open(output_stream).convert("RGBA")
        image_width, image_height = image.size

        # Extract wheel coordinates safely
        try:
            x1, y1 = wheelCoordinates["front_bottom_x"], wheelCoordinates["front_bottom_y"]
            x2, y2 = wheelCoordinates["back_bottom_x"], wheelCoordinates["back_bottom_y"]
        except KeyError:
            print("Invalid wheel coordinates. Processing the entire image as is.")
        
        # points
        wheel_right = (x1,y1)
        wheel_left = (x2, y2)
        image_bottom_left = (0, image_height)
        image_bottom_right = (1920, image_height)

        # Calculate the car angle 

        angle_values = calculate_angle(wheel_left, wheel_right) 
        original_angle = angle_values["degree"]
        if isinstance(angle_values, dict) and "degree" in angle_values:
            angle_values["degree"] += 180

        positive_angle = angle_values["degree"]
        print(f"Calculated rotation angle: { original_angle} degrees")

        # Calculate the corner angle from front wheel
        corner_angle_values = calculate_angle(image_bottom_left, wheel_right) 
        corner_original_angle = corner_angle_values["degree"]
        if isinstance(corner_angle_values, dict) and "degree" in corner_angle_values:
            corner_angle_values["degree"] += 180

        corner_positive_angle = corner_angle_values["degree"]
        print(f"Calculated corner poiint angle: { corner_original_angle} degrees")

        # calculate left corner angle 
        after_rotate_angle = -(corner_original_angle) + 2*(original_angle)
        
        print(f"Calculated cpprodmate after roation: { after_rotate_angle} degrees")

        # calcuilate left corner distance
        distance_lc_from_rw = calculate_distance(image_bottom_left, wheel_right)


        # find rotatated corner cooridnate
        lb_after_rotate = find_point_by_angle_and_distance(wheel_right, 
                                                                    after_rotate_angle, 
                                                                    distance_lc_from_rw, "min")

        print(f"Calculated after rotated left bottom: { lb_after_rotate} and distance : {distance_lc_from_rw}")

        rb_after_rotate = find_point_by_angle_and_distance(lb_after_rotate, 2*original_angle, 1920, "max")
        lt_after_rotate = find_point_by_angle_and_distance(lb_after_rotate, 2*original_angle - 90, image_height, "min")
        rt_after_rotate = find_point_by_angle_and_distance(rb_after_rotate, 2*original_angle - 90, image_height, "min")

        print(f"Other points right_bottom {rb_after_rotate}  left_top {lt_after_rotate} right_top {rt_after_rotate}")

        # Split the rotated image into left and right parts
     
        # Return processed left and right parts
        return {
            "left_top": lt_after_rotate,
            "left_bottom": lb_after_rotate,
            "right_top": rt_after_rotate,
            "right_bottom": rb_after_rotate
        }

    except Exception as e:
        log_to_json(f"Error during splitImage execution: {e}. Processing the entire image as is.", current_file_name)
        # Return the original image processed as leftPart and rightPart  # Ensure the stream is reset
        return {
            "left_top": (0,0),
            "left_bottom": (0,0),
            "right_top": (0,0),
            "right_bottom": (0,0)
        }


def crop_transparent_area(image):
    """
    Crop the transparent area from the top and bottom of an RGBA image.
    """
    # Get the alpha channel
    try:
        alpha = image.split()[-1]

        # Get the bounding box of non-transparent pixels
        bbox = ImageChops.difference(alpha, Image.new("L", alpha.size, 0)).getbbox()

        if bbox:
            return image.crop(bbox)
        return image 
    except Exception as e:
        print(f"Error occurs for {e}", )
        return None

