import math
from typing import Dict, Tuple, List
import os
from datetime import datetime
from PIL import Image
from io import BytesIO
from my_logging_script import log_to_json
import cv2

current_file_name = "features/functionalites/backgroundoperations/utlis.py"

def find_line_angle(x1, y1, x2, y2):
    """
    Find the angle to rotate the line between (x1, y1) and (x2, y2) to make it horizontal.
    """
    # Calculate the change in coordinates
    try:
        delta_x = x2 - x1
        delta_y = y2 - y1

        # Calculate the angle in radians using atan2
        angle_radians = math.atan2(delta_y, delta_x)

        # Convert the angle to degrees
        angle_degrees = math.degrees(angle_radians)

        return angle_degrees
    except Exception as e:
        log_to_json(f"Error occurs for {e}", current_file_name)
        return None


def calculate_six_points(wheelCoordinates, carCoordinates):
    try:
        flx, fly = wheelCoordinates["front_left_x"], wheelCoordinates["front_left_y"]
        fbx, fby = wheelCoordinates["front_bottom_x"], wheelCoordinates["front_bottom_y"]
        bbx, bby = wheelCoordinates["back_bottom_x"], wheelCoordinates["back_bottom_y"]
        cbx, cby = carCoordinates["cars_bottom_left_x"], carCoordinates["cars_bottom_left_y"]
        ctx, cty = carCoordinates["cars_top_left_x"], carCoordinates["cars_top_left_y"]
        crbx, crby = carCoordinates["cars_bottom_right_x"], carCoordinates["cars_bottom_right_y"]

        # Leftmost points
        cbxr, cbyr = cbx, cby 
        ctxr, ctyr = ctx, cbyr + (cby - cty) + 80

            # Middle points
        mbxr, mbyr = flx - 80, fby + 50
        mtxr, mtyr = flx - 150, mbyr + fby + 140

        # Rightmost points
        cbrxr, cbryr = crbx + 50 , bby -50
        ctrxr, ctryr = cbrxr - 150 , cbryr + bby + 150



        # Print the calculated points
        print(f"Leftmost Bottom Point: ({cbxr}, {cbyr})")
        print(f"Leftmost Top Point: ({ctxr}, {ctyr})")
        print(f"Rightmost Bottom Point: ({cbrxr}, {cbryr})")
        print(f"Rightmost Top Point: ({ctrxr}, {ctryr})")
        print(f"Middle Bottom Point: ({mbxr}, {mbyr})")
        print(f"Middle Top Point: ({mtxr}, {mtyr})")

        # Return the calculated points
        return {
            "ref_leftmost_bottom": (cbxr, cbyr),
            "ref_leftmost_top": (ctxr, ctyr),
            "ref_rightmost_bottom": (cbrxr, cbryr),
            "ref_rightmost_top": (ctrxr, ctryr),
            "ref_middle_bottom": (mbxr, mbyr),
            "ref_middle_top": (mtxr, mtyr),
        }
    except FileNotFoundError as e:
        log_to_json(f"File not found error {e}", current_file_name)
        return f'Sorry the image is not processed for: File not found: {str(e)}'
    except ValueError as e:
        log_to_json(f"ValueError {e}", current_file_name)
        return f'Sorry the image is not processed for: Value error: {str(e)}'
    except Exception as e:
        log_to_json(f"Exceptional error {e}", current_file_name)
        return  f'Sorry : An unexpected error occurred: {str(e)}'
    



    
def calculate_reverse_six_points(wheelCoordinates, carCoordinates):
    try:
        print(f"Car coordinates value in reverse funciton is : {carCoordinates}")
        flx, fly = wheelCoordinates["front_left_x"], wheelCoordinates["front_left_y"]
        fbx, fby = wheelCoordinates["front_bottom_x"], wheelCoordinates["front_bottom_y"]
        bbx, bby = wheelCoordinates["back_bottom_x"], wheelCoordinates["back_bottom_y"]
        cbx, cby = carCoordinates["cars_bottom_left_x"], carCoordinates["cars_bottom_left_y"]
        ctx, cty = carCoordinates["cars_top_left_x"], carCoordinates["cars_top_left_y"]
        crbx, crby = carCoordinates["cars_bottom_right_x"], carCoordinates["cars_bottom_right_y"]

        # Leftmost points
        cbxr, cbyr = cbx - 60, bby - 70 
        ctxr, ctyr = ctx, cbyr + (bby - cty) + 180

            # Middle points
        mbxr, mbyr = crbx - 50 , fby + 120 
        mtxr, mtyr = crbx + 100, mbyr + 440

        # Rightmost points
        cbrxr, cbryr = crbx + 50 , bby -50
        ctrxr, ctryr = cbrxr - 150 , cbryr + bby + 150



        # Print the calculated points
        print(f"Leftmost Bottom Point: ({cbxr}, {cbyr})")
        print(f"Leftmost Top Point: ({ctxr}, {ctyr})")
        print(f"Rightmost Bottom Point: ({cbrxr}, {cbryr})")
        print(f"Rightmost Top Point: ({ctrxr}, {ctryr})")
        print(f"Middle Bottom Point: ({mbxr}, {mbyr})")
        print(f"Middle Top Point: ({mtxr}, {mtyr})")

        # Return the calculated points
        return {
            "ref_leftmost_bottom": (cbxr, cbyr),
            "ref_leftmost_top": (ctxr, ctyr),
            "ref_rightmost_bottom": (cbrxr, cbryr),
            "ref_rightmost_top": (ctrxr, ctryr),
            "ref_middle_bottom": (mbxr, mbyr),
            "ref_middle_top": (mtxr, mtyr),
        }
    except FileNotFoundError as e:
        log_to_json(f"File not found error {e}", current_file_name)
        return f'Sorry the image is not processed for: File not found: {str(e)}'
    except ValueError as e:
        log_to_json(f"ValueError {e}", current_file_name)
        return f'Sorry the image is not processed for: Value error: {str(e)}'
    except Exception as e:
        log_to_json(f"Exceptional error {e}", current_file_name)
        return  f'Sorry : An unexpected error occurred: {str(e)}'
    


    
def modify_perspective_points(perspective_points):
    try:
        threshold = 200
        left_top_x , left_top_y = perspective_points["left_top"]
        right_top_x , right_top_y = perspective_points["right_top"]
        left_bottom_x, left_bottom_y = perspective_points["left_bottom"]
        right_bottom_x, right_bottom_y = perspective_points["right_bottom"]
        
        left_top_x , left_top_y = 0 , left_top_y - 40
        right_top_x , right_top_y = right_bottom_x  , right_top_y 
        left_bottom_x, left_bottom_y = 0, left_bottom_y 
        
        
        modified_points = {
                "left_top": (left_top_x, left_top_y),
                "left_bottom": (left_bottom_x, left_bottom_y),
                "right_top": (right_top_x, right_top_y),
                "right_bottom": perspective_points["right_bottom"]
            }
                    
        return modified_points
    except Exception as e:
        log_to_json(f"Exceptional error {e}", current_file_name)
        return  f'Sorry : An unexpected error occurred: {str(e)}'



def calculate_pos_angle(coord1, coord2, angle_value_type):
 

    try:
        # Extract x and y components
        x1, y1 = coord1
        x2, y2 = coord2

        # Calculate differences
        delta_x = x2 - x1
        delta_y = y2 - y1

        # Calculate the angle in radians
        angle_radians = math.atan2(delta_y, delta_x)

        # Convert radians to degrees
        angle_degrees = math.degrees(angle_radians)

        # Adjust angle based on angle_value_type
        if angle_value_type == "pos" and angle_degrees < 0:
            angle_degrees += 180

        # Prepare the result in dictionary format
        result = {
            "degree": round(angle_degrees, 2),  # Angle in degrees (rounded to 2 decimals)
            "val": round(angle_radians, 4)  # Angle in radians (rounded to 4 decimals)
        }

        return result  # Return as a dictionary
    except Exception as e:
        log_to_json(f"Exceptional error {e}", current_file_name)
        return f'Sorry : An unexpected error occurred: {str(e)}'


def find_point_by_angle_and_distance(point, angle, distance, x_value_type):
    try:
        x1, y1 = point

        # Convert the angle to radians for trigonometric calculations
        angle_rad = math.radians(angle)

        # Calculate the two possible points
        x2_1 = x1 + distance * math.cos(angle_rad)
        y2_1 = y1 + distance * math.sin(angle_rad)

        x2_2 = x1 - distance * math.cos(angle_rad)
        y2_2 = y1 - distance * math.sin(angle_rad)

        point1 = (x2_1, y2_1)
        point2 = (x2_2, y2_2)

        # Return the point based on x_value_type
        if x_value_type == "min":
            return point1 if x2_1 < x2_2 else point2
        elif x_value_type == "max":
            return point1 if x2_1 > x2_2 else point2
        else:
            log_to_json("x_value_type must be either 'min' or 'max'", current_file_name)
    except Exception as e:
        log_to_json(f"Exceptional error {e}", current_file_name)
        return  f'Sorry : An unexpected error occurred: {str(e)}'



def calculate_distance(point1, point2):

    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return int(distance)




def calculate_shadow_points(wheelCoordinates: Dict[str, float], carCoordinates: Dict[str, float]) -> Tuple[List[Dict], List[Dict]]:
    try:
    # Extract the wheel coordinates from the provided dictionary
        flx, fly = wheelCoordinates["front_left_x"], wheelCoordinates["front_left_y"]
        fbx, fby = wheelCoordinates["front_bottom_x"], wheelCoordinates["front_bottom_y"]
        bbx, bby = wheelCoordinates["back_bottom_x"], wheelCoordinates["back_bottom_y"]
        cbx, cby = carCoordinates["cars_bottom_left_x"], carCoordinates["cars_bottom_left_y"]
        ctx, cty = carCoordinates["cars_top_left_x"], carCoordinates["cars_top_left_y"]
        crbx, crby = carCoordinates["cars_bottom_right_x"], carCoordinates["cars_bottom_right_y"]
        
        # Calculate the points as needed for shadow calculation
        # Leftmost points (calculated based on wheel coordinates)
        cbxs, cbys = cbx+ 40, cby + 40

        # Rightmost points
        cbrxs, cbrys = bbx + 50, bby - 40 
        ctrxs, ctrys = bbx - 100, fly 

        # Middle points
        mbxs, mbys = flx, fby + 20

        # Print the calculated points
        print(f"shadow Leftmost Bottom Point: ({cbxs}, {cbys})")
        print(f"shadow rightmost bottom Point: ({cbrxs}, {cbrys})")
        print(f"Shadow Rightmost top Point: ({ctrxs}, {ctrys})")
        print(f"shadow middle Point: ({mbxs}, {mbys})")

        # Define curve_pts and straight_pts based on the calculated points
        curve_pts = [
            {"points": [(cbxs, cbys), (mbxs, mbys)], "direction": "counterclockwise"}
        ]

            # Change from set to tuple for straight_pts
        straight_pts = [
            {(mbxs, mbys), (cbrxs, cbrys)},
            {(cbrxs, cbrys), (ctrxs, ctrys)},
            {(cbxs, cbys), (ctrxs, ctrys)}  # Fixed mistake with (cbys, cbys)
        ]
        
        all_points = {
            "cbxs": cbxs,
            "cbys": cbys,
            "mbxs": mbxs,
            "mbys": mbys,
            "cbrxs": cbrxs,
            "cbrys": cbrys,
            "ctrxs": ctrxs,
            "ctrys": ctrys
        }

        return all_points
    except FileNotFoundError as e:
        log_to_json(f"File not found error {e}", current_file_name)
        return f'Sorry the image is not processed for: File not found: {str(e)}'
    except ValueError as e:
        log_to_json(f"ValueError {e}", current_file_name)
        return f'Sorry the image is not processed for: Value error: {str(e)}'
    except Exception as e:
        log_to_json(f"Exceptional error {e}", current_file_name)
        return  f'Sorry : An unexpected error occurred: {str(e)}'




def save_image_with_timestamp(IMAGE_PATH, file_location: str, file_name: str):
    try:
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
    except Exception as e:
        log_to_json(f"Error occurs for {e}", current_file_name)
        return None



def remove_all_images(folder_path: str):
    # Supported image file extensions
    try:
        image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}

        # Counter for the number of images removed
        removed_count = 0

        # Ensure the folder exists
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"The folder {folder_path} does not exist.")

        # Iterate through all files in the folder
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)

            # Check if it's a file and has an image extension
            if os.path.isfile(file_path) and os.path.splitext(file_name)[1].lower() in image_extensions:
                os.remove(file_path)
                removed_count += 1

        return removed_count
    except Exception as e:
        log_to_json(f"Error occurs for {e}", current_file_name)
        return None



#save_image_with_timestamp(byte_image, file_location, file_name)
    



#curve_pts, straight_pts = calculate_shadow_points(calculated_points)

