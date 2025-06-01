import os
from rembg import remove
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from datetime import datetime
from io import BytesIO
import requests
import math
import io
import json
import re
import cv2
import numpy as np
from features.core.licenseplateoperations.licenseplateblurfunc import make_the_licenseplate_blur
from features.core.licenseplateoperations.licenseplateimagefunc import addLicensePlateImage
from features.functionalites.backgroundoperations.addbackground.addwallinitiate import addBackgroundInitiate
from features.functionalites.car_straightline_angle.carbottomcoordinates import find_car_bottom_points
from my_logging_script import log_to_json
from features.functionalites.backgroundoperations.roundedsegmented import detect_left_right_wheels, detect_cars_coordinates, detect_wheels_and_annotate
from features.functionalites.backgroundoperations.utlis import find_line_angle

car_bottom_height_threshold = 0
middle_point_threshold = 920
current_file_name= "backgroundconfs.py"
def calculate_angle(coord1, coord2):
    # Extract x and y components
    try:
        x1, y1 = coord1
        x2, y2 = coord2

        # Calculate differences
        delta_x = x2 - x1
        delta_y = y2 - y1

        # Calculate the angle in radians
        angle_radians = math.atan2(delta_y, delta_x)

        # Convert radians to degrees
        angle_degrees = math.degrees(angle_radians)

        # Prepare the result in dictionary format (no JSON encoding here)
        result = {
            "degree": round(angle_degrees , 2),  # Angle in degrees (rounded to 2 decimals)
            # "val": round(angle_radians +  0.0874886635 , 4)     # tan⁻¹ value (angle in radians, rounded to 4 decimals)
            "val": round(angle_radians  , 4)  
        }

        return result  # Return as a dictionary
    except Exception as e:
        log_to_json(f"calculate_angle: Exceptional error {e}", current_file_name)
        return  None
    

def validate_and_rearrange_points(points):

    if len(points) != 4:
        raise ValueError("Exactly 4 points are required for perspective warp.")
    
    # Convert points to numpy array for easy manipulation
    points = np.array(points, dtype="float32")
    
    # Calculate the centroid of the points
    centroid = np.mean(points, axis=0)
    
    # Sort points based on their angle with respect to the centroid
    def angle_from_centroid(point):
        return np.arctan2(point[1] - centroid[1], point[0] - centroid[0])
    
    sorted_points = sorted(points, key=angle_from_centroid)
    
    # Ensure sorted points form a convex quadrilateral
    def is_convex(points):
        """Check if the given quadrilateral is convex."""
        cross_products = []
        for i in range(4):
            # Get three consecutive points
            p1 = points[i]
            p2 = points[(i + 1) % 4]
            p3 = points[(i + 2) % 4]
            
            # Calculate the cross product of vectors (p2 - p1) and (p3 - p2)
            v1 = p2 - p1
            v2 = p3 - p2
            cross_product = v1[0] * v2[1] - v1[1] * v2[0]
            cross_products.append(cross_product)
        
        # If all cross products have the same sign, the shape is convex
        return all(cp > 0 for cp in cross_products) or all(cp < 0 for cp in cross_products)
    
    sorted_points = np.array(sorted_points)
    if not is_convex(sorted_points):
        return False, "Points cannot form a valid convex quadrilateral."
    
    # Return the sorted points in clockwise order
    return True, sorted_points.tolist()


def calculate_line_and_distant_point(m, coord, distance):
    try:
        x, y = coord

        # Step 1: Calculate c using the equation y = mx + c
        c = y - m * x

        # Step 2: Find the direction vector for the line
        dx = 1 / math.sqrt(1 + m**2)  # x-component of the unit vector
        dy = m / math.sqrt(1 + m**2)  # y-component of the unit vector

        # Scale the direction vector to the desired distance
        # distance = 4000
        dx *= -distance  # Negative to ensure negative x direction
        dy *= -distance

        # Calculate the new point
        new_x = x + dx
        new_y = y + dy

        # Return c and the distant point
        return {
            "c": round(c, 4),
            "distant_point": (int(new_x), int(new_y))
        }
    except Exception as e:
        log_to_json(f"calculate_line_and_distant_point: Exceptional error {e}", current_file_name)
        return  None
    




def get_dynamic_wall_coordinates(image):
    # Define the coordinates in this function or pass them dynamically
    try:
        # bottom_left, bottom_right = find_car_bottom_points(image)

        carCoordinates = detect_cars_coordinates(image)

        cbx, cby = carCoordinates["cars_bottom_left_x"], carCoordinates["cars_bottom_left_y"]
        ctx, cty = carCoordinates["cars_top_left_x"], carCoordinates["cars_top_left_y"]

        wheelCoordinates = detect_wheels_and_annotate(image, 'outputdebug')
        if wheelCoordinates is None:
            return None
        x1, y1 = wheelCoordinates["front_bottom_x"], wheelCoordinates["front_bottom_y"]
        x2, y2 = wheelCoordinates["back_bottom_x"], wheelCoordinates["back_bottom_y"]
        #angle_from_wheel = find_line_angle(x1, y1, x2, y2)
        log_to_json(f"Car Coordinates valuessssss ===== bottom ({cbx} {cby})   top ({ctx} {cty})", current_file_name)

        half_distance = int((cby-cty)/2)
        if half_distance > 150 :  
            half_distance = 150
        # calculate middle point

        image.seek(0)  # Ensure we're at the start of the stream
        file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        height = img.shape[0] 

        

        middle_point_y = 1440 - height - car_bottom_height_threshold + cty + half_distance
        
        if middle_point_y < middle_point_threshold:
            middle_point_y = middle_point_threshold
        middle_point_x = cbx + 60

        middle_point = (middle_point_x,  middle_point_y)





        # Get angle values
        #angle_values = calculate_angle(bottom_left, bottom_right)
        wheel_left = (x1,y1)
        wheel_right = (x2, y2)
        angle_values = calculate_angle(wheel_left, wheel_right) 
        log_to_json(f"values of coordinate: {wheel_left} ,,, {wheel_right},, angle: {angle_values}", current_file_name)

        #calculate reference point in canvas middle
        verticle_line = 1253
        assump_value = -38
        if angle_values['degree'] < 0:
            assump_value = angle_values['degree']

        one_third_angle = assump_value/3
        backwall_rad = math.radians(backwall_calculate(assump_value))
        log_to_json(f"backwall angle :{backwall_calculate(assump_value)} - {backwall_rad} from assump value {assump_value}", current_file_name)

        
        
        one_third_radial = math.radians(one_third_angle)
        extra_angle_rad = 0
        if abs(assump_value) > 35 and abs(assump_value) < 45:
            extra_angle = abs(assump_value/3) - abs(assump_value/5)
            print(f"{abs(assump_value)} extra angle {extra_angle}")
            extra_angle_rad = math.radians(extra_angle) 
        elif abs(assump_value) >= 45:
            # verticle_line = 1000
            extra_angle = abs(assump_value/3) - abs(assump_value/7)
            print(f"{abs(assump_value)} extra angle {extra_angle}")
            extra_angle_rad = math.radians(extra_angle)

        canvas_middle_ref = find_intersection(middle_point, verticle_line, one_third_radial)
        can_midx, can_midy = canvas_middle_ref
        # if can_midy < 720 and abs(angle_values['degree']) > 45:
        #     can_midy = 720
        #     canvas_middle_ref = (can_midx, can_midy)

        # calculate left_wall left bottom point
        verticle_line_0  = 0
        lwall_bottom_left = find_intersection(canvas_middle_ref, verticle_line_0, one_third_radial)

        log_to_json(f"Canvas middle point : {canvas_middle_ref} , distance_middle: {middle_point} height: {height}")
        #calculate_left wall top right
        lwall_top_left =  (0,0)
        lwall_top_right = find_intersection(lwall_top_left, verticle_line, -one_third_radial)
        lwtrx , lwtry = lwall_top_right

        print(f"left wall right top {lwall_top_right} {type(lwall_top_right)} {type(lwtrx)}")
        if(lwtry > 130):
            lwtry = 130
        
        lwall_top_right = (lwtrx, lwtry)


        #calculate right wall right bottom
        verticle_line_1920 = 1920
        rwall_right_bottom = find_intersection(canvas_middle_ref, verticle_line_1920, -(backwall_rad))

        #calculate right wall right top
       
        rwall_right_top = find_intersection(lwall_top_right, verticle_line_1920, (backwall_rad))

        #calculate floor right
        # verticle_line_floor = 1920 + 1253
        # floor_right_top = find_intersection(canvas_middle_ref, verticle_line_floor, one_third_radial)


        log_to_json(f"left wall left bottom : {lwall_bottom_left} \n left wall left top : {lwall_top_left} \n right wall right bottom : {rwall_right_bottom} \n right wall top left : {rwall_right_top}", current_file_name)
        log_to_json(f"prev coordinate right wall {find_intersection(canvas_middle_ref, verticle_line_1920, -one_third_radial)}", current_file_name)

        #---------------prev calculation-----------------------------
        # ref_point = (1253, 651)
        ref_point_parallel = (4198, 1321)

        # Calculate line and distant points
        get_eq_value = calculate_line_and_distant_point(one_third_radial, canvas_middle_ref, 4000)
        floor_parallel_val = calculate_line_and_distant_point(one_third_radial, canvas_middle_ref, 4000)

        print(f"equation values {get_eq_value}, parallel points {floor_parallel_val}")

        # Define wall coordinates
        left_mid = (0, get_eq_value['c'])
        corner_right = (0, -100)
        mid_top = (1253, 144)
        # mid_bottom = ref_point
        right_corner = (1920, -700)
        right_top = (1920, 79)
        right_mid = (1920, 780)

        floor_left_top = get_eq_value['distant_point']
        floor_left_bottom_json = calculate_line_and_distant_point(-one_third_radial, floor_left_top, 3000) #floor_parallel_val['distant_point']
        floor_left_bottom =  (800, 2500) # get_eq_value['distant_point']
        floor_right_bottom = find_intersection(canvas_middle_ref, 3000, -(backwall_rad))
        floor_right_top = canvas_middle_ref

        ceiling_right_angle = calculate_angle(mid_top, right_top)
        ceiling_left_angle = calculate_angle(mid_top, corner_right)

        get_ceiling_left_points = calculate_line_and_distant_point(ceiling_left_angle['val'], mid_top, 1920)
        get_ceiling_right_points = calculate_line_and_distant_point(ceiling_right_angle['val'], mid_top, 1440)
        get_ceiling_top_points = calculate_line_and_distant_point(ceiling_left_angle['val'], get_ceiling_right_points['distant_point'], 1440)

        # get_floor_valuie = get_ceiling_top_points = calculate_line_and_distant_point(tuple(ref_point), tuple(right_mid), 3000)
        # print(f"floor value is : {get_floor_valuie}")

        
        print(f"celeinng points {get_ceiling_left_points} {get_ceiling_right_points} {get_ceiling_top_points}")

        floor_coordinates = [floor_left_top, floor_left_bottom, floor_right_bottom, floor_right_top]
        #is_valid, floor_coordinates = validate_and_rearrange_points(floor_coordinates_check)

        
        left_wall_coordinates = [lwall_top_right, lwall_top_left, lwall_bottom_left, canvas_middle_ref]
        right_wall_coordinates = [lwall_top_right, canvas_middle_ref, rwall_right_bottom, rwall_right_top]
        ceiling_coordinates = [lwall_top_left, lwall_top_right, rwall_right_top, right_corner]

        log_to_json(f"floor coordinates: {floor_coordinates} \n left wall coordinates: {left_wall_coordinates} \n right wall coordinates {right_wall_coordinates} \n ceiling coordinates {ceiling_coordinates}", current_file_name)
    
        return floor_coordinates, left_wall_coordinates, right_wall_coordinates, ceiling_coordinates
    except Exception as e:
        log_to_json(f"get_dynamic_wall_coordinates: Exceptional error {e}", current_file_name)
        return  None
    

def find_intersection(point, verticle_line, angle):

    x1, y1 = point
    x2 = verticle_line  # x-coordinate of the vertical line
    
    # Convert the angle to radians
    # angle_rad = math.radians(angle)
    
    # Calculate the slope (tan of the angle)
    if math.isclose(math.cos(angle), 0, abs_tol=1e-9):  # Handle case where angle is vertical
        raise ValueError("The line is vertical and does not intersect the vertical line at a finite point.")
    
    slope = math.tan(angle)
    
    # Calculate the y-coordinate of the intersection point
    y2 = int(slope * (x2 - x1) + y1)

    return (x2, y2)


def get_dynamic_wall_reverse_coordinates(image):
    # Define the coordinates in this function or pass them dynamically
    try:
        # bottom_left, bottom_right = find_car_bottom_points(image)

        carCoordinates = detect_cars_coordinates(image)
        print(carCoordinates)

        cbx, cby = carCoordinates["cars_bottom_right_x"], carCoordinates["cars_bottom_right_y"]
        ctx, cty = carCoordinates["cars_top_right_x"], carCoordinates["cars_top_right_y"]

        wheelCoordinates = detect_left_right_wheels(image)
        if wheelCoordinates is None:
            return None
        x1, y1 = wheelCoordinates["front_bottom_x"], wheelCoordinates["front_bottom_y"]
        x2, y2 = wheelCoordinates["back_bottom_x"], wheelCoordinates["back_bottom_y"]
        #angle_from_wheel = find_line_angle(x1, y1, x2, y2)
        print(f"Car Coordinates valuessssss ===== bottom ({cbx} {cby})   top ({ctx} {cty})")
        print(f"Car wheel coordinate valuessssss ===== left ({x1} {y1})   right ({x2} {y2})")

        half_distance = int((cby-cty)/2)
        if half_distance > 450 :  
            half_distance = 450
        # calculate middle point

        image.seek(0)  # Ensure we're at the start of the stream
        file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        height = img.shape[0] 
        print(f"Image height ==> {height}")

        

        middle_point_y = 1440 - height - car_bottom_height_threshold + cty + half_distance
        if middle_point_y < middle_point_threshold:
            middle_point_y = middle_point_threshold
        middle_point_x = cbx + 60 

        middle_point = (middle_point_x,  middle_point_y)





        # Get angle values
        #angle_values = calculate_angle(bottom_left, bottom_right)
        wheel_left = (x1,y1)
        wheel_right = (x2, y2)
        angle_values = calculate_angle(wheel_left, wheel_right) 
        if isinstance(angle_values, dict) and "degree" in angle_values:
            angle_values["degree"] += 180
        # print(f"values of coordinate: {bottom_left} ,,, {bottom_right},, angle: {angle_values}")
        log_to_json(f"Applicable angle : {angle_values}")
        #calculate reference point in canvas middle
        verticle_line = 615
        one_third_angle = (angle_values['degree'])/3
        one_third_radial = math.radians(one_third_angle)

        backwall_rad = math.radians(backwall_calculate(angle_values['degree']))
        log_to_json(f"backwall angle :{backwall_calculate(angle_values['degree'])} - {backwall_rad} from assump value {angle_values['degree']}", current_file_name)

        canvas_middle_ref = find_intersection(middle_point, verticle_line, one_third_radial)

        # calculate right wall right bottom point
        verticle_line_0  = 0
        verticle_line_1920 = 1920
        rwall_right_bottom = find_intersection(canvas_middle_ref, verticle_line_1920, one_third_radial)

        print(f"Canvas middle point : {canvas_middle_ref} , distance_middle: {middle_point} height: {height}")
        #calculate_left wall top right
        rwall_top_right =  (1920,0)
        rwall_top_left = find_intersection(rwall_top_right, verticle_line, -one_third_radial)

        rwtlx, rwtly = rwall_top_left
        if(rwtly > 100):
            rwtly = 100
        
        rwall_top_left = (rwtlx, rwtly)

        #calculate left wall left bottom
        
        lwall_left_bottom = find_intersection(canvas_middle_ref, verticle_line_0, -backwall_rad)

        #calculate left wall top left
       
        lwall_left_top = find_intersection(rwall_top_left, verticle_line_0, backwall_rad)

        #calculate floor right
        # verticle_line_floor = 1920 + 1253
        # floor_right_top = find_intersection(canvas_middle_ref, verticle_line_floor, one_third_radial)


        log_to_json(f"left wall left bottom : {lwall_left_bottom} \n left wall left top : {lwall_left_top} \n right wall right bottom : {rwall_right_bottom} \n right wall top left : {rwall_top_left}", current_file_name)
      
        

        #---------------prev calculation-----------------------------
        # ref_point = (1253, 651)
        # ref_point_parallel = (4198, 1321)

        # Calculate line and distant points
        get_eq_value = calculate_line_and_distant_point(one_third_radial, canvas_middle_ref, 4000)
        floor_parallel_val = calculate_line_and_distant_point(one_third_radial, canvas_middle_ref, 4000)


        # Define wall coordinates
        left_mid = (0, get_eq_value['c'])
        corner_right = (0, -100)
        mid_top = (1253, 144)
        # mid_bottom = ref_point
        ceiling_top = (0, -700)
        right_top = (1920, 79)
        right_mid = (1920, 780)


        get_floo_left_top = calculate_line_and_distant_point(-backwall_rad, canvas_middle_ref, 1920)
        floor_left_top =  get_floo_left_top['distant_point'] #get_eq_value['distant_point']
        floor_left_bottom = (615, 3120) #floor_parallel_val['distant_point']
        floor_right_bottom = find_intersection(canvas_middle_ref, 3220, one_third_radial)
        floor_right_top = canvas_middle_ref


        ceiling_right_angle = calculate_angle(mid_top, right_top)
        ceiling_left_angle = calculate_angle(mid_top, corner_right)

        get_ceiling_left_points = calculate_line_and_distant_point(ceiling_left_angle['val'], mid_top, 1920)
        get_ceiling_right_points = calculate_line_and_distant_point(ceiling_right_angle['val'], mid_top, 1440)
        get_ceiling_top_points = calculate_line_and_distant_point(ceiling_left_angle['val'], get_ceiling_right_points['distant_point'], 1440)

        # get_floor_valuie = get_ceiling_top_points = calculate_line_and_distant_point(tuple(ref_point), tuple(right_mid), 3000)
        # print(f"floor value is : {get_floor_valuie}")

        


        floor_coordinates = [floor_left_top, floor_left_bottom, floor_right_bottom, floor_right_top]
        left_wall_coordinates = [rwall_top_left, lwall_left_top, lwall_left_bottom, canvas_middle_ref]
        right_wall_coordinates = [rwall_top_left, canvas_middle_ref, rwall_right_bottom, rwall_top_right]
        ceiling_coordinates = [lwall_left_top, rwall_top_left, rwall_top_right, ceiling_top]

        log_to_json(f"floor coordinates: {floor_coordinates} \n left wall coordinates: {left_wall_coordinates} \n right wall coordinates {right_wall_coordinates} \n ceiling coordinates {ceiling_coordinates}", current_file_name)

        return floor_coordinates, left_wall_coordinates, right_wall_coordinates, ceiling_coordinates
    except Exception as e:
        log_to_json(f"get_dynamic_wall_reverse_coordinates: Exceptional error {e}", current_file_name)
        return None


def get_wall_coordinates():
    # Define the coordinates in this function or pass them dynamically
    try:
        left_top = (0, 0)
        left_mid = (0, 920)
        left_bottom = (0, 2440)
        corner_right = (0, -100)
        mid_top = (1253, 144)
        mid_bottom = (1253, 751)
        right_corner = (1920, 0)
        right_top = (1920, 79)
        right_mid = (1920, 880)
        right_bottom = (1920, 1440)
        mid_finish_right = (1920, 516)

        floor_left_top = (-2627,1273)
        floor_left_bottom= (318, 2293)
        floor_right_bottom = (4198, 1321)
        floor_right_top = (1253, 751)

        floor_coordinates = [floor_left_top, floor_left_bottom, floor_right_bottom, floor_right_top]
        left_wall_coordinates = [mid_top , corner_right, left_mid, mid_bottom]
        right_wall_coordinates = [mid_top, mid_bottom, right_mid, right_top]
        ceiling_coordinates = [corner_right, mid_top, right_top, right_corner]

        return floor_coordinates, left_wall_coordinates, right_wall_coordinates, ceiling_coordinates
    except Exception as e:
        log_to_json(f"get_wall_coordinates: Exceptional error {e}", current_file_name)
        return  None


def get_basic_wall_coordinates(coord):
    # Define the coordinates in this function or pass them dynamically
    try:
        wall_size = 1440
        wall_left_top_y = coord['top_y'] - wall_size
        wall_left_top = (0, wall_left_top_y)
        # wall_left_top = (0,0)
        wall_left_bottom = (0,coord['top_y'])
        wall_right_bottom = (1920, coord['top_y'])
        # wall_right_top = (1920,0)
        wall_right_top = (1920, wall_left_top_y)

        floor_left_top = wall_left_bottom
        bottom =  1440
        floor_left_bottom = (-1200,bottom)
        floor_right_bottom = (3200, bottom)
        floor_right_top = wall_right_bottom

        floor_coordinates = [floor_left_top, floor_left_bottom, floor_right_bottom, floor_right_top]
        wall_coordinates = [wall_left_top, wall_left_bottom, wall_right_bottom, wall_right_top]

        return floor_coordinates, wall_coordinates
    except Exception as e:
        log_to_json(f"get_basic_wall_coordinates: Exceptional error {e}", current_file_name)
        return  None

def get_wall_coordinates_recverse_angle():
    # Define the coordinates in this function or pass them dynamically
    try:
        ceil_left_top = (0, 0)
        ceil_left_bottom = (0, 63)
        ceil_right_bottom = (612, 165)
        ceil_right_top = (0, 564)

        leftwall_left_top = ceil_left_bottom
        leftwall_left_bottom = (0,850)
        leftwall_right_top = ceil_right_bottom
        leftwall_right_bottom = (612, 800)

        rightwall_left_top = leftwall_right_top
        rightwall_left_bottom = leftwall_right_bottom
        rightwall_right_top = (1920, -120)
        rightwall_right_bottom = (1920, 1000)

        floor_left_top = (0, 574)
        floor_left_bottom = (0, 1440)
        floor_right_top = rightwall_right_bottom
        floor_right_bottom = (1920, 2000)


        floor_coordinates = [floor_left_top, floor_left_bottom, floor_right_bottom, floor_right_top]
        left_wall_coordinates = [leftwall_left_top , leftwall_left_bottom, leftwall_right_bottom, leftwall_right_top]
        right_wall_coordinates = [rightwall_left_top, rightwall_left_bottom, rightwall_right_bottom, rightwall_right_top]
        ceiling_coordinates = [ceil_left_top, ceil_left_bottom, ceil_right_bottom, rightwall_right_top]

        return floor_coordinates, left_wall_coordinates, right_wall_coordinates, ceiling_coordinates
    except Exception as e:
        log_to_json(f"get_wall_coordinates_recverse_angle: Exceptional error {e}", current_file_name)
        return  None


def contains_string(main_string, sub_string):
    try:
        if not isinstance(main_string, str) or not isinstance(sub_string, str):
            raise ValueError("Both arguments must be strings.")
        
        return sub_string in main_string
    except Exception as e:
        log_to_json(f"contains_string: Exceptional error {e}", current_file_name)
        return  False

def get_dynamic_wall_images(image_path: str) -> dict:
    # Define the prefix for non-system paths
    try:
        prefix = "https://vroomview.obs.eu-de.otc.t-systems.com/"

        # Extract the counter number from the string using regex
        match = re.search(r'photo_box_(\d+)\.[a-zA-Z0-9]+', image_path)
        #is_basic = contains_string(image_path, "basic")

        
        # if not match:
        #     raise ValueError("Invalid image path format or no counter number found.")

        counter = match.group(1)
        base_path = None
        print(f"Counter value is {counter}")


        if image_path.startswith("system"):
            # For system-based path
            extens = ".jpg"
            base_path = image_path.split("bg_type/")[-1].replace("\\", "/")
            wall_images = {
                "floor_wall_img": f'assets/backgrounds/{counter}/floor{extens}',
                "left_wall_img": f'assets/backgrounds/{counter}/lw{extens}',
                "right_wall_img": f'assets/backgrounds/{counter}/rw{extens}',
                "ceiling_wall_img": f'assets/backgrounds/{counter}/ceiling{extens}'
            }

            print(f"match {match} counter {counter}")
            print(f"{wall_images}")

            wall_images_bytes = {}
            for wall, image_path in wall_images.items():
                with open(image_path, 'rb') as f:
                    img = Image.open(f)
                    img = resize_image(img, 1920)  # Resize image to 1920 width
                    img_byte_io = BytesIO()
                    img.save(img_byte_io, format='JPEG')
                    img_byte_io.seek(0)
                    wall_images_bytes[wall] = img_byte_io

            return wall_images_bytes

        else:
            # For non-system path
            base_path = '/'.join(image_path.split('/')[:-2])  # Keep everything except the last part (processed/bg_<counter>.jpg)
            # extension = image_path.split('.')[-1]
            extension = ".jpg"

            urls = {
                    "floor_wall_img": f"{prefix}{base_path}/floor_{counter}{extension}",
                    "left_wall_img": f"{prefix}{base_path}/lw_{counter}{extension}",
                    "right_wall_img": f"{prefix}{base_path}/rw_{counter}{extension}",
                    "ceiling_wall_img": f"{prefix}{base_path}/ceiling_{counter}{extension}",
                }
 

            wall_images_bytes = {}

            print(f"bg urls new ==> {urls}")

            # Check if the image exists and download it
            for wall, url in urls.items():
                response = requests.get(url)
                if response.status_code == 200:
                    img = Image.open(BytesIO(response.content))
                    img = resize_image(img, 1920)
                    img_byte_io = BytesIO()
                    img.save(img_byte_io, format='JPEG')  # Save image in PNG format
                    img_byte_io.seek(0)
                    wall_images_bytes[wall] = img_byte_io
                else:
                    wall_images_bytes[wall] = None  # If the image doesn't exist, set to None

            return wall_images_bytes
    except Exception as e:
        log_to_json(f"get_dynamic_wall_images: Exceptional error here hahaha{e}", current_file_name)
        return  None
    
def resize_image(image: Image.Image, target_width: int) -> Image.Image:
    aspect_ratio = image.height / image.width
    target_height = int(target_width * aspect_ratio)
    resized_image = image.resize((target_width, target_height), Image.LANCZOS)
    return resized_image
    
def get_universal_wall_images(image_path: str) -> dict:
    # Define the prefix for non-system paths
    try:
        prefix = "https://vroomview.obs.eu-de.otc.t-systems.com/"

        # Extract the counter number from the string using regex
        match = re.search(r'universal_(\d+)\.[a-zA-Z0-9]+', image_path)
        #is_basic = contains_string(image_path, "basic")

        
        # if not match:
        #     raise ValueError("Invalid image path format or no counter number found.")

        counter = match.group(1)
        base_path = None
        print(f"Counter value is {counter}")


        if image_path.startswith("system"):
            # For system-based path
            base_path = image_path.split("bg_type/")[-1].replace("\\", "/")
            image_path = f'assets/universal/{counter}/floor.jpg'

            with open(image_path, 'rb') as f:
                img_byte_io = BytesIO(f.read())
                img_byte_io.seek(0)  # Reset the stream position
                return img_byte_io

        else:
            # For non-system path
            base_path = '/'.join(image_path.split('/')[:-2])  # Keep everything except the last part (processed/bg_<counter>.jpg)
            # extension = image_path.split('.')[-1]
            extension = ".jpg"
            url = f"{prefix}{base_path}/universal_{counter}{extension}"

            print(f"Downloading image from URL: {url}")

            # Check if the image exists and download it
            response = requests.get(url)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                img_byte_io = BytesIO()
                img.save(img_byte_io, format='JPEG')  # Save image in PNG format
                img_byte_io.seek(0)  # Reset the stream position
                return img_byte_io
            else:
                log_to_json(f"Universal Image not found at URL: {url}", current_file_name)
            
    except Exception as e:
        log_to_json(f"get_universal_wall_images: Exceptional error here hahaha{e}", current_file_name)
        return  None
    
def get_dynamic_basic_wall_images(image_path: str) -> dict:
    # Define the prefix for non-system paths
    try:
        prefix = "https://vroomview.obs.eu-de.otc.t-systems.com/"

        # Extract the counter number from the string using regex
        match = re.search(r'basic_(\d+)\.[a-zA-Z0-9]+', image_path)
        # if not match:
        #     raise ValueError("Invalid image path format or no counter number found.")

        counter = match.group(1)
        base_path = None

        if image_path.startswith("system"):
            # For system-based path
            base_path = image_path.split("bg_type/")[-1].replace("\\", "/")
            extens = ".jpg"
            wall_images = {
                "floor_wall_img": f'assets/basicbg/{counter}/floor{extens}',
                "left_wall_img": f'assets/basicbg/{counter}/wall{extens}',
            }

            wall_images_bytes = {}

            print(f"match {match} counter {counter}")
            print(f"{wall_images}")

            for wall, image_path in wall_images.items():
                with open(image_path, 'rb') as f:
                    img = Image.open(f)
                    img = resize_image(img, 1920)  # Resize image to 1920 width
                    img_byte_io = BytesIO()
                    img.save(img_byte_io, format='JPEG')
                    img_byte_io.seek(0)
                    wall_images_bytes[wall] = img_byte_io

            return wall_images_bytes

        else:
            # For non-system path
            base_path = '/'.join(image_path.split('/')[:-2])  # Keep everything except the last part (processed/bg_<counter>.jpg)
            # extension = image_path.split('.')[-1]
            extension = ".jpg"

            # Generate URLs for the walls
            urls = {
                "floor_wall_img": f"{prefix}{base_path}/floor_{counter}{extension}",
                "left_wall_img": f"{prefix}{base_path}/lw_{counter}{extension}",
            }

            wall_images_bytes = {}

            print(f"bg urls {urls}")

            # Check if the image exists and download it
            for wall, url in urls.items():
                response = requests.get(url)
                if response.status_code == 200:
                    img = Image.open(BytesIO(response.content))
                    img = resize_image(img, 1920)  # Resize image to 1920 width
                    img_byte_io = BytesIO()
                    img.save(img_byte_io, format='JPEG')  # Save image in PNG format
                    img_byte_io.seek(0)
                    wall_images_bytes[wall] = img_byte_io
                else:
                    wall_images_bytes[wall] = None  # If the image doesn't exist, set to None

            return wall_images_bytes
    except Exception as e:
        log_to_json(f"get_dynamic_basic_wall_images: Exceptional error {e}", current_file_name)
        return  None

def get_wall_images():
    try:
        wallImages = {
            "floor_wall_img": 'outputs/backgrounds/floorwoord.png',
            "left_wall_img": 'outputs/backgrounds/leftwall1.png',
            "right_wall_img": 'outputs/backgrounds/leftwall.jpeg',
            "ceiling_wall_img": 'outputs/backgrounds/ceiling.jpg'
        }

        wallImages_bytes = {}
        for wall, image_path in wallImages.items():
            with open(image_path, 'rb') as f:
                wallImages_bytes[wall] = BytesIO(f.read())

        return wallImages_bytes
    except Exception as e:
        log_to_json(f"get_wall_images: Exceptional error {e}", current_file_name)
        return  None

def get_basic_wall_images():
    try:
        wallImages = {
            "floor_wall_img": 'outputs/backgrounds/basicbg/floorbg.png',
            "left_wall_img": 'outputs/backgrounds/basicbg/wallbg.png',
        }

        wallImages_bytes = {}
        for wall, image_path in wallImages.items():
            with open(image_path, 'rb') as f:
                wallImages_bytes[wall] = BytesIO(f.read())

        return wallImages_bytes
    except Exception as e:
        log_to_json(f"get_basic_wall_images: Exceptional error {e}", current_file_name)
        return  None


def get_wall_images_from_urls(image_urls):
   
    try:
        print(f"Got the image URLs in get_wall_images_from_urls: {image_urls}")
    # Initialize the dictionary to store images as BytesIO
        wallImages_bytes = {
            "floor_wall_img": None,
            "left_wall_img": None,
            "right_wall_img": None,
            "ceiling_wall_img": None
        }

        # Dynamically map wall keys to the image keys by matching patterns
        mapping = {}
        for key in image_urls:
            if "floor" in key:
                mapping["floor_wall_img"] = key
            elif "lw" in key:
                mapping["left_wall_img"] = key
            elif "rw" in key:
                mapping["right_wall_img"] = key
            elif "ceiling" in key:
                mapping["ceiling_wall_img"] = key

        print(f"Dynamic mapping of wall images: {mapping}")

        for wall_key, image_key in mapping.items():
            if image_key in image_urls:
                url = image_urls[image_key]
                try:
                    # Fetch the image from the URL
                    response = requests.get(url)
                    response.raise_for_status()  # Raise HTTPError for bad responses (4xx and 5xx)

                    # Load the image with Pillow
                    pil_image = Image.open(BytesIO(response.content))

                    # Convert the Pillow image to BytesIO
                    bytes_io = BytesIO()
                    pil_image.save(bytes_io, format=pil_image.format)
                    bytes_io.seek(0)  # Reset the pointer to the beginning of the BytesIO object

                    # Add the BytesIO object to the dictionary
                    wallImages_bytes[wall_key] = bytes_io
                    print(f"Successfully fetched and processed {wall_key} from {url}")
                except requests.exceptions.RequestException as e:
                    print(f"get_wall_images_from_urls: Failed to fetch image for {wall_key} from {url}: {e}")
                except Exception as e:
                    print(f"get_wall_images_from_urls: Failed to process image for {wall_key}: {e}")
            else:
                print(f"Image key {image_key} not found in image URLs")

        return wallImages_bytes
    except Exception as e:
        log_to_json(f"get_wall_images_from_urls:  Exceptional error {e}", current_file_name)
        return  None


def get_basic_wall_images_from_urls(image_urls):
    try:
        print(f"Got the image URLs in get_wall_images_from_urls: {image_urls}")
        
        # Initialize the dictionary to store images as BytesIO
        wallImages_bytes = {
            "floor_wall_img": None,
            "left_wall_img": None
        }

        # Dynamically map wall keys to the image keys by matching patterns
        mapping = {}
        for key in image_urls:
            if "floor" in key:
                mapping["floor_wall_img"] = key
            elif "lw" in key:
                mapping["left_wall_img"] = key

        print(f"Dynamic mapping of wall images: {mapping}")

        for wall_key, image_key in mapping.items():
            if image_key in image_urls:
                url = image_urls[image_key]
                try:
                    # Fetch the image from the URL
                    response = requests.get(url)
                    response.raise_for_status()  # Raise HTTPError for bad responses (4xx and 5xx)

                    # Load the image with Pillow
                    pil_image = Image.open(BytesIO(response.content))

                    # Convert the Pillow image to BytesIO
                    bytes_io = BytesIO()
                    pil_image.save(bytes_io, format=pil_image.format)
                    bytes_io.seek(0)  # Reset the pointer to the beginning of the BytesIO object

                    # Add the BytesIO object to the dictionary
                    wallImages_bytes[wall_key] = bytes_io
                    print(f"Successfully fetched and processed {wall_key} from {url}")
                except requests.exceptions.RequestException as e:
                    print(f"get_basic_wall_images_from_urls: Failed to fetch image for {wall_key} from {url}: {e}")
                    return None
                except Exception as e:
                    print(f"get_basic_wall_images_from_urls: Failed to process image for {wall_key}: {e}")
                    return None
            else:
                print(f"Image key {image_key} not found in image URLs")

        return wallImages_bytes
    except Exception as e:
        log_to_json(f"Exceptional error for: get_basic_wall_images_from_urls:  {e}", current_file_name)
        return  None

def get_logo_image():
    try:
        logo_path = 'outputs/backgrounds/app_logo.png'
        with open(logo_path, "rb") as image_file:
            logo_bytesio = BytesIO(image_file.read())
        
        return logo_bytesio 
    except Exception as e:
        log_to_json(f"Exceptional error for:get_logo_image:  {e}", current_file_name)
        return  None

def get_dynamic_image(logo_bytesio):
   
    try:

        logo_bytesio.seek(0)  # Reset the stream position

        new_dir = "bottom_left"


        result_bytesio = apply_drop_shadow(logo_bytesio) # apply_3d_effect(logo_bytesio, direction="top_left"))  


        return result_bytesio
    except requests.exceptions.RequestException as e:
        log_to_json(f"Error on: get_dynamic_image for:  fetching the logo image: {e}")
        return None
    except Exception as e:
        log_to_json(f"Error on: get_dynamic_image for: {e}")
        return None
    
def apply_3d_effect(image_bytesio, direction="bottom_right", depth=4):
    # Open the image from BytesIO
    try:
        img = Image.open(image_bytesio).convert("RGBA")

        # Separate the alpha channel and the main image
        alpha = img.split()[3]
        img_no_alpha = img.convert("RGB")

        # Create a canvas to hold the 3D effect
        depth_image = Image.new("RGBA", img.size, (0, 0, 0, 0))

        # Repeat the image multiple times with slight offsets to simulate depth
        num_repeats = depth  # Adjust for depth
        offset_step = 1  # Step size for each repeat

        # Determine offset direction
        direction_offsets = {
            "top_left": (-offset_step, -offset_step),
            "top_right": (offset_step, -offset_step),
            "bottom_left": (-offset_step, offset_step),
            "bottom_right": (offset_step, offset_step)
        }
        offset_direction = direction_offsets.get(direction, (offset_step, offset_step))

        for i in range(num_repeats):
            offset = (i * offset_direction[0], i * offset_direction[1])
            repeated_layer = img_no_alpha.copy()
            depth_image.paste(repeated_layer, offset, alpha)

        # Add a slight shadow always in the bottom direction
        shadow_image = Image.new("RGBA", img.size, (0, 0, 0, 0))
        shadow_color = (0, 0, 0, 50)  # Semi-transparent black
        shadow_offset_step = 2  # Slightly larger step for shadow
        for i in range(num_repeats // 2):
            shadow_offset = (0, i * shadow_offset_step)  # Shadow moves downward
            shadow_layer = Image.new("RGBA", img.size, shadow_color)
            shadow_image.paste(shadow_layer, shadow_offset, alpha)

        depth_image = Image.alpha_composite(shadow_image, depth_image)

        # Enhance brightness and contrast to make the depth more pronounced
        depth_image_no_alpha = depth_image.convert("RGB")
        enhancer = ImageEnhance.Brightness(depth_image_no_alpha)
        depth_image_no_alpha = enhancer.enhance(1.2)

        enhancer = ImageEnhance.Contrast(depth_image_no_alpha)
        depth_image_no_alpha = enhancer.enhance(1.3)

        # Combine the depth image and the original image
        depth_image = Image.composite(depth_image_no_alpha.convert("RGBA"), depth_image, alpha)
        result = Image.alpha_composite(depth_image, img)

        # Save the result to BytesIO
        result_bytesio = BytesIO()
        result.save(result_bytesio, format="PNG")
        result_bytesio.seek(0)

        return result_bytesio
    except Exception as e:
        log_to_json(f"Error on: apply_3d_effect for: {e}")
        return None

def dynamic_addBackground():
    try:

        logo_path = 'outputs/backgrounds/app_logo.png'
        output_path = 'outputs/dumtest/outputdebug/finalimages'
        wallImages = {
            "floor_wall_img": 'outputs/backgrounds/floor.png',
            "left_wall_img": 'outputs/backgrounds/leftwall.png',
            "right_wall_img": 'outputs/backgrounds/rightwall.png',
            "ceiling_wall_img": 'outputs/backgrounds/ceiling.png'
        }

        wallImages_bytes = {}
        for wall, image_path in wallImages.items():
            with open(image_path, 'rb') as f:
                wallImages_bytes[wall] = BytesIO(f.read())


        os.makedirs(output_path, exist_ok=True)  # Ensure the output directory exists

        # Convert the images to BytesIO format
        with open(logo_path, "rb") as image_file:
            logo_bytesio = BytesIO(image_file.read())


        # Call the addReflection function
        floor_coordinates, left_wall_coordinates, right_wall_coordinates, ceiling_coordinates = get_wall_coordinates()
        background = addBackgroundInitiate(wallImages_bytes, logo_bytesio, floor_coordinates, left_wall_coordinates, right_wall_coordinates, ceiling_coordinates)

        # Save the image in output path
        if background:
            output_image_path = os.path.join(output_path, "generated_background.png")
            with open(output_image_path, "wb") as output_file:
                output_file.write(background.getvalue())
            print(f"Background image saved at: {output_image_path}")
        else:
            print("Failed to process the image.")
    except Exception as e:
        log_to_json(f"Exceptional error on the function of dynamic_addBackground {e}", current_file_name)
        return  None
    



def apply_drop_shadow(input_bytes_io: io.BytesIO) -> io.BytesIO:

    # rewind and open
    input_bytes_io.seek(0)
    img = Image.open(input_bytes_io).convert("RGBA")

    # parameters
    angle, distance = 176, 3
    blur_radius, spread_pct, opacity = 3, 0.90, 0.70

    # extract & optionally spread the alpha mask
    alpha = img.split()[3]
    spread_radius = int(round(blur_radius * spread_pct))
    if spread_radius > 0:
        # dilate the mask by a tiny amount
        alpha = alpha.filter(ImageFilter.MaxFilter(spread_radius*2+1))

    # make a solid-black image, masked by the (spread) alpha
    shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    shadow.putalpha(alpha)

    # blur to create the feather (size)
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # apply the 50% opacity to the blurred alpha channel
    r, g, b, a = shadow.split()
    a = a.point(lambda p: p * opacity)
    shadow = Image.merge("RGBA", (r, g, b, a))

    # compute offset (shadow is cast opposite to the light angle)
    rad = math.radians(angle + 180)
    dx = int(round(distance * math.cos(rad)))
    dy = int(round(distance * math.sin(rad)))

    # build a canvas large enough for both shadow and original
    w, h = img.size
    new_w = w + abs(dx)
    new_h = h + abs(dy)
    canvas = Image.new("RGBA", (new_w, new_h), (0, 0, 0, 0))

    # paste shadow, then original on top
    canvas.paste(shadow, (max(dx, 0), max(dy, 0)), shadow)
    canvas.paste(img,    (max(-dx, 0), max(-dy, 0)), img)

    # export to BytesIO
    out = io.BytesIO()
    canvas.save(out, format="PNG")
    out.seek(0)
    return out





def backwall_calculate(x):

    # For negative x, mirror the function by computing for |x| and reversing the sign.
    if x < 0:
        return -backwall_calculate(-x)
    
    # Now x is nonnegative. We allow x up to 90.
    if x > 90:
        return 0
    
    # Region 1: For 0 <= x < 22: interpolate between (0, 85) and (22, 22/3)
    if x < 22:
        # slope = ((22/3) - 85) / 22  # Compute the slope
        slope = ((22/3) - 20) / 40
        # y = 85 + slope * x
        y = 20 + slope * x 
        # y = 17
        return y
    
    # Region 2: For 22 <= x <= 30: use the rule y = x/3
    range_to = 30
    if 22 <= x <= range_to:
        return x / 3
    
    # Region 3: For 30 < x <= 90: interpolate between (30, 10) and (90, 0)
    if x > range_to:
        slope = (0 - 10) / (90 - range_to)  # The slope over (30, 90)
        y = 10 + slope * (x - range_to)
        return y


