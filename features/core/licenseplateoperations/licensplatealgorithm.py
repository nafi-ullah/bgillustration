import cv2
import numpy as np
import os
from ultralytics import YOLO
from PIL import Image
from pathlib import Path
from io import BytesIO
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression
current_file_name = "features/core/licenseplateoperations/licenseplateimageupdated.py"
licensepalate_model_path = './models/augmented/augmented.pt'
input_image_path = transparent_image_path = './data/clientpics/KMHB35117SW024635/photo_1.jpg'
license_plate_image_path = './fill/Artboard 51-100.jpg'
import math
from itertools import combinations
import matplotlib.pyplot as plt
import itertools
from my_logging_script import log_to_json
import re
import numpy as np
from sklearn.linear_model import RANSACRegressor, LinearRegression

def append_to_log(new_entry):
    try:
        # log_to_json(new_entry, current_file_name)
        printing = True
    except Exception as e:
        print(f"Error in append_to_log: {e}")
        # Fallback to print if logging fails
        print(new_entry)



def euclidean_distance(p1, p2):
    try:
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    except Exception as e:
        append_to_log(f"euclidean_distance: Error calculating distance between {p1} and {p2}: {e}")
        return None

def average_coordinates(points, max_distance=8):
    result = []
    points = points.copy()  # To avoid mutating the input list

    try:
        while points:
            base_point = points[0]
            group = [base_point]

            # Find points within distance 8 from the base_point
            for pt in points[1:]:
                if euclidean_distance(base_point, pt) <= max_distance:
                    group.append(pt)

            # If group has 2 or more points, average them and remove from main list
            if len(group) >= 2:
                avg_x = sum(p[0] for p in group) / len(group)
                avg_y = sum(p[1] for p in group) / len(group)
                result.append((avg_x, avg_y))

                # Remove grouped points from the list
                points = [p for p in points if p not in group]
            else:
                # If not enough neighbors, remove just the base point
                points.pop(0)
    except Exception as e:
        append_to_log(f"average_coordinates: Error in average_coordinates: {e}")

    return result

def average_slope(points):
    try:
        slopes = []
        positive_slope_count = 0
        negative_slope_count = 0
        slop_point_json_array = []  # [{point1: (x1, y1), point2: (x2, y2), slope: m}, ...]
        n = len(points)
        # Use all unique pairs (i < j)
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                if x2 != x1:
                    m = (y2 - y1) / (x2 - x1)
                    slopes.append(m)
                    if m >= 0:
                        positive_slope_count += 1
                    else :
                        negative_slope_count += 1
                    slop_point_json_array.append({"point1": f"{x1},{y1}", "point2": f"{x2},{y2}", "slope": m})
                # else: vertical line (slope is undefined), skip

        if not slopes:
            return 0  # No valid slopes to average

        append_to_log(f"average_slope: slopes: {slopes} slop_point_json_array: {slop_point_json_array}")
        if positive_slope_count > negative_slope_count:
            avg_slope = abs(sum(slopes) / len(slopes))
        else:
            avg_slope = sum(slopes) / len(slopes)
        return avg_slope
    except Exception as e:
        append_to_log(f"average_slope: Error in average_coordinates: {e}")


def get_intersection_point(m1, c1, m2, c2):
    try:
        # Case 1: Both lines are vertical (x = c1 and x = c2) — no intersection unless c1 == c2
        if m1 == 'inf' and m2 == 'inf':
            return None if c1 != c2 else (c1, None)

        # Case 2: First line is vertical (x = c1)
        if m1 == 0:
            x = c1
            y = m2 * x + c2
            return (x, y)

        # Case 3: Second line is vertical (x = c2)
        if m2 == 0:
            x = c2
            y = m1 * x + c1
            return (x, y)

        # Case 4: Both lines have same slope — parallel lines
        if m1 == m2:
            return None

        # Normal intersection calculation
        x = (c2 - c1) / (m1 - m2)
        y = m1 * x + c1
        return (x, y)
    except Exception as e:
        append_to_log(f"get_intersection_point: Error in get_intersection_point: {e}")
        return None


def get_10_splitted_values_array(x1, x2):
    try:
        x1 = float(x1)
        x2 = float(x2)
        if x1 == x2:
            return [x1] * 10  # If both values are the same, return a list of that value

        step = (x2 - x1) / 9
        return [x1 + i * step for i in range(10)]
    except Exception as e:
        append_to_log(f"get_10_splitted_values_array: Error in get_10_splitted_values_array: {e}")
        return []

def find_segment_points(contour, pt_start, pt_end):

    try:
        start_idx = end_idx = -1
        for i, pt in enumerate(contour):
            if tuple(pt[0]) == pt_start:
                start_idx = i
            if tuple(pt[0]) == pt_end:
                end_idx = i
        if start_idx == -1 or end_idx == -1:
            return []

        # Extract the full segment first
        if start_idx <= end_idx:
            segment = contour[start_idx:end_idx + 1]
        else:
            segment = np.concatenate((contour[start_idx:], contour[:end_idx + 1]), axis=0)

        # Trim 10% from start and end
        total_points = len(segment)
        trim = int(total_points * 0.15)
        trimmed_segment = segment[trim:total_points - trim]

        return [tuple(pt[0]) for pt in trimmed_segment]
    except Exception as e:
        append_to_log(f"find_segment_points: Error in find_segment_points: {e}")
        return None
    

def get_right_left_most_coordinates(datas):
    try:
        if not datas:
            return None, None

        # Convert to numpy array for easier manipulation
        data = np.array(datas)

        # Find the rightmost and leftmost points
        right_most = data[np.argmax(data[:, 0])]
        left_most = data[np.argmin(data[:, 0])]

        return {
            "right_most": tuple(right_most),
            "left_most": tuple(left_most)
        }
    except Exception as e:
        append_to_log(f"get_right_left_most_coordinates: Error in get_right_left_most_coordinates: {e}")
        return None



def get_intercept(m, point):
    # y = mx + c => c = y - mx
    try:
        x, y = point
        return y - m * x
    except Exception as e:
        append_to_log(f"get_intercept: Error in get_intercept: {e}")
        return None

def get_top_bottom_points(datas, m,c ):
    try:
        def line_eq(x,m,c):
            return m * x + c
        

        data = np.array(datas)

        x_vals = data[:, 0]
        y_vals = data[:, 1]
        line_y_vals = line_eq(x_vals,m,c)

        # Step 2: Assign clusters based on position relative to line
        # 0 = top cluster (above line), 1 = bottom cluster (below or on line)
        cluster_labels = np.where(y_vals > line_y_vals, 0, 1)

        # Step 3: Store top and bottom cluster points
        bottom = data[cluster_labels == 0].tolist()
        top = data[cluster_labels == 1].tolist()
        # append_to_log(f"data: {[(x, y) for x, y in datas]}")
        # append_to_log(f"top : {top}")
        # append_to_log(f"angle : {m} intercept: {c}")


        return top, bottom
    except Exception as e:
        append_to_log(f"get_top_bottom_points: Error in get_top_bottom_points: {e}")
        return None

def get_right_bottom_points(datas, m, c):
    try:
        def inverse_line_eq(y, m, c):
            return (y - c) / m if m != 0 else float('inf')  # avoid division by zero

        data = np.array(datas)

        x_vals = data[:, 0]
        y_vals = data[:, 1]
        line_x_vals = inverse_line_eq(y_vals, m, c)

        # 0 = right cluster (right of the line), 1 = left cluster (left or on the line)
        cluster_labels = np.where(x_vals > line_x_vals, 0, 1)

        right = data[cluster_labels == 0].tolist()
        left = data[cluster_labels == 1].tolist()

        # append_to_log(f"data: {[(x, y) for x, y in datas]}")
        return left, right
    except Exception as e:
        append_to_log(f"get_right_bottom_points: Error in get_right_bottom_points: {e}")
        return None



def trimming_data(data, direction, volume):
    try:
        if not data or not (0 < volume <= 1):
            return []

        # Sort based on direction
        if direction == "right":
            sorted_data = sorted(data, key=lambda point: point[0])  # ascending x
        elif direction == "left":
            sorted_data = sorted(data, key=lambda point: point[0], reverse=True)  # descending x
        else:
            raise ValueError("direction must be 'right' or 'left'")

        # Keep the first volume% of the data
        total_points = len(sorted_data)
        keep = int(total_points * volume)

        if keep == 0:
            return []

        trimmed_segment = sorted_data[:keep]
        return trimmed_segment
    except Exception as e:
        append_to_log(f"trimming_data: Error in trimming_data: {e}")
        return None




def point_one_third_along(p1, p2):
    try:
        x1, y1 = p1
        x2, y2 = p2
        x_1_3 = x1 + (x2 - x1) * (1/3)
        y_1_3 = y1 + (y2 - y1) * (1/3)
        return (x_1_3, y_1_3)
    except Exception as e:
        append_to_log(f"point_one_third_along: Error in point_one_third_along: {e}")
        return None


# if any portion of license plate is not covered continously then use following function

def adjust_left_right_most_line(top_coord, bottom_coord, line_m, line_c, top_cluster, bottom_cluster, top_line_m, bottom_line_m, top_line_c, bottom_line_c):
    try:
        # losest x value of top_cluster tuple array is the topcluster_left_most 
        top_left_most = min(top_cluster, key=lambda x: x[0])
        bottom_left_most = min(bottom_cluster, key=lambda x: x[0])
        is_top_blank = top_left_most[0] < top_coord[0]
        is_bottom_blank = bottom_left_most[0] < bottom_coord[0]

        if line_m != 0:
            if is_top_blank and not is_bottom_blank:
                # Adjust the top line to the leftmost point of the top cluster
                left_line_c = top_left_most[1] - (line_m * top_left_most[0])
            elif not is_top_blank and is_bottom_blank:
                # Adjust the bottom line to the leftmost point of the bottom cluster
                left_line_c = bottom_left_most[1] - (line_m * bottom_left_most[0])
            elif is_top_blank and is_bottom_blank:
                leftest_point = top_left_most if top_left_most[0] < bottom_left_most[0] else bottom_left_most
                left_line_c = leftest_point[1] - (line_m * leftest_point[0])
            else:
                # No adjustment needed, use the original line_c
                left_line_c = line_c

            top_left_coord = (top_coord[0], line_m * top_coord[0] + left_line_c)
            bottom_left_coord = (bottom_coord[0], line_m * bottom_coord[0] + left_line_c)
        else:
            leftest_point = top_left_most if top_left_most[0] < bottom_left_most[0] else bottom_left_most
            top_left_coord = (leftest_point[0], top_line_m * leftest_point[0] + top_line_c)
            bottom_left_coord = (leftest_point[0], bottom_line_m * leftest_point[0] + bottom_line_c)

        return top_left_coord, bottom_left_coord
    except Exception as e:
        append_to_log(f"adjust_left_right_most_line: Error in adjust_left_right_most_line: {e}")
        return top_coord, bottom_coord


def licensplate_coordinates(model_path, input_image ,transparent_image=None, license_plate=None):
    try:
    # Load the input image
        input_image.seek(0)
        pil_image = Image.open(input_image)
        img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        if img is None:
            log_to_json(f"addLicensePlateImage: Unable to read withbg image.", current_file_name)
            return None
        
        transparent_image.seek(0)
        pil_image_transparent = Image.open(transparent_image).convert("RGBA")
        img_transparent = np.array(pil_image_transparent)

        if img_transparent is None:
            log_to_json(f"addLicensePlateImage: Unable to read transparent image.", current_file_name)
            return None
        # if img_transparent.shape[-1] != 4:
        #     log_to_json(f"Transparent image does not have an alpha channel.", current_file_name)
        #     return None


        original_height, original_width = img.shape[:2]

        # Load the YOLO model
        model = YOLO(model_path)

        # Run inference on the image
        results = model(img)

        if not results:
            log_to_json("No licenseplate found in car by the YOLO model.", current_file_name)
            return None

        # Combine all masks into a single binary mask
        combined_mask = np.zeros((original_height, original_width), dtype=np.uint8)
        for result in results:
            if result.masks is None:
                log_to_json("No masks found for the detected object.", current_file_name)
                continue
            for mask in result.masks.data:
                # Move the mask to CPU if needed, scale to 0-255, and convert to uint8
                mask = (mask.cpu().numpy() * 255).astype('uint8')

                # Resize the mask to match the dimensions of the original image
                mask_resized = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

                # Combine the resized mask with the combined_mask using bitwise OR
                combined_mask = cv2.bitwise_or(combined_mask, mask_resized)

        # Find contours in the combined mask
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            append_to_log("No segmented areas found.")
            return None

        # Find the largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)
        contour_coordinates = [(point[0][0], point[0][1]) for point in largest_contour]

        # Find extreme points of the largest contour
        top_left = tuple(largest_contour[np.argmin(largest_contour[:, 0, 0] + largest_contour[:, 0, 1])][0])
        top_right = tuple(largest_contour[np.argmax(largest_contour[:, 0, 0] - largest_contour[:, 0, 1])][0])
        bottom_left = tuple(largest_contour[np.argmin(largest_contour[:, 0, 0] - largest_contour[:, 0, 1])][0])
        bottom_right = tuple(largest_contour[np.argmax(largest_contour[:, 0, 0] + largest_contour[:, 0, 1])][0])
        
        left_mid = ((top_left[0] + bottom_left[0]) // 2, (top_left[1] + bottom_left[1]) // 2)
        right_mid = ((top_right[0] + bottom_right[0]) // 2, (top_right[1] + bottom_right[1]) // 2)
        center_mid = ((left_mid[0] + right_mid[0]) // 2, (left_mid[1] + right_mid[1]) // 2)

        mid_line_angle = math.degrees(math.atan2(right_mid[1] - left_mid[1], right_mid[0] - left_mid[0]))
        angle_rad, mid_slope = get_line_properties(left_mid, right_mid)["angle_rad"], get_line_properties(left_mid, right_mid)["slope"]
        # Prevent division by zero for angle_rad
        if angle_rad == 0:
            center_red = 0
        else:
            center_red = -(1/angle_rad)
        mid_line_intercept = left_mid[1] - (mid_slope * left_mid[0])
        left_one_third = point_one_third_along(bottom_left, top_left)
        # right_one_third and right_one_third_t are not used, so we remove them to fix the "not accessed" error
        # right_one_third = point_one_third_along(bottom_right, top_right)
        one_third_line_intercept = left_one_third[1] - (angle_rad * left_one_third[0])
        center_line_intercept = center_mid[1] - (center_red * center_mid[0])

        # for top coordinates
        left_one_third_t = point_one_third_along(top_left, bottom_left)
        # right_one_third_t = point_one_third_along(top_right, bottom_right)
        one_third_line_intercept_t = left_one_third_t[1] - (angle_rad * left_one_third_t[0])


        log_to_json(f"licencese plate points are tl, tr, bl, br: {top_left} {top_right} {bottom_left} {bottom_right} ", current_file_name)
        # append_to_log(f"left_mid: {left_mid} right_mid: {right_mid}")
        append_to_log(f"mid slope {mid_slope} angle_rad : {angle_rad} mid_line_intercept: {mid_line_intercept} center {center_mid}")
        append_to_log(f"center_red: {center_red} center_line_intercept: {center_line_intercept}")

        top , bottom = get_top_bottom_points(contour_coordinates, angle_rad, one_third_line_intercept)
        top_t , bottom_t = get_top_bottom_points(contour_coordinates, angle_rad, one_third_line_intercept_t)

  

        slope = np.tan(angle_rad)
        perp_slope = -1 / slope if slope != 0 else 1e6
        perp_intercept = center_mid[1] - perp_slope * center_mid[0]
        append_to_log(f"perpendicular slope: {perp_slope} intercept: {perp_intercept}")
        append_to_log(f"angle_rad: {angle_rad} intercept: {one_third_line_intercept} equation y = {angle_rad}x + {one_third_line_intercept}")

        left_bottom_quarter , right_bottom_quarter = get_right_bottom_points(bottom, perp_slope, perp_intercept )

        left_trim = trimming_data(left_bottom_quarter, "left", 0.8)
        right_trim = trimming_data(right_bottom_quarter, "right", 0.8)
        
        # original_points = [(285, 626),(448, 689)  ,(285, 691) ,(442, 773)]

        # top trimming
        left_top_quarter , right_top_quarter = get_right_bottom_points(top_t, perp_slope, perp_intercept )
        left_trim_t = trimming_data(left_top_quarter, "left", 0.8)
        right_trim_t = trimming_data(right_top_quarter, "right", 0.8)

        reg_top_m , reg_top_c = get_regression_line(left_trim_t + right_trim_t + [top_left, top_right])
        reg_bottom_m , reg_bottom_c = get_regression_line(left_trim + right_trim + [bottom_left, bottom_right])
        append_to_log(f"addLicenseplateImage: reg_top_m: {reg_top_m} reg_top_c: {reg_top_c}")
        append_to_log(f"addLicenseplateImage: reg_bottom_m: {reg_bottom_m} reg_bottom_c: {reg_bottom_c}")

        top_left_final, top_right_final, bottom_left_final, bottom_right_final = get_left_right_line(
            datas=contour_coordinates,
            corners={
                "top_left": top_left,
                "top_right": top_right,
                "bottom_left": bottom_left,
                "bottom_right": bottom_right
            },
            lines={
                "top": (reg_top_m, reg_top_c),
                "bottom": (reg_bottom_m, reg_bottom_c)
            },
            filename="new_file.png"
        )
        if top_left_final is None or top_right_final is None or bottom_left_final is None or bottom_right_final is None:
            log_to_json("Error in get_left_right_line, previous calculated coordinates", current_file_name)
            top_left_final , top_right_final, bottom_left_final, bottom_right_final = top_left, top_right, bottom_left, bottom_right



        log_to_json(f"licencese plate adjusted points are tl, tr, bl, br: {top_left_final} {top_right_final} {bottom_left} {bottom_left_final} ", current_file_name)


        return top_left_final, top_right_final, bottom_left_final, bottom_right_final

    except FileNotFoundError as e:
        log_to_json(f"File not found error: {e}", current_file_name)
        return None, None, None, None
    except ValueError as e:
        log_to_json(f"ValueError: {e}", current_file_name)
        return None, None, None, None
    except Exception as e:
        log_to_json(f"Exceptional error: {e}", current_file_name)
        return None, None, None, None
    
def get_angle_difference_from_slope(slope1, slope2):

    try:
        # Calculate the angle in radians for each slope
        angle1 = math.atan(slope1)
        angle2 = math.atan(slope2)

        # Calculate the absolute difference in angles
        angle_difference = abs(angle1 - angle2)

        # Convert the angle difference to degrees
        angle_difference_degrees = math.degrees(angle_difference)

        return angle_difference_degrees
    except Exception as e:
        append_to_log(f"get_angle_difference_from_slope: Error in get_angle_difference_from_slope: {e}")
        return None
    
def get_left_right_line(datas, corners, lines, filename=None):
    # Extract the coordinates of the corners
    try:
        top_left, top_right, bottom_left, bottom_right = corners["top_left"], corners["top_right"], corners["bottom_left"], corners["bottom_right"]

        top_m , top_c = lines["top"]
        bottom_m , bottom_c = lines["bottom"]

        top_left_derived = (top_left[0], top_m * top_left[0] + top_c)
        top_right_derived = (top_right[0], top_m * top_right[0] + top_c)
        bottom_left_derived = (bottom_left[0], bottom_m * bottom_left[0] + bottom_c)
        bottom_right_derived = (bottom_right[0], bottom_m * bottom_right[0] + bottom_c)

        distance_percantage = 1
        distance_left = distance_percantage * math.sqrt((top_left_derived[0] - bottom_left_derived[0]) ** 2 + (top_left_derived[1] - bottom_left_derived[1]) ** 2)
        distance_right = distance_percantage * math.sqrt((top_right_derived[0] - bottom_right_derived[0]) ** 2 + (top_right_derived[1] - bottom_right_derived[1]) ** 2)

        middle_left = ((top_left_derived[0] + bottom_left_derived[0]) / 2, (top_left_derived[1] + bottom_left_derived[1]) / 2)
        middle_right = ((top_right_derived[0] + bottom_right_derived[0]) / 2, (top_right_derived[1] + bottom_right_derived[1]) / 2)

        # find coordinates from datas which distance from middle_left is less than or equal to distance_left
        left = []
        coordinates_array = np.array(datas)
        for i in range(len(coordinates_array)):
            distance = math.sqrt((coordinates_array[i][0] - middle_left[0]) ** 2 + (coordinates_array[i][1] - middle_left[1]) ** 2)
            if distance <= int(distance_left/2):
                left.append((coordinates_array[i][0], coordinates_array[i][1]))
            # if distance <= int((distance_left*0.5)/2):
            #     left.append((coordinates_array[i][0], coordinates_array[i][1]))

        right = []
        for i in range(len(coordinates_array)):
            distance = math.sqrt((coordinates_array[i][0] - middle_right[0]) ** 2 + (coordinates_array[i][1] - middle_right[1]) ** 2)
            if distance <= int(distance_right/2):
                right.append((coordinates_array[i][0], coordinates_array[i][1]))
            # if distance <= int((distance_right*0.5)/2):
            #     right.append((coordinates_array[i][0], coordinates_array[i][1]))

        # points close to top line and bottom line among coordinates_array using perpendicular distance
        threshold_distance = distance_left * 0.1 if distance_left < distance_right else distance_right * 0.1
        top_close_coordinates = []
        bottom_close_coordinates = []
        for i in range(len(coordinates_array)):
            x, y = coordinates_array[i][0], coordinates_array[i][1]
            # Perpendicular distance from point (x, y) to line y = m*x + c is |m*x - y + c| / sqrt(m^2 + 1)
            distance_top = abs(top_m * x - y + top_c) / math.sqrt(top_m ** 2 + 1)
            distance_bottom = abs(bottom_m * x - y + bottom_c) / math.sqrt(bottom_m ** 2 + 1)
            if distance_top <= threshold_distance:
                top_close_coordinates.append((x, y))
            if distance_bottom <= threshold_distance:
                bottom_close_coordinates.append((x, y))

        # sort top_close_coordinates and bottom_close_coordinates according to the x value of each coordinates
        top_close_coordinates = sorted(top_close_coordinates, key=lambda coord: coord[0])
        bottom_close_coordinates = sorted(bottom_close_coordinates, key=lambda coord: coord[0])

        # new top_left and top_right coordinates are first and last coordinates of top_close_coordinates
        # new bottom_left and bottom_right coordinates are first and last coordinates of bottom_close_coordinates
        new_top_left = top_close_coordinates[0] if len(top_close_coordinates) > 0 and top_close_coordinates[0][0] <= top_left[0] else top_left
        new_top_right = top_close_coordinates[-1] if len(top_close_coordinates) > 0 and top_close_coordinates[-1][0] >= top_right[0] else top_right
        new_bottom_left = bottom_close_coordinates[0] if len(bottom_close_coordinates) > 0 and bottom_close_coordinates[0][0] <= bottom_left[0] else bottom_left
        new_bottom_right = bottom_close_coordinates[-1] if len(bottom_close_coordinates) > 0 and bottom_close_coordinates[-1][0] >= bottom_right[0] else bottom_right

        append_to_log(f"new_top_left: {new_top_left} new_top_right: {new_top_right} new_bottom_left: {new_bottom_left} new_bottom_right: {new_bottom_right}")
        new_top_left_derived = (new_top_left[0], top_m * new_top_left[0] + top_c)
        new_top_right_derived = (new_top_right[0], top_m * new_top_right[0] + top_c)
        new_bottom_left_derived = (new_bottom_left[0], bottom_m * new_bottom_left[0] + bottom_c)
        new_bottom_right_derived = (new_bottom_right[0], bottom_m * new_bottom_right[0] + bottom_c)
        append_to_log(f"new_top_left_derived: {new_top_left_derived} new_top_right_derived: {new_top_right_derived} new_bottom_left_derived: {new_bottom_left_derived} new_bottom_right_derived: {new_bottom_right_derived}")

        # remove the points which are exist in bottom_close_coordinates , top_close_coordinates from left and right array
        left = [coord for coord in left if coord not in bottom_close_coordinates and coord not in top_close_coordinates]
        right = [coord for coord in right if coord not in bottom_close_coordinates and coord not in top_close_coordinates]

        # average points
        left_avg = average_coordinates(left)
        # left_avg = average_coordinates(left_avg, max_distance=8) if len(left_avg) > 1 else left_avg
        right_avg = average_coordinates(right)
        # right_avg = average_coordinates(right_avg, max_distance=8) if len(right_avg) > 1 else right_avg

        append_to_log(f"left_avg: {left_avg} right_avg: {right_avg}")

        if len(right_avg) == 1:
            all_points_avg_m = average_slope(right_avg + [new_top_right_derived, new_bottom_right_derived])
            corners_slope = average_slope([new_top_right_derived, new_bottom_right_derived])
            right_line_m = all_points_avg_m if all_points_avg_m > corners_slope else corners_slope
        elif len(right_avg) > 1:
            right_line_m = average_slope(right_avg)
        else:
            right_line_m = average_slope([new_top_right_derived, new_bottom_right_derived])

        if len(left_avg) == 1:
            all_points_avg_m = average_slope(left_avg + [new_top_left_derived, new_bottom_left_derived])
            corners_slope = average_slope([new_top_left_derived, new_bottom_left_derived])
            append_to_log(f"comparing points all_points_avg_m: {all_points_avg_m} corners_slope: {corners_slope}")
            left_line_m = all_points_avg_m if all_points_avg_m < corners_slope else corners_slope
        elif len(left_avg) > 1:
            left_line_m = average_slope(left_avg)
        else:
            left_line_m = average_slope([new_top_left_derived, new_bottom_left_derived])

        append_to_log(f"left_line_m: {left_line_m} right_line_m: {right_line_m}")

        left_most_point = get_right_left_most_coordinates(left + [new_top_left_derived, new_bottom_left_derived])
        right_most_point = get_right_left_most_coordinates(right + [new_top_right_derived, new_bottom_right_derived])

        left_line_c = get_intercept(left_line_m, left_most_point["left_most"])
        left_line_parallel_c = get_intercept(right_line_m, left_most_point["left_most"])
        right_line_c = get_intercept(right_line_m, right_most_point["right_most"])

        left_len = len(left)
        right_len = len(right)
        left_reg = [] if left_len < 9 else left
        right_reg = [] if right_len < 9 else right
        left_coordinates =  [ new_top_left_derived, new_bottom_left_derived] + left_reg if left_len <=9 else left
        right_coordinates =   [ new_top_right_derived, new_bottom_right_derived] + right_reg if right_len <=9 else right


        # append_to_log(f"left coordinates: {left_coordinates}")

        # left_line_m , left_line_c = get_regression_line(left_coordinates) if len(left_coordinates) <= 6 else get_ransac_line(left_coordinates)  
        # right_line_m , right_line_c = get_regression_line(right_coordinates) if len(right_coordinates) <= 6 else get_ransac_line(right_coordinates)

        append_to_log(f"left_line_m: {left_line_m} left_line_c: {left_line_c}")
        append_to_log(f"right_line_m: {right_line_m} right_line_c: {right_line_c}")
        

        top_left_final = get_intersection_point(top_m, top_c, left_line_m, left_line_c)
        append_to_log(f"top_left_final: {top_left_final} from top_m: {top_m} top_c: {top_c} left_line_m: {left_line_m} left_line_c: {left_line_c}")
        top_right_final = get_intersection_point(top_m, top_c, right_line_m, right_line_c)
        append_to_log(f"top_right_final: {top_right_final} from top_m: {top_m} top_c: {top_c} right_line_m: {right_line_m} right_line_c: {right_line_c}")
        bottom_left_final = get_intersection_point(bottom_m, bottom_c, left_line_m, left_line_c)
        append_to_log(f"bottom_left_final: {bottom_left_final} from bottom_m: {bottom_m} bottom_c: {bottom_c} left_line_m: {left_line_m} left_line_c: {left_line_c}")
        bottom_right_final = get_intersection_point(bottom_m, bottom_c, right_line_m, right_line_c)
        append_to_log(f"bottom_right_final: {bottom_right_final} from bottom_m: {bottom_m} bottom_c: {bottom_c} right_line_m: {right_line_m} right_line_c: {right_line_c}")

        append_to_log(f"top_left_final: {top_left_final} top_right_final: {top_right_final} bottom_left_final: {bottom_left_final} bottom_right_final: {bottom_right_final}")



        # Top-Right
        if right_line_m == 0:
            x = right_most_point["right_most"][0]
            y = top_m * x + top_c
            top_right_dum = np.array([x, y])
        else:
            top_right_dum = intersection(top_m, top_c, right_line_m, right_line_c)

        # Bottom-Right
        if right_line_m == 0:
            x = right_most_point["right_most"][0]
            y = bottom_m * x + bottom_c
            bottom_right_dum = np.array([x, y])
        else:
            bottom_right_dum = intersection(bottom_m, bottom_c, right_line_m, right_line_c)

        # Top-Left
        if right_line_m == 0:  # vertical line
            x = left_most_point["left_most"][0]
            y = top_m * x + top_c
            top_left_dum = np.array([x, y])
        else:
            top_left_dum = intersection(top_m, top_c, right_line_m, left_line_parallel_c) # angle should be same as right line

        # Bottom-Left
        if right_line_m == 0:
            x = left_most_point["left_most"][0]
            y = bottom_m * x + bottom_c
            bottom_left_dum = np.array([x, y])
        else:
            bottom_left_dum = intersection(bottom_m, bottom_c, right_line_m, left_line_parallel_c) # angle should be same as right line


        #----------- adjusting top_left_dum, top_right_dum according to the top and bottom slope ---------
        top_bottom_angle_difference = get_angle_difference_from_slope(top_m, bottom_m)
        if abs(top_bottom_angle_difference) > 2:
                    # find left length and right length
            left_length = math.sqrt((top_left_dum[0] - bottom_left_dum[0]) ** 2 + (top_left_dum[1] - bottom_left_dum[1]) ** 2)
            right_length = math.sqrt((top_right_dum[0] - bottom_right_dum[0]) ** 2 + (top_right_dum[1] - bottom_right_dum[1]) ** 2)


            approximate_angle_rise = top_bottom_angle_difference * 0.75 if top_bottom_angle_difference < 11 else top_bottom_angle_difference - 3
            # kototku diffeerence chai = 30%, that means 70% angle upore uthatie hobe
            # mane 30 r 40 angle hoile, 37 banaite hobe
            # ei j 7 uthabo.. etar 50% daan dike pointer jonno uthbe, baam diker ta 50% er jnno nambe.. agy uthaite hobe
            top_anlge = math.degrees(math.atan(top_m))
            
            first_point_rise = approximate_angle_rise * 0.6
            opposite_angle_point_down= approximate_angle_rise - first_point_rise
        
            append_to_log(f"top_bottom_angle_difference: {top_bottom_angle_difference} approximate_angle_rise: {approximate_angle_rise} first_point_rise: {first_point_rise} opposite_angle_point_down: {opposite_angle_point_down}")
            # line rise for nicha poiint
            if right_length > left_length:
                top_anlge_rise = top_anlge + first_point_rise
                top_angle_down = top_anlge + first_point_rise + opposite_angle_point_down
            else:
                top_anlge_rise = top_anlge - first_point_rise
                top_angle_down = top_anlge - first_point_rise - opposite_angle_point_down

            # slope m value from top_angle_right_rise
            top_rise_m = math.tan(math.radians(top_anlge_rise))
            top_down_m = math.tan(math.radians(top_angle_down))

            append_to_log(f"top_rise_m: {top_rise_m} top_down_m: {top_down_m} top_anlge_rise: {top_anlge_rise} top_angle_down: {top_angle_down}")
            



            # Decide which slope/intercept to use for each side
            if right_length < left_length:
                # Use top_rise for right, top_down for left
                # New Top-Right
                top_c_rise = get_intercept(top_rise_m, top_left_dum)
                if right_line_m == 0:
                    x = right_most_point["right_most"][0]
                    y = top_rise_m * x + top_c_rise
                    top_right_dum_temp = np.array([x, y])
                else:
                    top_right_dum_temp = intersection(top_rise_m, top_c_rise, right_line_m, right_line_c)

                top_c_down = get_intercept(top_down_m, top_right_dum_temp)

                # New Top-Left
                if right_line_m == 0:  # vertical line
                    x = left_most_point["left_most"][0]
                    y = top_down_m * x + top_c_down
                    top_left_dum_temp = np.array([x, y])
                else:
                    top_left_dum_temp = intersection(top_down_m, top_c_down, right_line_m, left_line_parallel_c)
            else:
                # Use top_down for right, top_rise for left
                top_c_rise = get_intercept(top_rise_m, top_right_dum)

                # New Top-Left
                if right_line_m == 0:  # vertical line
                    x = left_most_point["left_most"][0]
                    y = top_rise_m * x + top_c_rise
                    top_left_dum_temp = np.array([x, y])
                else:
                    top_left_dum_temp = intersection(top_rise_m, top_c_rise, right_line_m, left_line_parallel_c)

                top_c_down = get_intercept(top_down_m, top_left_dum_temp)
                    # New Top-Right
                if right_line_m == 0:
                    x = right_most_point["right_most"][0]
                    y = top_down_m * x + get_intercept(top_down_m, top_right_dum)
                    top_right_dum_temp = np.array([x, y])
                else:
                    top_right_dum_temp = intersection(top_down_m, get_intercept(top_down_m, top_right_dum), right_line_m, right_line_c)

            

            append_to_log(f"top_left_dum : {top_left_dum} top_right_dum: {top_right_dum} bottom_left_dum: {bottom_left_dum} bottom_right_dum: {bottom_right_dum}")
            top_left_dum = top_left_dum_temp
            top_right_dum = top_right_dum_temp
            append_to_log(f"top_left_dum_temp: {top_left_dum_temp} top_right_dum_temp: {top_right_dum_temp}")




        # # plotting datas, lines , corners and left and right coordinates
        # plt.figure(figsize=(10, 8))

        # # Final Y limits (inverted axis style)
        # plt.ylim(max(coordinates_array[:, 1]), min(coordinates_array[:, 1]))

        # plt.scatter(coordinates_array[:, 0], coordinates_array[:, 1], color='blue', label='Data Points')
        # plt.scatter([top_left[0], top_right[0], bottom_left[0], bottom_right[0]], [top_left[1], top_right[1], bottom_left[1], bottom_right[1]], color='red', label='Corners')
        # # plt.scatter([top_left_derived[0], top_right_derived[0], bottom_left_derived[0], bottom_right_derived[0]], [top_left_derived[1], top_right_derived[1], bottom_left_derived[1], bottom_right_derived[1]], color='green', label='Derived Corners')
        # plt.scatter([middle_left[0], middle_right[0]], [middle_left[1], middle_right[1]], color='yellow', label='Middle Points')
        # plt.scatter([coord[0] for coord in left], [coord[1] for coord in left], color='purple', label='Left Coordinates')
        # plt.scatter([coord[0] for coord in right], [coord[1] for coord in right], color='orange', label='Right Coordinates')
        # # plot all left avg points
        # plt.scatter([coord[0] for coord in left_avg], [coord[1] for coord in left_avg], color='deepskyblue', label='Left Avg Points')
        # # plot all right avg points
        # plt.scatter([coord[0] for coord in right_avg], [coord[1] for coord in right_avg], color='lime', label='Right Avg Points')
        # # plot finar corners
        # # plt.scatter([top_left_final[0], top_right_final[0], bottom_left_final[0], bottom_right_final[0]], [top_left_final[1], top_right_final[1], bottom_left_final[1], bottom_right_final[1]], color='cyan', label='Final Corners')
        # # plot new derived corners 
        # plt.scatter([new_top_left_derived[0], new_top_right_derived[0], new_bottom_left_derived[0], new_bottom_right_derived[0]], [new_top_left_derived[1], new_top_right_derived[1], new_bottom_left_derived[1], new_bottom_right_derived[1]], color='green')
        # # plt.scatter([top_left_final[0], top_right_final[0], bottom_left_final[0], bottom_right_final[0]], [top_left_final[1], top_right_final[1], bottom_left_final[1], bottom_right_final[1]], color='cyan', label='Final Corners')
        # plt.scatter([top_left_dum[0], top_right_dum[0], bottom_left_dum[0], bottom_right_dum[0]], [top_left_dum[1], top_right_dum[1], bottom_left_dum[1], bottom_right_dum[1]], color='cyan', label='Dummy Corners')

        # # plot line according to the slope and intercept of top and bottom
        # x_vals = np.linspace(min(coordinates_array[:, 0]), max(coordinates_array[:, 0]), 100)
        # y_vals = np.linspace(min(coordinates_array[:, 1]), max(coordinates_array[:, 1]), 100)
        # # append_to_log(f"x_vals: {x_vals}")


        # y_vals_top = top_m * x_vals + top_c
        # y_vals_bottom = bottom_m * x_vals + bottom_c
        # plt.plot(x_vals, y_vals_top, color='black', linestyle='--', label='Top Line')
        # plt.plot(x_vals, y_vals_bottom, color='black', linestyle='--', label='Bottom Line')

        # # plot line according to the slope and intercept of left and right
        # # Sort x_vals for proper segmentation

        # left_right_percentage = 0.2
        # x_vals_sorted = np.sort(x_vals)
        # n = len(x_vals_sorted)
        # left_x = np.array(get_10_splitted_values_array(top_left_dum[0], bottom_left_dum[0])) # x_vals_sorted[:int(n * left_right_percentage)]
        # right_x = np.array(get_10_splitted_values_array(top_right_dum[0], bottom_right_dum[0])) 

        # # right_x = x_vals_sorted[-int(n * 0.1):]

        # y_vals_left = left_line_m * left_x + left_line_c
        # y_vals_right = right_line_m * right_x + right_line_c



        # if left_line_m == 0:
        #     # line equation is x = top_left[0], draw vertical line
        #     plt.plot([ left_most_point["left_most"][0], left_most_point["left_most"][0]], [min(y_vals), max(y_vals)], color='purple', linestyle='--', label='Left Line')
        # else:
        #     plt.plot(left_x, y_vals_left, color='purple', linestyle='--', label='Left Line')
        # if right_line_m == 0:
        #     # line equation is y = top_right[0]
        #     plt.plot([right_most_point["right_most"][0], right_most_point["right_most"][0]], [min(y_vals), max(y_vals)], color='orange', linestyle='--', label='Right Line')
        
        # else:
        #     plt.plot(right_x, y_vals_right, color='orange', linestyle='--', label='Right Line')

        
        

        # plt.title(f"LEft and right illustration")
        # plt.xlabel('X')
        # plt.ylabel('Y')
        # plt.legend()
        # plt.grid(True)
        # plt.axis('equal')
        # plt.tight_layout()
        # # plt.show()
        # output_dir = 'output/clientpics/balanced'
        # output_filename = filename
        # if output_filename is None:

        #     os.makedirs(output_dir, exist_ok=True)

        #     # Generate a sanitized filename based on the title
        #     sanitized_title = re.sub(r'[^\w\-_\. ]', '_')
        #     output_filename = f"{output_dir}/{sanitized_title}.png"

        # plt.savefig(f"{output_dir}/{output_filename}")
        # plt.close()
        # append_to_log(f"Plot saved to: {output_filename}")
        # print(f"Plot saved to: {output_filename}")

        append_to_log(f"dum final points are top_left_dum: {top_left_dum} top_right_dum: {top_right_dum} bottom_left_dum: {bottom_left_dum} bottom_right_dum: {bottom_right_dum}")
        return top_left_dum, top_right_dum, bottom_left_dum, bottom_right_dum
    except Exception as e:
        append_to_log(f"get_left_right_line: Error in get_left_right_line: {e}")
        return None, None, None, None

def intersection(m1, c1, m2, c2):
   
    try:
        if m1 == m2:
            raise ValueError("Lines are parallel and do not intersect.")
        x = (c2 - c1) / (m1 - m2)
        y = m1 * x + c1
        return np.array([x, y])
    except Exception as e:
        append_to_log(f"intersection: Error in intersection: {e}")
        return None




def get_ransac_line(coordinates_array, max_distance=10.0, min_samples=6):
    try:
        coordinates_array = np.array(coordinates_array)
        lengths = len(coordinates_array)
        min_samples = min(lengths, min_samples)
        x = coordinates_array[:, 0].reshape(-1, 1)
        y = coordinates_array[:, 1]

        # Fit RANSAC model
        ransac = RANSACRegressor(
            estimator=LinearRegression(),
            residual_threshold=max_distance,
            min_samples=min_samples,
            random_state=0
        )
        ransac.fit(x, y)

        # Get slope and intercept from the inlier model
        m = ransac.estimator_.coef_[0]
        c = ransac.estimator_.intercept_
        return m, c
    except Exception as e:
        append_to_log(f"intersection: Error in intersection: {e}")
        return None



def get_regression_line(coordinates_array):
    try:
        coordinates_array = np.array(coordinates_array)
        x_vals = coordinates_array[:, 0]
        y_vals = coordinates_array[:, 1]

        # Fit a linear regression model
        reg = LinearRegression().fit(x_vals.reshape(-1, 1), y_vals)

        # Get slope (m) and intercept (c)
        m = reg.coef_[0]
        c = reg.intercept_

        return m, c
    except Exception as e:
        append_to_log(f"intersection: Error in intersection: {e}")
        return None




def perpendicular_distances(m, c, coordinates):
    try:
        # Calculates perpendicular distances from points to the line y = mx + c
        x = coordinates[:, 0]
        y = coordinates[:, 1]
        return np.abs(m * x - y + c) / np.sqrt(m**2 + 1)
    except Exception as e:
        append_to_log(f"intersection: Error in intersection: {e}")
        return None

def get_regression_line_conditional(coordinates_array, min_points=4, max_distance=4):
    try:
        coordinates_array = np.array(coordinates_array)

        n = len(coordinates_array)
        min_points = n
        best_m = None
        best_c = None
        max_count = -1  # Track maximum number of close points

        # Try all combinations of at least min_points points (min needed for regression)
        for r in range(min_points, n + 1):
            for subset in combinations(coordinates_array, r):
                subset = np.array(subset)
                x_vals = subset[:, 0]
                y_vals = subset[:, 1]

                # Fit linear regression
                reg = LinearRegression().fit(x_vals.reshape(-1, 1), y_vals)
                m = reg.coef_[0]
                c = reg.intercept_

                # Count points within distance threshold
                distances = perpendicular_distances(m, c, coordinates_array)
                count_within_threshold = np.sum(distances <= max_distance)

                # Update best line if current line is better
                if count_within_threshold > max_count:
                    max_count = count_within_threshold
                    best_m = m
                    best_c = c

        if best_m is not None:
            return best_m, best_c
        else:
            return None, None
    except Exception as e:
        append_to_log(f"intersection: Error in intersection: {e}")
        return None



def get_line_properties(p1, p2):
    try:
        x1, y1 = p1
        x2, y2 = p2

        dx = x2 - x1
        dy = y2 - y1

        result = {}

        if dx == 0:
            # Vertical line
            result['slope'] = None
            result['angle_rad'] = math.pi / 2
            result['angle_deg'] = 90.0
            result['intercept'] = None  # No y-intercept
            result['type'] = 'vertical'
        else:
            m = dy / dx
            angle_rad = math.atan(m)
            angle_deg = math.degrees(angle_rad)
            intercept = y1 - m * x1

            result['slope'] = m
            result['angle_rad'] = angle_rad
            result['angle_deg'] = angle_deg
            result['intercept'] = intercept
            result['type'] = 'normal'

        return result
    except Exception as e:
        append_to_log(f"intersection: Error in intersection: {e}")
        return None






    

