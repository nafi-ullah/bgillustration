import cv2
import numpy as np
from ultralytics import YOLO
import os
from io import BytesIO
from shapely.geometry import Polygon, LineString, Point, MultiPoint, GeometryCollection, MultiLineString

model_path = 'models/wheels/wheeldetect3folder/wheel-3folder-best.pt'
fullcar_model_path = 'models/full-car-1800images/full-car-1800images-best.pt'
from my_logging_script import log_to_json
from features.functionalites.backgroundoperations.utlis import calculate_six_points, calculate_shadow_points, save_image_with_timestamp

current_file_name = "features/functionalites/backgroundoperations/roundedsegmented.py"
headlight_model_path = 'models/headlights/v900/v900_best.pt'
def detect_wheels_and_annotate(image_bytes_io, output_folder):
    try:
    # Load the image from BytesIO
        log_to_json(f"got the file for detect wheels coordinate", current_file_name)
        image_bytes_io.seek(0)
        img_array = np.frombuffer(image_bytes_io.read(), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            log_to_json(f"detect_wheels_and_annotate-- Error: Unable to read image.", current_file_name)
            return None

        original_height, original_width = img.shape[:2]

        # Load the YOLO model for segmentation
        model = YOLO(model_path)

        # Run inference on the image
        results = model(img)
        if not results:
            log_to_json("detect_wheels_and_annotate-- No wheel found in car by the YOLO model.", current_file_name)
            return None

        # Store masks and their bounding boxes
        masks_and_boxes = []

        # Iterate over the detected results
        for result in results:
            if result.masks is None:
                log_to_json("detect_wheels_and_annotate-- No masks found for the detected object.", current_file_name)
                continue
            for mask in result.masks.data:
                # Process the mask
                mask = (mask.cpu().numpy() * 255).astype('uint8')
                mask_resized = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

                # Find contours and bounding box
                contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    x, y, w, h = cv2.boundingRect(contours[0])
                    masks_and_boxes.append({"mask": mask_resized, "bbox": (x, y, w, h)})

        # Ensure there are at least two masks detected
        # if len(masks_and_boxes) < 2:
        #     print("Error: Less than two wheels detected.")
        #     return

        # Sort masks by their horizontal position (x-coordinate of bounding box)
        masks_and_boxes.sort(key=lambda x: x["bbox"][0])

        # Assign front and back wheels
        front_wheel = masks_and_boxes[0]  # Leftmost mask
        back_wheel = masks_and_boxes[-1]  # Rightmost mask

        # Extract coordinates
        front_leftmost_x = front_wheel["bbox"][0]  # Leftmost X of front wheel
        front_leftmost_y = front_wheel["bbox"][1]  # Top Y of front wheel

        front_bottom_x = front_wheel["bbox"][0] + front_wheel["bbox"][2] // 2  # Center X of front wheel at the bottom
        front_bottom_y = front_wheel["bbox"][1] + front_wheel["bbox"][3]  # Bottom Y of front wheel

        back_bottom_x = back_wheel["bbox"][0] + back_wheel["bbox"][2] // 2  # Center X of back wheel at the bottom
        back_bottom_y = back_wheel["bbox"][1] + back_wheel["bbox"][3]  # Bottom Y of back wheel

        # Print the extracted coordinates for debugging
        print(f"Front Wheel - Leftmost (X, Y): ({front_leftmost_x}, {front_leftmost_y})")
        print(f"Front Wheel - Bottom (X, Y): ({front_bottom_x}, {front_bottom_y})")
        print(f"Back Wheel - Bottom (X, Y): ({back_bottom_x}, {back_bottom_y})")

        # Draw circles on the image
        annotated_img = img.copy()
        # Circle for front wheel leftmost coordinate
        cv2.circle(annotated_img, (front_leftmost_x, front_leftmost_y), 10, (0, 0, 255), -1)
        cv2.circle(annotated_img, (front_bottom_x, front_bottom_y), 10, (0, 255, 0), -1)
        # Circle for back wheel bottom-most coordinate
        cv2.circle(annotated_img, (back_bottom_x, back_bottom_y), 10, (255, 0, 0), -1)  # Blue circle

        # Save the annotated image using save_image_with_timestamp
        _, buffer = cv2.imencode('.png', annotated_img)
        annotated_image_bytes = BytesIO(buffer.tobytes())
        #output_path = save_image_with_timestamp(annotated_image_bytes, "outputdebug/coordinate", "image.png")

        #print(f"Annotated image saved to '{output_path}'.")
        return {
            "front_left_x": front_leftmost_x,
            "front_left_y": front_leftmost_y,
            "front_bottom_x": front_bottom_x,
            "front_bottom_y": front_bottom_y,
            "back_bottom_x": back_bottom_x,
            "back_bottom_y": back_bottom_y
        }
    except FileNotFoundError as e:
        log_to_json(f"detect_wheels_and_annotate-- File not found error {e}", current_file_name)
        return None
    except ValueError as e:
        log_to_json(f"detect_wheels_and_annotate-- ValueError {e}", current_file_name)
        return None
    except Exception as e:
        log_to_json(f"detect_wheels_and_annotate-- Exceptional error {e}", current_file_name)
        return  None
    
def detect_left_right_wheels(image_bytes_io):
    try:
        # Load the image from BytesIO
        log_to_json(f"got the file for detect wheels coordinate", current_file_name)
        image_bytes_io.seek(0)
        img_array = np.frombuffer(image_bytes_io.read(), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            log_to_json(f"detect_left_right_wheels-- Error: Unable to read image.", current_file_name)
            return None

        original_height, original_width = img.shape[:2]

        # Load the YOLO model for segmentation
        model = YOLO(model_path)

        # Run inference on the image
        results = model(img)

        if not results:
            log_to_json("detect_left_right_wheels-- No wheels found in car by the YOLO model.", current_file_name)
            return None

        # Store masks with their areas
        masks_with_areas = []

        # Iterate over the detected results
        for result in results:
            if result.masks is None:
                log_to_json("detect_left_right_wheels-- No masks found for the detected object.", current_file_name)
                continue
            for mask in result.masks.data:
                # Process the mask
                mask = (mask.cpu().numpy() * 255).astype('uint8')
                mask_resized = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

                # Calculate the area of the mask
                area = np.sum(mask_resized > 0)

                # Store the mask and its area
                masks_with_areas.append({"mask": mask_resized, "area": area})

        # Ensure at least two masks are detected
        if len(masks_with_areas) < 2:
            print("Error: Less than two masks detected.")
            return

        # Sort masks by area in descending order
        masks_with_areas.sort(key=lambda x: x["area"], reverse=True)

        # Get the two largest masks
        largest_mask = masks_with_areas[0]["mask"]
        second_largest_mask = masks_with_areas[1]["mask"]

        # Function to find the bottom-most coordinate of a mask
        def get_bottom_most(mask):
            indices = np.column_stack(np.where(mask > 0))
            if indices.size > 0:
                bottom_most_idx = np.argmax(indices[:, 0])  # Row index for the y-axis
                bottom_x = indices[bottom_most_idx, 1]  # Column index (x-axis)
                bottom_y = indices[bottom_most_idx, 0]  # Row index (y-axis)
                return bottom_x, bottom_y
            return None, None

        # Get bottom-most coordinates for the two largest masks
        largest_bottom_x, largest_bottom_y = get_bottom_most(largest_mask)
        second_largest_bottom_x, second_largest_bottom_y = get_bottom_most(second_largest_mask)

        # Annotate the image
        annotated_img = img.copy()
        if largest_bottom_x is not None and largest_bottom_y is not None:
            cv2.circle(annotated_img, (largest_bottom_x, largest_bottom_y), 10, (0, 0, 255), -1)  # Red for largest
        if second_largest_bottom_x is not None and second_largest_bottom_y is not None:
            cv2.circle(annotated_img, (second_largest_bottom_x, second_largest_bottom_y), 10, (0, 255, 0), -1)  # Green for second largest

        # Save the annotated image
        _, buffer = cv2.imencode('.png', annotated_img)
        annotated_image_bytes = BytesIO(buffer.tobytes())
        # output_path = save_image_with_timestamp(annotated_image_bytes, output_folder, "image.png")

        # print(f"Annotated image saved to '{output_path}'.")
        return {
            "front_left_x": largest_bottom_x +250,
            "front_left_y": largest_bottom_y - 150,
            "front_bottom_x": largest_bottom_x,
            "front_bottom_y": largest_bottom_y,
            "back_bottom_x": second_largest_bottom_x,
            "back_bottom_y": second_largest_bottom_y
        }
    except FileNotFoundError as e:
        log_to_json(f"detect_left_right_wheels-- File not found error {e}", current_file_name)
        return None
    except ValueError as e:
        log_to_json(f"detect_left_right_wheels-- ValueError {e}", current_file_name)
        return None
    except Exception as e:
        log_to_json(f"detect_left_right_wheels- Exceptional error {e}", current_file_name)
        return None



def detect_cars_coordinates(image_bytes_io):
    try:
        # Load the image from BytesIO
        log_to_json(f"got the file for detect wheels coordinate", current_file_name)
        image_bytes_io.seek(0)
        img_array = np.frombuffer(image_bytes_io.read(), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            log_to_json(f"detect_cars_coordinates-- Error: Unable to load image from BytesIO", current_file_name)
            return None

        original_height, original_width = img.shape[:2]

        # Load the YOLO model for segmentation
        model = YOLO(fullcar_model_path)

        # Run inference on the image
        results = model(img)

        if not results:
            log_to_json("detect_cars_coordinates -- No cars found in car by the YOLO model.", current_file_name)
            return None

        # Initialize variables
        largest_mask = None
        largest_area = 0
        detected_masks_count = 0

        # Process each result and find the largest mask
        for result in results:
            if result.masks is None:
                log_to_json("detect_cars_coordinates-- No masks found for the detected object.", current_file_name)
                continue
            for mask in result.masks.data:
                mask = (mask.cpu().numpy() * 255).astype('uint8')
                mask_resized = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

                # Calculate the area of the current mask
                current_area = np.sum(mask_resized > 0)
                detected_masks_count += 1

                # Keep track of the largest mask
                if current_area > largest_area:
                    largest_area = current_area
                    largest_mask = mask_resized

        print(f"Number of detected masks: {detected_masks_count}")

        if largest_mask is not None:
            # Find all coordinates of the mask
            indices = np.column_stack(np.where(largest_mask > 0))

            if indices.size > 0:
                # Left-most coordinate
                left_most_idx = np.argmin(indices[:, 1])
                left_most_x = indices[left_most_idx, 1]
                left_most_y = indices[left_most_idx, 0]

                # Bottom-left coordinate
                vertical_line_mask = (indices[:, 1] >= left_most_x) & (indices[:, 1] <= left_most_x)
                filtered_points = indices[vertical_line_mask]
                bottom_leftmost_x, bottom_leftmost_y = None, None
                if filtered_points.size > 0:
                    bottom_most_idx = np.argmax(filtered_points[:, 0])
                    bottom_leftmost_x = filtered_points[bottom_most_idx, 1]
                    bottom_leftmost_y = filtered_points[bottom_most_idx, 0]

                # Top-left coordinate
                top_leftmost_x, top_leftmost_y = left_most_x, left_most_y - 80

                # Right-most coordinate
                right_most_idx = np.argmax(indices[:, 1])
                right_most_x = indices[right_most_idx, 1]
                right_most_y = indices[right_most_idx, 0]

                # Bottom-right coordinate
                vertical_line_mask_right = (indices[:, 1] <= right_most_x) & (indices[:, 1] >= right_most_x)
                filtered_points_right = indices[vertical_line_mask_right]
                bottom_rightmost_x, bottom_rightmost_y = None, None
                if filtered_points_right.size > 0:
                    bottom_most_idx = np.argmax(filtered_points_right[:, 0])
                    bottom_rightmost_x = filtered_points_right[bottom_most_idx, 1]
                    bottom_rightmost_y = filtered_points_right[bottom_most_idx, 0]

                # Top-right coordinate
                top_rightmost_x, top_rightmost_y = None, None
                if filtered_points_right.size > 0:
                    top_most_idx = np.argmin(filtered_points_right[:, 0])
                    top_rightmost_x = filtered_points_right[top_most_idx, 1]
                    top_rightmost_y = filtered_points_right[top_most_idx, 0]

                # Annotate the image for debugging
                annotated_img = img.copy()
                if bottom_leftmost_x is not None and bottom_leftmost_y is not None:
                    cv2.circle(annotated_img, (bottom_leftmost_x, bottom_leftmost_y), 10, (0, 0, 255), -1)  # Red for bottom-left
                cv2.circle(annotated_img, (top_leftmost_x, top_leftmost_y), 10, (255, 0, 0), -1)  # Blue for top-left
                if bottom_rightmost_x is not None and bottom_rightmost_y is not None:
                    cv2.circle(annotated_img, (bottom_rightmost_x, bottom_rightmost_y), 10, (0, 255, 0), -1)  # Green for bottom-right
                if top_rightmost_x is not None and top_rightmost_y is not None:
                    cv2.circle(annotated_img, (top_rightmost_x, top_rightmost_y), 10, (255, 255, 0), -1)  # Yellow for top-right

                # Save the annotated image
                _, buffer = cv2.imencode('.png', annotated_img)
                annotated_image_bytes = BytesIO(buffer.tobytes())
                # output_path = save_image_with_timestamp(annotated_image_bytes, "outputdebug/carcoordinate", "image.png")
                # print(f"Annotated image saved to '{output_path}'.")

                print(f"Priority Bottom-Left Coordinate: ({bottom_leftmost_x}, {bottom_leftmost_y})")
                print(f"Priority Top-Left Coordinate: ({top_leftmost_x}, {top_leftmost_y})")
                print(f"Priority Bottom-Right Coordinate: ({bottom_rightmost_x}, {bottom_rightmost_y})")
                print(f"Priority Top-Right Coordinate: ({top_rightmost_x}, {top_rightmost_y})")
                return {
                    "cars_bottom_left_x": int(bottom_leftmost_x),
                    "cars_bottom_left_y": int(bottom_leftmost_y),
                    "cars_top_left_x": int(top_leftmost_x),
                    "cars_top_left_y": int(top_leftmost_y),
                    "cars_bottom_right_x": int(bottom_rightmost_x),
                    "cars_bottom_right_y": int(bottom_rightmost_y),
                    "cars_top_right_x": int(top_rightmost_x),
                    "cars_top_right_y": int(top_rightmost_y),
                }

        print("No car detected in the image.")
        return None

    except FileNotFoundError as e:
        log_to_json(f"detect_cars_coordinates-- File not found error {e}", current_file_name)
        return None
    except ValueError as e:
        log_to_json(f"detect_cars_coordinates -- ValueError {e}", current_file_name)
        return None
    except Exception as e:
        log_to_json(f"detect_cars_coordinates -- Exceptional error {e}", current_file_name)
        return  None
    

def headlight_points(image_bytes: BytesIO):
    try:
        # Log the start of the function
        log_to_json("Got the file in wheel_headlight_process function", current_file_name)

        # Load the input image
        image_bytes.seek(0)  # Ensure the BytesIO stream is at the start
        image_array = np.frombuffer(image_bytes.read(), np.uint8)
        if image_array.size == 0:
            log_to_json("headlight_points: Error: Empty image buffer.", current_file_name)
            return None

        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if img is None:
            log_to_json("headlight_points: Error: Unable to decode image.", current_file_name)
            return None

        original_height, original_width = img.shape[:2]

        # Load the YOLO model
        model = YOLO(headlight_model_path)  # Replace with your model path

        # Run inference on the image
        results = model(img)
        if not results:
            log_to_json("headlight_points: No headlight or wheel found in car by the YOLO model.", current_file_name)
            return None

        # Combine all masks into a single binary mask
        combined_mask = np.zeros((original_height, original_width), dtype=np.uint8)
        for result in results:
            if result.masks is None:
                log_to_json("headlight_points: No masks found for the detected object.", current_file_name)
                continue

            for mask in result.masks.data:
                mask = (mask.cpu().numpy() * 255).astype('uint8')
                mask_resized = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
                combined_mask = cv2.bitwise_or(combined_mask, mask_resized)

        # Find contours in the combined mask
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            log_to_json("headlight_points: No segmented areas found.", current_file_name)
            return None

        # Find the largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)

        # Find extreme points of the largest contour
        top_left = tuple(largest_contour[np.argmin(np.sum(largest_contour[:, 0], axis=1))][0])
        top_right = tuple(largest_contour[np.argmax(np.diff(largest_contour[:, 0], axis=1))][0])
        bottom_left = tuple(largest_contour[np.argmin(np.diff(largest_contour[:, 0], axis=1))][0])
        bottom_right = tuple(largest_contour[np.argmax(np.sum(largest_contour[:, 0], axis=1))][0])

        # Calculate the midpoint of the trapezium
        midpoint_x = (top_left[0] + top_right[0] + bottom_left[0] + bottom_right[0]) // 4
        midpoint_y = (top_left[1] + top_right[1] + bottom_left[1] + bottom_right[1]) // 4

        return {"headlight_mid_x": midpoint_x, "headlight_mid_y": midpoint_y}

    except FileNotFoundError as e:
        log_to_json(f"headlight_points: File not found error {e}", current_file_name)
        return None
    except ValueError as e:
        log_to_json(f"headlight_points: ValueError {e}", current_file_name)
        return None
    except Exception as e:
        log_to_json(f"headlight_points: Exceptional error {e}", current_file_name)
        return None
    

def detect_cars_top_bottom_certainx(image_bytes_io, x):
    try:
        # Load the image from BytesIO
        image_bytes_io.seek(0)
        img_array = np.frombuffer(image_bytes_io.read(), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            print("Error: Unable to load image from BytesIO")
            return None

        original_height, original_width = img.shape[:2]

        # Load the YOLO model for segmentation
        model = YOLO(fullcar_model_path)  # Replace with your model path

        # Run inference on the image
        results = model(img)

        if not results:
            print("No cars found in the image by the YOLO model.")
            return None

        # Initialize variables
        largest_mask = None
        largest_area = 0

        # Process each result and find the largest mask
        for result in results:
            if result.masks is None:
                continue

            for mask in result.masks.data:
                # Convert the mask to binary format
                mask = (mask.cpu().numpy() * 255).astype('uint8')
                mask_resized = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

                # Calculate the area of the current mask
                current_area = np.sum(mask_resized > 0)

                # Keep track of the largest mask
                if current_area > largest_area:
                    largest_area = current_area
                    largest_mask = mask_resized

        if largest_mask is None:
            print("No valid masks found.")
            return None

        # Extract the polygonal contour from the largest mask
        contours, _ = cv2.findContours(largest_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print("No contours found in the largest mask.")
            return None

        # Assume the largest contour represents the object
        largest_contour = max(contours, key=cv2.contourArea)
        polygon = Polygon([pt[0] for pt in largest_contour])
        if not polygon.is_valid:
            print("The generated polygon is invalid.")
            return None


        # Define the vertical line at x = x
        line = LineString([(x, 0), (x, original_height)])

        # Find intersections between the polygon and the vertical line
        intersection = polygon.intersection(line)

        print(f"Intersection type: {type(intersection)}")
        print(f"Intersection details: {intersection}")


        if intersection.is_empty:
            print("No intersection found with the vertical line.")
            return None

        # Extract the y-coordinates of the intersection points
        y_coordinates = []
        if isinstance(intersection, Point):
            y_coordinates.append(intersection.y)
        elif isinstance(intersection, MultiPoint):
            y_coordinates.extend([point.y for point in intersection])
        elif isinstance(intersection, LineString):
            y_coordinates.extend([coord[1] for coord in intersection.coords])
        elif isinstance(intersection, MultiLineString):
            # Use .geoms to iterate over the LineStrings in the MultiLineString
            for linestring in intersection.geoms:
                y_coordinates.extend([coord[1] for coord in linestring.coords])
        elif isinstance(intersection, GeometryCollection):
            for geom in intersection.geoms:
                if isinstance(geom, Point):
                    y_coordinates.append(geom.y)
                elif isinstance(geom, LineString):
                    y_coordinates.extend([coord[1] for coord in geom.coords])
        else:
            print(f"Unhandled geometry type for intersection: {type(intersection)}")
            print(f"Intersection details: {intersection}")
            return None

        if not y_coordinates:
            print("No y-coordinates found from the intersection.")
            return None

        # Find the topmost and bottommost y-coordinates
        topmost = min(y_coordinates)
        bottommost = max(y_coordinates)

        return {
            "topmost_coord" : (x, topmost),
            "bottommost_coord" : (x, bottommost)
        }

    except Exception as e:
        print(f"Error in detect_cars_top_bottom_certainx: {e}")
        return None


def full_car_bounding_box_crop(image_bytes_io):
    # Load the image from BytesIO
    try:
        image_bytes_io.seek(0)
        img_array = np.frombuffer(image_bytes_io.read(), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            log_to_json("Error: Unable to load image from BytesIO.", current_file_name)
            return None

        original_height, original_width = img.shape[:2]

        # Load the YOLO model for segmentation
        model = YOLO(fullcar_model_path)

        # Run inference on the image
        results = model(img)
        if not results:
            log_to_json("full_car_bounding_box_crop: No cars found in car image by the YOLO model.", current_file_name)
            return None

        # Initialize variables
        car_bounding_box = None
        detected_cars_count = 0

        # Process each result to find bounding boxes
        for result in results:
            if result.masks is None:
                log_to_json("full_car_bounding_box_crop - No masks found for the detected object.", current_file_name)
                continue
            for box in result.boxes.data:  # Access detected boxes
                if box[4] > 0.5:  # Only consider detections with confidence > 0.5
                    detected_cars_count += 1
                    x1, y1, x2, y2 = map(int, box[:4])  # Get coordinates of the bounding box
                    car_bounding_box = (x1-15, y1-10, x2+15, y2+15)
                    break  # Assume one car in the image, break after finding the first car bounding box

        print(f"Number of detected cars: {detected_cars_count}")

        if car_bounding_box is not None:
            x1, y1, x2, y2 = car_bounding_box

            # Crop the image based on the bounding box
            cropped_img = img[y1:y2, x1:x2]

            # Convert the cropped image to a byte stream
            _, cropped_img_bytes = cv2.imencode('.png', cropped_img)
            cropped_img_bytes_io = BytesIO(cropped_img_bytes)

            print(f"Cropped image with bounding box coordinates: {car_bounding_box}")

            return cropped_img_bytes_io
        else:
            print("No car detected in the image.")
            return None
    except Exception as e:
        log_to_json(f"full_car_bounding_box_crop -- Error occurs for {e}", current_file_name)
        return None


def full_car_maskdetect_cars_bottomleft(image_bytes_io):
    # Load the image from BytesIO
    try:
        image_bytes_io.seek(0)
        img_array = np.frombuffer(image_bytes_io.read(), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            log_to_json("full_car_maskdetect_cars_bottomleft: Error: Unable to load image from BytesIO.", current_file_name)
            return None

        original_height, original_width = img.shape[:2]

        # Load the YOLO model for segmentation
        model = YOLO(fullcar_model_path)

        # Run inference on the image
        results = model(img)
        if not results:
            log_to_json("full_car_maskdetect_cars_bottomleft: No full car found in car image by the YOLO model.", current_file_name)
            return None

        # Initialize variables
        largest_mask = None
        largest_area = 0
        detected_masks_count = 0

        # Process each result and find the largest mask
        for result in results:
            if result.masks is None:
                log_to_json("full_car_maskdetect_cars_bottomleft: No masks found for the detected object.", current_file_name)
                continue
            for mask in result.masks.data:
                # Process the mask
                mask = (mask.cpu().numpy() * 255).astype('uint8')
                mask_resized = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

                # Calculate the area of the current mask
                current_area = np.sum(mask_resized > 0)
                detected_masks_count += 1

                # Keep track of the largest mask
                if current_area > largest_area:
                    largest_area = current_area
                    largest_mask = mask_resized

        print(f"Number of detected masks: {detected_masks_count}")

        if largest_mask is not None:
            # Find the bottom-left coordinate of the largest mask
            indices = np.column_stack(np.where(largest_mask > 0))
            if indices.size > 0:
                # Bottom-most row is determined by maximum y-value
                bottom_most = indices[np.argmax(indices[:, 0])]
                bottom_leftmost_x = bottom_most[1]
                bottom_leftmost_y = bottom_most[0]

                # Annotate the original image with the mask
                annotated_img = img.copy()
                mask_overlay = cv2.merge([largest_mask, largest_mask, largest_mask])  # Convert mask to 3 channels
                annotated_img = cv2.addWeighted(annotated_img, 0.7, mask_overlay, 0.3, 0)  # Overlay the mask
                
                # Highlight the bottom-left coordinate with a red circle
                cv2.circle(annotated_img, (bottom_leftmost_x, bottom_leftmost_y), 10, (0, 0, 255), -1)  # Red circle

                # Save the annotated image
                output_folder = 'outputdebug'
                os.makedirs(output_folder, exist_ok=True)
                output_path = os.path.join(output_folder, "full_car_mask.png")
                cv2.imwrite(output_path, annotated_img)
                print(f"Annotated image with full car mask saved to '{output_path}'.")

                print(f"Largest Mask's Bottom-Left Coordinate: ({bottom_leftmost_x}, {bottom_leftmost_y})")
                return {
                    "cars_bottom_left_x": int(bottom_leftmost_x),
                    "cars_bottom_left_y": int(bottom_leftmost_y),
                }

        print("No car detected in the image.")
        return None
    except Exception as e:
        log_to_json(f"full_car_maskdetect_cars_bottomleft: Error occurs for {e}", current_file_name)
        return None



def segment_and_save_image(model_path, input_image_path, output_image_path):
    # Load the original image
    try:
        img = cv2.imread(input_image_path)
        if img is None:
            print(f"Error: Unable to read image {input_image_path}")
            return

        original_height, original_width = img.shape[:2]

        # Load the YOLO model for segmentation
        model = YOLO(model_path)

        # Run inference on the image
        results = model(img)
        if not results:
            log_to_json("segment_and_save_image: No segment in segment_and_save_image func  found in car by the YOLO model.", current_file_name)
            return None

        # Create a blank mask with the same dimensions as the original image
        combined_mask = np.zeros((original_height, original_width), dtype=np.uint8)

        # Iterate over the detected results to combine each mask
        for result in results:
            if result.masks is None:
                log_to_json("segment_and_save_image: No masks found for the detected object.", current_file_name)
                continue
            for mask in result.masks.data:
                # Move the mask to CPU if needed and scale to 0-255
                mask = (mask.cpu().numpy() * 255).astype('uint8')

                # Resize the mask to match the original image dimensions
                mask_resized = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

                # Combine the resized mask with the blank mask using bitwise OR
                combined_mask = cv2.bitwise_or(combined_mask, mask_resized)

        # Create a gray version of the mask
        gray_value = (181, 178, 171)

        # Create a new image to hold the segmented output
        segmented_image = np.zeros_like(img)

        # Fill the segmented areas with the gray color
        segmented_image[combined_mask > 0] = gray_value

        # Create a mask for rounding corners
        rounded_mask = cv2.GaussianBlur(combined_mask, (15, 15), 0)  # Apply Gaussian blur to soften edges
        rounded_mask = np.where(rounded_mask > 0, 255, 0).astype('uint8')  # Convert to binary mask

        # Combine the original image and the gray mask
        final_image = np.where(segmented_image == 0, img, segmented_image)

        # Use the rounded mask to blend the edges
        final_image = cv2.addWeighted(final_image, 1, np.zeros_like(final_image), 0, 0)  # Placeholder for blending
        final_image[rounded_mask > 0] = gray_value

        # Save the resulting image
        cv2.imwrite(output_image_path, final_image)
        print(f"Image segmented and saved as '{output_image_path}'.")
    except Exception as e:
        log_to_json(f"segment_and_save_image: Error occurs for {e}", current_file_name)
        return None

# def process_all_images_in_folder(model_path, input_folder):
#     # Set output folder to the desired path
#     # output_folder = './output/v495images'
#     os.makedirs(output_folder, exist_ok=True)

#     # Process each image in the input folder
#     for filename in os.listdir(input_folder):
#         if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
#             input_image_path = os.path.join(input_folder, filename)

#             # Construct the output file name
#             name, ext = os.path.splitext(filename)
#             output_image_path = os.path.join(output_folder, f"{name}_blurred_licenseplate{ext}")

#             # Segment and save the image
#             segment_and_save_image(model_path, input_image_path, output_image_path)




# Define paths to the model and the input folder


# Process all images in the folder
# process_all_images_in_folder(model_path, input_folder)
