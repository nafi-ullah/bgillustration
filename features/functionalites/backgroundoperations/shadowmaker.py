import numpy as np
import cv2
from PIL import Image
from io import BytesIO
from scipy.interpolate import splprep, splev
from my_logging_script import log_to_json
from features.functionalites.backgroundoperations.carbasicoperations import rotate_image_by_angle,apply_blur_and_opacity 
from features.functionalites.backgroundoperations.utlis import calculate_six_points, calculate_shadow_points, save_image_with_timestamp
current_file_name = "features/functionalites/backgroundoperations/shadowmaker.py"
def calculate_curve(p1, p2, direction="clockwise", radius_factor=1 / 6):
    try:
        p1 = np.array(p1, dtype=np.float32)
        p2 = np.array(p2, dtype=np.float32)
        midpoint = (p1 + p2) / 2  # Calculate midpoint
        distance = np.linalg.norm(p2 - p1)
        radius = radius_factor * distance

        # Offset midpoint for curvature
        direction_vector = np.array([-(p2[1] - p1[1]), p2[0] - p1[0]])  # Perpendicular vector
        direction_vector = direction_vector / np.linalg.norm(direction_vector)  # Normalize direction

        if direction == "clockwise":
            control_point = midpoint + direction_vector * radius
        elif direction == "counterclockwise":
            control_point = midpoint - direction_vector * radius
        else:
            raise ValueError("Direction must be 'clockwise' or 'counterclockwise'")

        # Create curve points using cubic Bezier curve
        curve_points = np.array([p1, control_point, p2], dtype=np.float32)

        # Interpolate curve points
        tck, u = splprep([curve_points[:, 0], curve_points[:, 1]], s=0, k=min(3, len(curve_points) - 1))
        u_fine = np.linspace(0, 1, 100)
        x_fine, y_fine = splev(u_fine, tck)
        return np.array([x_fine, y_fine]).T  # Return as a 2D array
    except Exception as e:
        log_to_json(f"Error occurs for {e}", current_file_name)
        return None

def create_shadow_shape(curve_points, straight_points, image_size=(1500, 800), blur_radius=30, thickness=2):
    log_to_json(f"Curve points: {curve_points}", current_file_name)
    log_to_json(f"Straight points: {straight_points}", current_file_name)
    try:
    # Create a blank transparent canvas
        canvas = np.zeros((image_size[1], image_size[0], 4), dtype=np.uint8)

        # Generate points for the final shape
        shape_points = []
        for curve_data in curve_points:
            p1, p2 = list(curve_data["points"])
            direction = curve_data.get("direction", "clockwise")
            curve_points_interpolated = calculate_curve(p1, p2, direction=direction)
            shape_points.extend(curve_points_interpolated.astype(np.int32).tolist())

        for straight_set in straight_points:
            p1, p2 = list(straight_set)
            straight_points_np = np.array([p1, p2], dtype=np.int32)
            shape_points.extend(straight_points_np.tolist())

        # Convert shape points to a NumPy array for OpenCV
        shape_points = np.array(shape_points, dtype=np.int32)

        # Create a mask for the shape
        mask = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)
        cv2.fillPoly(mask, [shape_points], 255)  # Solid white shape

        # Create the shadow effect
        shadow_mask = cv2.GaussianBlur(mask, (2 * blur_radius + 1, 2 * blur_radius + 1), 0)  # Blurred mask
        shadow_layer = np.zeros_like(canvas, dtype=np.uint8)

        for c in range(3):  # Add shadow color to R, G, B channels
            shadow_layer[..., c] = (shadow_mask / 255 * 0).astype(np.uint8)

        # Blend shadow layer with transparency
        shadow_layer[..., 3] = shadow_mask  # Use the blurred mask as alpha

        # Overlay shadow on the canvas
        filled_shape = canvas.copy()
        cv2.fillPoly(filled_shape, [shape_points], (0, 0, 0, 255))  # Fill shape with solid black
        final_canvas = cv2.addWeighted(shadow_layer, 0.7, filled_shape, 1, 0)  # Blend shadow and filled shape

        # Convert the image to Pillow format for transparency handling and save to byte buffer
        pil_image = Image.fromarray(final_canvas, mode="RGBA")
        byte_io = BytesIO()
        pil_image.save(byte_io, format="PNG")
        byte_io.seek(0)

        output_path = save_image_with_timestamp(byte_io, "outputdebug/shadow", "shadow.png")
        # output_path = "outputdebug/shadowed_shape_done.png"
        # with open(output_path, "wb") as f:
        #     f.write(byte_io.read()) 
        # print(f"Image saved to {output_path}")

        # Return byte byteio format
        return byte_io
    except FileNotFoundError as e:
        log_to_json(f"File not found error {e}", current_file_name)
        return f'Sorry the image is not processed for: File not found: {str(e)}'
    except ValueError as e:
        log_to_json(f"ValueError {e}", current_file_name)
        return f'Sorry the image is not processed for: Value error: {str(e)}'
    except Exception as e:
        log_to_json(f"Exceptional error {e}", current_file_name)
        return  f'Sorry : An unexpected error occurred: {str(e)}'

# Example usage
# curve_pts = [
#     {"points": {(21, 535), (482, 665)}, "direction": "counterclockwise"},
# ]
# straight_pts = [{(482, 665), (1297, 463)}, {(1297, 463), (1147, 331)}, {(21, 535), (1147, 331)}]
# byte_image = apply_blur_and_opacity(create_shadow_shape(curve_pts, straight_pts),  blur_intensity=0.6, opacity_intensity=0.5, brightness_intensity=-0.0)


# output_path = "outputdebug/shadowed_shape.png"
# with open(output_path, "wb") as f:
#     f.write(byte_image.read()) 
# print(f"Image saved to {output_path}")




