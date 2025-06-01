import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import os
from my_logging_script import log_to_json
current_file_name = "features/functionalites/backgroundoperations/finalwrapreflection.py"


def process_split_and_perspective_warp(output_stream, calculated_points, split_result):
    try:
    # Load the original image and shadow image
        output_stream.seek(0)
        # shadow.seek(0)
        original_image = Image.open(output_stream).convert("RGBA")
        # shadow_image = Image.open(shadow).convert("RGBA")

        # Get dimensions of the original image
        original_width, original_height = original_image.size

        # Increase the canvas size to double the height
        new_height = original_height * 2
        canvas = Image.new('RGBA', (original_width + 100, new_height), (255, 255, 255, 0))

        # canvas.alpha_composite(shadow_image, (0,0))
        # Paste the original image onto the canvas
        canvas.paste(original_image, (0, 0))

        # Load the left and right part images from split_result
        left_part_flipped_stream = flip_image_vertically(split_result["leftPart"])
        right_part_flipped_stream = flip_image_vertically(split_result["rightPart"])

        # Convert BytesIO streams back to PIL images
        left_part_image = Image.open(left_part_flipped_stream).convert("RGBA")
        right_part_image = Image.open(right_part_flipped_stream).convert("RGBA")

        # Convert PIL images to OpenCV format for transformation
        left_part_cv = cv2.cvtColor(np.array(left_part_image), cv2.COLOR_RGBA2BGRA)
        right_part_cv = cv2.cvtColor(np.array(right_part_image), cv2.COLOR_RGBA2BGRA)

        # Extract points from calculated_points
        ref_leftmost_bottom = calculated_points["ref_leftmost_bottom"]
        ref_leftmost_top = calculated_points["ref_leftmost_top"]
        ref_middle_bottom = calculated_points["ref_middle_bottom"]
        ref_middle_top = calculated_points["ref_middle_top"]
        ref_rightmost_bottom = calculated_points["ref_rightmost_bottom"]
        ref_rightmost_top = calculated_points["ref_rightmost_top"]

        # Warp and transform the left and right parts
        h, w = left_part_cv.shape[:2]
        left_src_points = np.float32([[0, h], [0, 0], [w, h], [w, 0]])
        left_dst_points = np.float32([ref_leftmost_bottom, ref_leftmost_top, ref_middle_bottom, ref_middle_top])
        left_homography_matrix, _ = cv2.findHomography(left_src_points, left_dst_points)
        left_warped = cv2.warpPerspective(left_part_cv, left_homography_matrix, (original_width + 100, new_height), flags=cv2.INTER_LINEAR)
        left_warped_pil = Image.fromarray(cv2.cvtColor(left_warped, cv2.COLOR_BGRA2RGBA))

        h, w = right_part_cv.shape[:2]
        right_src_points = np.float32([[0, h], [0, 0], [w, h], [w, 0]])
        right_dst_points = np.float32([ref_middle_bottom, ref_middle_top, ref_rightmost_bottom, ref_rightmost_top])
        right_homography_matrix, _ = cv2.findHomography(right_src_points, right_dst_points)
        right_warped = cv2.warpPerspective(right_part_cv, right_homography_matrix, (original_width + 100, new_height), flags=cv2.INTER_LINEAR)
        right_warped_pil = Image.fromarray(cv2.cvtColor(right_warped, cv2.COLOR_BGRA2RGBA))

        # Paste the warped parts onto the canvas
        #canvas.alpha_composite(left_warped_pil, (0, 0))
        canvas.alpha_composite(right_warped_pil, (0, 0))

        # Resize and position the shadow image to fit over the canvas
        # shadow_resized = shadow_image.resize((original_width + 100, new_height))
        # canvas.alpha_composite(shadow_resized, (0, 0))  # Place the shadow on top

        # Save the final image into the output_stream
        final_output_stream = BytesIO()
        canvas.save(final_output_stream, format="PNG")
        final_output_stream.seek(0)

        # Return the final processed stream
        return final_output_stream
    except FileNotFoundError as e:
        log_to_json(f"File not found error {e}", current_file_name)
        return f'Sorry the image is not processed for: File not found: {str(e)}'
    except ValueError as e:
        log_to_json(f"ValueError {e}", current_file_name)
        return f'Sorry the image is not processed for: Value error: {str(e)}'
    except Exception as e:
        log_to_json(f"Exceptional error {e}", current_file_name)
        return  f'Sorry : An unexpected error occurred: {str(e)}'



def create_canvas_with_perspective(points, original_stream, reverse_stream):
    print(f"modified coordinate is: {points}")
    try:
        # Load original image
        original_stream.seek(0)
        original_image = Image.open(original_stream).convert("RGBA")

        # Load reverse image
        reverse_stream.seek(0)
        reverse_image = Image.open(reverse_stream).convert("RGBA")

        # Create a transparent 1920x1440 canvas
        canvas_width, canvas_height = 1920, 1440
        canvas = Image.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 0))

        # Perform perspective warp on the reverse image
        src_points = np.float32([
            [0, 0],
            [0, reverse_image.height],
            [reverse_image.width, 0],
            [reverse_image.width, reverse_image.height]
        ])

        dst_points = np.float32([
            points["left_top"],
            points["left_bottom"],
            points["right_top"],
            points["right_bottom"]
        ])

        # Compute perspective transform matrix
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)

        # Warp the reverse image
        warped_image = cv2.warpPerspective(
            np.array(reverse_image),
            matrix,
            (canvas_width, canvas_height),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)  # Transparent border
        )

        # Convert warped image back to PIL format
        warped_pil_image = Image.fromarray(warped_image, mode="RGBA")

        # Paste the warped image onto the canvas
        canvas.paste(warped_pil_image, (0, 0), warped_pil_image)

        # Paste the original image on top of the canvas
        canvas.paste(original_image, (0, 0), original_image)

        # Save the final image to a BytesIO stream
        output_stream = BytesIO()
        canvas.save(output_stream, format="PNG")
        output_stream.seek(0)

        return output_stream
    except Exception as e:
        log_to_json(f"Exceptional error {e}", current_file_name)
        return  f'Sorry : An unexpected error occurred: {str(e)}'




def flip_image_vertically(image_stream):
    try:
    # Ensure the stream is at the beginning
        image_stream.seek(0)
        
        # Load the image with PIL
        image = Image.open(image_stream)

        # Flip the image vertically
        flipped_image = image.transpose(Image.FLIP_TOP_BOTTOM)

        # Save the flipped image to the outputdebug folder
        output_folder = "outputdebug"
        os.makedirs(output_folder, exist_ok=True)
        flipped_image_path = os.path.join(output_folder, "flipped_image.png")
        flipped_image.save(flipped_image_path)

        # Save the flipped image back to BytesIO
        flipped_stream = BytesIO()
        flipped_image.save(flipped_stream, format="PNG")
        flipped_stream.seek(0)  # Reset the stream position

        return flipped_stream
    except Exception as e:
        log_to_json(f"Error occurs for {e}", current_file_name)
        return None
