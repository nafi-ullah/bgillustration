import cv2
import numpy as np
from io import BytesIO
from PIL import Image
from my_logging_script import log_to_json
current_file_name = "features/exterior_process/reflectionwrap.py"

def transform_image_with_tps(source_points, destination_points, image_bytesio):
    try:
        # Create TPS transformer
        tps = cv2.createThinPlateSplineShapeTransformer()

        # Prepare points in the required format
        src_points_cv = source_points.reshape(1, -1, 2)  # Shape: (1, N, 2)
        dst_points_cv = destination_points.reshape(1, -1, 2)  # Shape: (1, N, 2)

        # Create matches for TPS transformation
        matches = [cv2.DMatch(i, i, 0) for i in range(len(source_points))]
        tps.estimateTransformation(dst_points_cv, src_points_cv, matches)

        # Load the source image from BytesIO
        image_bytesio.seek(0)
        np_image = np.asarray(bytearray(image_bytesio.read()), dtype=np.uint8)
        src_image = cv2.imdecode(np_image, cv2.IMREAD_UNCHANGED)  # Preserve alpha channel

        if src_image.shape[2] == 4:  # Check if the image has an alpha channel
            h, w = src_image.shape[:2]
            dst_image = np.zeros_like(src_image)  # Initialize the output image with transparency

            # Separate the alpha channel
            bgr = src_image[:, :, :3]  # RGB channels
            alpha = src_image[:, :, 3]  # Alpha channel

            # Warp the RGB channels
            warped_bgr = np.zeros_like(bgr)
            tps.warpImage(bgr, warped_bgr)

            # Warp the alpha channel
            warped_alpha = np.zeros_like(alpha)
            tps.warpImage(alpha, warped_alpha)

            # Combine the warped RGB and alpha channels
            dst_image[:, :, :3] = warped_bgr
            dst_image[:, :, 3] = warped_alpha
        else:
            # For non-transparent images
            h, w = src_image.shape[:2]
            dst_image = np.zeros_like(src_image)
            tps.warpImage(src_image, dst_image)

        # Save the output image to BytesIO
        output_bytesio = BytesIO()
        _, buffer = cv2.imencode('.png', dst_image)
        output_bytesio.write(buffer)
        output_bytesio.seek(0)

        return output_bytesio
    except Exception as e:
        log_to_json(f"transform_image_with_tps: Exceptional error {e}", current_file_name)
        return None


def overlay_car_on_background(background_bytesio, car_bytesio):
    try:
        # Open background and car images
        background = Image.open(background_bytesio)
        car = Image.open(car_bytesio)

        # Create a canvas the same size as the background
        canvas = Image.new("RGBA", background.size, (255, 255, 255, 0))

        # Paste the background onto the canvas
        canvas.paste(background, (0, 0))

        # Paste the car image at (0, 0)
        canvas.paste(car, (0, 0), mask=car if car.mode == 'RGBA' else None)

        # Save the resulting image to BytesIO
        output_bytesio = BytesIO()
        canvas.save(output_bytesio, format="PNG")
        output_bytesio.seek(0)

        return output_bytesio
    except Exception as e:
        log_to_json(f"overlay_car_on_background: Exceptional error {e}", current_file_name)
        return None

# Example usage
# src_points = np.array([[85, 1318], [187, 1583],[437, 1150], [437, 1350], [868, 1090], [830, 1814], [1777, 1318], [1777, 1663]], dtype=np.float32)
# dst_points = np.array([[85, 1130], [187, 1430],[437, 1200], [437, 1400], [868, 1090], [900, 1714], [1777, 850], [1777, 1182]], dtype=np.float32)
# with open('source_image.png', 'rb') as f:
#     image_io = BytesIO(f.read())
# transformed_image = transform_image_with_tps(src_points, dst_points, image_io)
# with open('output_image.png', 'wb') as f:
#     f.write(transformed_image.getvalue())
