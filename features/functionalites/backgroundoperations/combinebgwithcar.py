from PIL import Image
from io import BytesIO
from my_logging_script import log_to_json
current_file_name = "features/functionalites/backgroundoperations/combinebgwithcar.py"
from config.configfunc import get_current_process_values
car_from_bottom_height = 0


def combine_car_with_bg(bg_image: BytesIO, car_image: BytesIO, car_height: int) -> BytesIO:
    try:
        # Open the background and car images from BytesIO
        bg = Image.open(bg_image)
        car = Image.open(car_image)

        current_config = get_current_process_values()
        angle_ids = current_config.get("angle_id", 2)

        # Create a canvas with size 1920x1440
        canvas_width, canvas_height = 1920, 1440
        canvas = Image.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 255))  # Transparent background
        print(f"Original car dimensions: width={car.width}, height={car.height}, requested height={car_height}")

        # Paste the background image at (0, 0) on the canvas
        canvas.paste(bg, (0, 0))

        threshold_height = 1250
        threshold_width = 1920
        move_right = 0
        # Resize the car if its width exceeds the threshold
        if car.width > threshold_width:
            aspect_ratio = car.height / car.width  # Maintain aspect ratio
            new_car_width = threshold_width
            new_car_height = int(new_car_width * aspect_ratio)
            car = car.resize((new_car_width, new_car_height), Image.LANCZOS)
            print(f"Car resized based on width: width={new_car_width}, height={new_car_height}")
        else:
            new_car_width = car.width
            new_car_height = car.height

        # Check if the car height exceeds the threshold after the first resize
        if new_car_height > threshold_height:
            aspect_ratio = car.width / car.height  # Maintain aspect ratio
            new_car_height = threshold_height
            new_car_width = int(new_car_height * aspect_ratio)
            car = car.resize((new_car_width, new_car_height), Image.LANCZOS)
            move_right = (canvas_width - new_car_width) / 2
            print(f"Car resized based on height: width={new_car_width}, height={new_car_height}")

        # Calculate the car's vertical position
        car_y = canvas_height - new_car_height

        # Calculate horizontal position adjustment
        
        car_x = move_right

        # Paste the car image onto the canvas
        car_move_down = 120  # already have 200 px bottom padding. so now padding is 200-120 = 80px
        car_position = (int(car_x), int(car_y + car_move_down))
        log_to_json(f"Car final position: x={car_x}, y={car_y}, width={new_car_width}, height={new_car_height}", current_file_name)
        canvas.paste(car, car_position, car.convert('RGBA'))

        # Flatten the image
        flattened_image = Image.new("RGB", canvas.size, (0, 0, 0))
        flattened_image.paste(canvas, mask=canvas.split()[3])

        # Save the final image to a BytesIO object
        output = BytesIO()
        flattened_image.save(output, format="PNG")
        output.seek(0)

        return output
    except Exception as e:
        log_to_json(f"Error occurs for {e}", current_file_name)
        return BytesIO()
