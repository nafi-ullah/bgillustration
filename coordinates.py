from features.functionalites.backgroundoperations.carbasicoperations import add_padding_to_image
from features.functionalites.imageoperations.basic import resize_image_bytesio
from PIL import Image, ImageTk
from io import BytesIO
from dynamicfuncs import is_feature_enabled, get_user_setting, save_image_with_timestamp


def getCoordinates(angle, car_image):
    if angle in ["1", "3", "5", "6"]:  # straight angles
        normalCoordinates = {
            "floor_left_top": (-2702, 1366),
            "floor_left_bottom": (800, 2500),
            "floor_right_bottom": (3000, 1037),
            
            "rwall_top_left": (1253, 130),
            "rwall_right_bottom": (1920, 873),
            "rwall_top_right": (1920, 29),
            "lwall_left_top": (0, 0),
            "lwall_left_bottom": (0, 962),
            "canvas_middle_ref": (1253, 773),
            "ceiling_top": (1920, -700)
        }

        return normalCoordinates
    elif angle == "2" or angle == "7":  # normal
        normalCoordinates = {
            "floor_left_top": (-2702, 1366),
            "floor_left_bottom": (800, 2500),
            "floor_right_bottom": (3000, 1037),
            
            "rwall_top_left": (1253, 130),
            "rwall_right_bottom": (1920, 873),
            "rwall_top_right": (1920, 29),
            "lwall_left_top": (0, 0),
            "lwall_left_bottom": (0, 962),
            "canvas_middle_ref": (1253, 773),
            "ceiling_top": (1920, -700)
        }
        return normalCoordinates
    elif angle == "4" or angle == "8": # reverse
        normalCoordinates = {
            "floor_left_top": (-1283, 1030),
            "floor_left_bottom": (615, 3120),
            "floor_right_bottom": (3220, 1139),
            
            "rwall_top_left": (615, 100),
            "rwall_right_bottom": (1920, 942),
            "rwall_top_right": (1920, 0),
            "lwall_left_top": (0, 6),
            "lwall_left_bottom": (0, 837),
            "canvas_middle_ref": (615, 744),
            "ceiling_top": (0, -700)
        }

        return normalCoordinates
    elif angle == "9" or angle == "12":  # top view normal
        normalCoordinates = {
            "floor_left_top": (-2702, 1366),
            "floor_left_bottom": (800, 2500),
            "floor_right_bottom": (3000, 1037),
            
            "rwall_top_left": (1253, 130),
            "rwall_right_bottom": (1920, 873),
            "rwall_top_right": (1920, 29),
            "lwall_left_top": (0, 0),
            "lwall_left_bottom": (0, 962),
            "canvas_middle_ref": (1253, 773),
            "ceiling_top": (1920, -700)
        }
        return normalCoordinates
    elif angle == "10" or angle == "11":  # top view reverse
        normalCoordinates = {
            "floor_left_top": (-1283, 1030),
            "floor_left_bottom": (615, 3120),
            "floor_right_bottom": (3220, 1139),
            
            "rwall_top_left": (615, 100),
            "rwall_right_bottom": (1920, 942),
            "rwall_top_right": (1920, 0),
            "lwall_left_top": (0, 6),
            "lwall_left_bottom": (0, 837),
            "canvas_middle_ref": (615, 744),
            "ceiling_top": (0, -700)
        }
        return normalCoordinates
    elif angle == "13" or angle == "16":  # bottom normal
        normalCoordinates = {
            "floor_left_top": (-2702, 1366),
            "floor_left_bottom": (800, 2500),
            "floor_right_bottom": (3000, 1037),
            
            "rwall_top_left": (1253, 130),
            "rwall_right_bottom": (1920, 873),
            "rwall_top_right": (1920, 29),
            "lwall_left_top": (0, 0),
            "lwall_left_bottom": (0, 962),
            "canvas_middle_ref": (1253, 773),
            "ceiling_top": (1920, -700)
        }
        return normalCoordinates
    elif angle == "14" or angle == "15": # bottom reverse
        normalCoordinates = {
            "floor_left_top": (-1283, 1030),
            "floor_left_bottom": (615, 3120),
            "floor_right_bottom": (3220, 1139),
            "rwall_top_left": (615, 100),
            "rwall_right_bottom": (1920, 942),
            "rwall_top_right": (1920, 0),
            "lwall_left_top": (0, 6),
            "lwall_left_bottom": (0, 837),
            "canvas_middle_ref": (615, 744),
            "ceiling_top": (0, -700)
        }
        return normalCoordinates
    

def getCarParameters(angle, car_image):
    parameters = {
        "position": (0,400),
        "lrpadding": (100,100),
        "tbpadding": (0,0)
    }

    return parameters


def padding_car(coords, detected_vehicle):
    try:
        detected_vehicle.seek(0)
        image_file_json = resize_image_bytesio(detected_vehicle, "width", 1920 )
        img_byte_arr = image_file_json['retruned_image']
        img_byte_arr.seek(0)

        left, right, top, bottom = coords["lrpadding"][0],  coords["lrpadding"][1], coords["tbpadding"][0], coords["tbpadding"][1]
        detected_vehicle_padding = add_padding_to_image(img_byte_arr, top, left, right, bottom)

        return detected_vehicle_padding

    except Exception as e:
        print(f"Error occars in padding car {e}")


def combine_car_with_bg(bg_image, car_image, car_position):
    try:
        # Open the background and car images from BytesIO and ensure RGBA mode
        bg = Image.open(bg_image).convert("RGBA")
        car = Image.open(car_image).convert("RGBA")
        # Save the car image to outputs/car folder using typical method
        car_image.seek(0)
        car_img = Image.open(car_image)
        car_img.save('outputs/car/car.png')
        car_image.seek(0)

        canvas_width, canvas_height = 1920, 1440
        canvas = Image.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 0))  # Fully transparent background

        # Paste the background image at (0, 0) on the canvas
        canvas.paste(bg, (0, 0))

        # Paste the car image at car_position using its alpha channel as mask for transparency
        canvas.paste(car, car_position, car)

        # Save the final image to a BytesIO object
        output = BytesIO()
        canvas.save(output, format="PNG")
        output.seek(0)

        save_image_with_timestamp(output, 'outputs/car', 'before.png')

        return output
    except Exception as e:
        print(f"Error occurss on combine car with bg {e}")
        return BytesIO()

