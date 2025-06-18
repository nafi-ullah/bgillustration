from features.functionalites.backgroundoperations.carbasicoperations import add_padding_to_image
from features.functionalites.imageoperations.basic import resize_image_bytesio
from PIL import Image, ImageTk
from io import BytesIO
from dynamicfuncs import is_feature_enabled, get_user_setting, save_image_with_timestamp


def getBasicCoordinates(angle, car_image):
    if angle in ["1", "3", "5", "6"]:  # straight angles
        normalCoordinates = {
            "floor_left_top": (0, 720),
            "floor_left_bottom": (800, 1765),
            "floor_right_bottom": (-665, 1765),
            "floor_right_top": (1920, 720),
            "wall_right_top": (1920, -720),
            "wall_left_top": (0, -720),
        }

        return normalCoordinates
    elif angle == "2" or angle == "7":  # normal
        normalCoordinates = {
            "floor_left_top": (0, 720),
            "floor_left_bottom": (-665, 1765),
            "floor_right_bottom": (2585, 1765),
            "floor_right_top": (1920, 720),
            "wall_right_top": (1920, -720),
            "wall_left_top": (0, -720),
        }

        return normalCoordinates
    elif angle == "4" or angle == "8": # reverse
        normalCoordinates = {
            "floor_left_top": (-2702, 1366),
            "floor_left_bottom": (800, 2500),
            "floor_right_bottom": (3000, 1037),
            "floor_right_top": (3000, 1037),
            "wall_right_top": (3000, 1037),
            "wall_left_top": (3000, 1037),
        }


        return normalCoordinates
    elif angle == "9" or angle == "12":  # top view normal
        normalCoordinates = {
            "floor_left_top": (0, 1020),
            "floor_left_bottom": (-665, 2765),
            "floor_right_bottom": (2585, 1765),
            "floor_right_top": (1920, 320),
            "wall_right_top": (1920, -720),
            "wall_left_top": (0, -720)
        }

        return normalCoordinates
    elif angle == "10" or angle == "11":  # top view reverse
        normalCoordinates = {
            "floor_left_top": (-2702, 1366),
            "floor_left_bottom": (800, 2500),
            "floor_right_bottom": (3000, 1037),
            "floor_right_top": (3000, 1037),
            "wall_right_top": (3000, 1037),
            "wall_left_top": (3000, 1037),
        }

        return normalCoordinates
    elif angle == "13" or angle == "16":  # bottom normal
        normalCoordinates = {
            "floor_left_top": (-2702, 1366),
            "floor_left_bottom": (800, 2500),
            "floor_right_bottom": (3000, 1037),
            "floor_right_top": (3000, 1037),
            "wall_right_top": (3000, 1037),
            "wall_left_top": (3000, 1037),
        }

        return normalCoordinates
    elif angle == "14" or angle == "15": # bottom reverse
        normalCoordinates = {
            "floor_left_top": (-2702, 1366),
            "floor_left_bottom": (800, 2500),
            "floor_right_bottom": (3000, 1037),
            "floor_right_top": (3000, 1037),
            "wall_right_top": (3000, 1037),
            "wall_left_top": (3000, 1037),
        }

        return normalCoordinates
    

def getCarParameters(angle, car_image):
    parameters = {
        "position": (0,400),
        "lrpadding": (100,100),
        "tbpadding": (0,0)
    }

    return parameters



