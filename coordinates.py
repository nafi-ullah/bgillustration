

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

