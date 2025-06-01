import math

def angle_between(p1, p2):
    """Returns angle in degrees between two points with respect to horizontal line"""
    return math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))

def adjust_point(p_fixed, angle_target, length):
    """Adjusts the second point to match the target angle and original length"""
    rad = math.radians(angle_target)
    x_new = p_fixed[0] + length * math.cos(rad)
    y_new = p_fixed[1] + length * math.sin(rad)
    return (x_new, y_new)

def point_with_larger_y(origin, angle_deg, length):
    """
    Given a starting point, angle, and length, this returns the point
    (from the two possible directions) with the larger Y (i.e., lower visually).
    """
    angle_rad = math.radians(angle_deg)
    
    # First point in given angle direction
    x1 = origin[0] + length * math.cos(angle_rad)
    y1 = origin[1] + length * math.sin(angle_rad)
    point1 = (x1, y1)

    # Second point in opposite direction
    angle_opposite = angle_rad + math.pi
    x2 = origin[0] + length * math.cos(angle_opposite)
    y2 = origin[1] + length * math.sin(angle_opposite)
    point2 = (x2, y2)

    # Return the point with larger Y
    return point1 if y1 > y2 else point2

def almost_equal(a, b, tolerance=1):
    return abs(a - b) <= tolerance

def straight_licenseplate(a, b):
    return 80 <= a <= 100 or 80 <= b <= 100
def make_parallelogram(top_left, top_right, bottom_left, bottom_right):
    # Step 1: Compute initial angles
    angle_left = angle_between(top_left, bottom_left)
    angle_right = angle_between(top_right, bottom_right)
    print(f"angle left {angle_left}, angle right {angle_right}")

    is_straight_licensplate = straight_licenseplate(angle_left, angle_right)
    print(f"straight license plate {is_straight_licensplate}")
    # Step 2: Adjust vertical lines if not parallel
    if not almost_equal(angle_left, angle_right) or is_straight_licensplate == False:
        # Decide which one to fix based on smaller abs(angle)
        if abs(angle_left) > abs(angle_right):
            # Fix left side, adjust right bottom
            length = math.dist(top_right, bottom_right)
            bottom_right = adjust_point(top_right, angle_left, length)
            angle_right = angle_left
        else:
            # Fix right side, adjust left bottom
            length = math.dist(top_left, bottom_left)
            bottom_left = adjust_point(top_left, angle_right, length)
            angle_left = angle_right

    
    if is_straight_licensplate:
        left_length = math.dist(top_left, bottom_left)
        right_length = math.dist(top_right, bottom_right)
        straight_angle = 90

        print(f"left length {left_length}, right length {right_length}")
        if left_length < 70 and right_length < 70:
            left_length = 70
            right_length = 70
        else:
            if abs(left_length) < abs(right_length):
                right_length = left_length
            else:
                left_length = right_length
            

        bottom_left = point_with_larger_y(top_left, straight_angle, right_length)
        bottom_right = point_with_larger_y(top_right, straight_angle, left_length)

        print(f"bottom left {bottom_left}, bottom right {bottom_right}")
        return {
            "top_left": top_left,
            "top_right": top_right,
            "bottom_left": bottom_left,
            "bottom_right": bottom_right
        }

   


    # Step 3: Compute horizontal angles
    angle_top = angle_between(top_left, top_right)
    angle_bottom = angle_between(bottom_left, bottom_right)

    # Step 4: Adjust horizontal lines if not parallel
    if not almost_equal(angle_top, angle_bottom) or not is_straight_licensplate:

        left_length = math.dist(top_left, bottom_left)
        right_length = math.dist(top_right, bottom_right)

        print(f"left length {left_length}, right length {right_length}")

        if abs(left_length) > abs(right_length):
            bottom_left = point_with_larger_y(top_left, angle_right, right_length)
        else:
            bottom_right = point_with_larger_y(top_right, angle_left, left_length)

        print(f"bottom left {bottom_left}, bottom right {bottom_right}")
        
        angle_bottom = angle_top

    return {
        "top_left": top_left,
        "top_right": top_right,
        "bottom_left": bottom_left,
        "bottom_right": bottom_right,
        "angles": {
            "left": angle_left,
            "right": angle_right,
            "top": angle_top,
            "bottom": angle_bottom
        }
    }
