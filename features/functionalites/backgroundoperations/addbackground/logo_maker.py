from PIL import Image
from io import BytesIO
from my_logging_script import log_to_json
current_file_name = "logo_maker.py"

def logo_maker_initiate(logo_bytes_io):
    # Open the logo image from the BytesIO object
    try:
        logo_image = Image.open(logo_bytes_io)

        # Get the size of the logo image
        logo_width, logo_height = logo_image.size

        # Create a new canvas with specified area dimensions and transparent background
        area_width = logo_width + 60
        area_height = logo_height + 120
        canvas = Image.new('RGBA', (area_width, area_height), (0, 0, 0, 0))  # Transparent background

        # Place the logo on the canvas at coordinates (30, 40)
        canvas.paste(logo_image, (30, 40), logo_image.convert('RGBA'))  # Assuming logo has transparency

        # Save the image to a BytesIO object
        byte_io = BytesIO()
        canvas.save(byte_io, format='PNG')
        byte_io.seek(0)

        return byte_io
    except Exception as e:
        log_to_json(f"logo_maker_initiate-- Error for: {e}", current_file_name)
        return None

# def combine_logo_with_bg(bg_image: BytesIO, position: str, logo_image: BytesIO) -> BytesIO:
#     # Open the background image and logo image from BytesIO
#     bg = Image.open(bg_image)
#     logo = Image.open(logo_image)

#     # Resize the logo image to width 800px, maintain aspect ratio
#     logo_width = 800
#     logo_height = int((logo.height / logo.width) * logo_width)
#     logo_resized = logo.resize((logo_width, logo_height))

#     # Get background image dimensions
#     bg_width, bg_height = bg.size

#     # Calculate the position of the logo based on the position value
#     if position == "left":
#         logo_position = (100, 100)
#     elif position == "center":
#         logo_position = ((bg_width - logo_width) // 2, 100)
#     elif position == "right":
#         logo_position = (bg_width - logo_width - 100, 100)
#     else:
#         raise ValueError("Invalid position. Use 'left', 'center', or 'right'.")

#     # Paste the logo onto the background image at the calculated position
#     bg.paste(logo_resized, logo_position, logo_resized.convert('RGBA'))

#     # Save the final image to a BytesIO object
#     output = BytesIO()
#     bg.save(output, format="PNG")
#     output.seek(0)

#     return output



def combine_logo_with_bg(bg_image: BytesIO, position: str = None, logo_image: BytesIO = None) -> BytesIO:
    # Open the background image from BytesIO
    try:
        bg = Image.open(bg_image)
        bg = bg.transpose(Image.FLIP_LEFT_RIGHT)

        # If logo_image is provided, add the logo to the background
        if logo_image:
            logo = Image.open(logo_image)

            # Resize the logo image to width 800px, maintain aspect ratio
            logo_width, logo_height = get_logo_resize_dimensions(logo)
            logo_resized = logo.resize((logo_width, logo_height))

            # Flip the logo horizontally
            logo_resized = logo_resized.transpose(Image.FLIP_LEFT_RIGHT)

            # Get background image dimensions
            bg_width, bg_height = bg.size
            logo_top_height = 80 # previous 100
            # Calculate the position of the logo based on the position value
            if position is None:
                position = "top_left" 

            if position == "top_left" or position == "auto":
                logo_position = (bg_width - logo_width - 100, logo_top_height)  
    
            elif position == "center":
                logo_position = (bg_width // 2 - logo_width // 2, logo_top_height) 
            elif position == "top_right":
                logo_position = (100, logo_top_height)
            elif position == "bottom_right":
                logo_position = (100, 700)  # Center the logo
            elif position == "bottom_left":
                logo_position = (bg_width - logo_width - 100, 700)
            else:
                logo_position = (100, 200)
                # raise ValueError("Invalid position. Use 'left', 'center', or 'right'.")

            # Paste the logo onto the background image at the calculated position
            bg.paste(logo_resized, logo_position, logo_resized.convert('RGBA'))

        # Save the final image to a BytesIO object
        output = BytesIO()
        bg.save(output, format="PNG")
        output.seek(0)

        return output
    except Exception as e:
            log_to_json(f"combine_logo_with_bg-- Error for: {e}", current_file_name)
            return bg_image

def basic_combine_logo_with_bg(bg_image: BytesIO, position: str = None, logo_image: BytesIO = None) -> BytesIO:
    # Open the background image from BytesIO
    try:
        bg = Image.open(bg_image)
        bg = bg.transpose(Image.FLIP_LEFT_RIGHT)
        bg = bg.rotate(90, expand=True)

        # If logo_image is provided, add the logo to the background
        if logo_image:
            logo = Image.open(logo_image)

            # Resize the logo image to width 800px, maintain aspect ratio
            logo_width = 350
            logo_height = int((logo.height / logo.width) * logo_width) 
            logo_resized = logo.resize((logo_width, logo_height))

            # Flip the logo horizontally
            logo_resized = logo_resized.transpose(Image.FLIP_LEFT_RIGHT)
            logo_resized = logo_resized.rotate(90, expand=True)

            # Get background image dimensions
            bg_width, bg_height = bg.size
            top_height = 800 # previous 100
            # Calculate the position of the logo based on the position value
            if position is None:
                position = "top_right" 

            if position == "top_right" or position == "auto":
                logo_position = ( top_height, bg_width  - 100)
    
            elif position == "center":
                logo_position = ( top_height, bg_width // 2 + 100) 
            elif position == "top_left":
                logo_position = ( top_height, 200)
            elif position == "bottom_right":
                logo_position = (bg_width - logo_width - 100, bg_width - 100)  # Center the logo
            elif position == "bottom_left":
                logo_position = (bg_width - logo_width - 100, 200)
            else:
                logo_position = (100, 200)
                # raise ValueError("Invalid position. Use 'left', 'center', or 'right'.")

            # Paste the logo onto the background image at the calculated position
            bg.paste(logo_resized, logo_position, logo_resized.convert('RGBA'))

        # Save the final image to a BytesIO object
        output = BytesIO()
        bg.save(output, format="PNG")
        output.seek(0)

        return output
    except Exception as e:
            log_to_json(f"basic_combine_logo_with_bg-- Error for: {e}", current_file_name)
            return bg_image


def get_logo_resize_dimensions(logo_img: Image.Image,
                               extra_length: int = 200,
                               max_width: int = 500,
                               max_height: int = 400
                              ) -> tuple[int, int]:

    orig_w, orig_h = logo_img.size
    ratio_range = max_width / max_height
    actual_ratio = orig_w / orig_h       # width/height
    inv_ratio    = orig_h / orig_w       # height/width

    if actual_ratio < ratio_range:
        # logo is ‘narrower’ than the target rectangle → height controls
        new_width  = int(max_height * actual_ratio)
        new_height = max_height + int((200/500)* new_width) # extra_length
    else:
        # logo is ‘wider’ than (or equal to) the target rectangle → width controls
        new_width  = max_width
        new_height = int(max_width * inv_ratio) +  extra_length

    print(f"orig_w {orig_w} orig_h {orig_h} new_width {new_width} new_height {new_height} ratio_range {ratio_range} actual_ratio {actual_ratio} inv_ratio {inv_ratio}")

    return new_width, new_height

def combine_logo_with_bg_Reverse(bg_image: BytesIO, position: str = None, logo_image: BytesIO = None) -> BytesIO:
    # Open the background image and logo image from BytesIO
    try:
        bg = Image.open(bg_image)
        bg = bg.rotate(270, expand=True)
        bg = bg.transpose(Image.FLIP_LEFT_RIGHT)

        if logo_image:
            logo = Image.open(logo_image)

            w, h = get_logo_resize_dimensions(logo)
            logo_resized = logo.resize((w, h))

            # Rotate the logo 90 degrees to the right
            logo_resized = logo_resized.rotate(270, expand=True)

            # Flip the logo horizontally
            logo_resized = logo_resized.transpose(Image.FLIP_LEFT_RIGHT)
        
            logo_top_height = 80 # previouisly 100
            # Get background image dimensions
            bg_width, bg_height = bg.size
            print(f"bg {bg_width} {bg_height}")
            # Calculate the position of the logo based on the position value
            if position is None:
                position = "top_right" 
                
            if position == "top_right" or position == "auto":
                logo_position = (logo_top_height, bg_width  - 100) 
            elif position == "center":
                logo_position = ( logo_top_height, bg_width // 2)
            elif position == "top_left":
                logo_position = (logo_top_height, 100)
            elif position == "bottom_left":
                logo_position = (700, 100)
            elif position == "bottom_right":
                logo_position = (700, bg_width  - 100)
            else:
                logo_position = (200, 100)

            # Paste the logo onto the background image at the calculated position
            bg.paste(logo_resized, logo_position, logo_resized.convert('RGBA'))

        # Save the final image to a BytesIO object
        output = BytesIO()
        bg.save(output, format="PNG")
        output.seek(0)

        return output
    except Exception as e:
            log_to_json(f"combine_logo_with_bg_Reverse-- Error for: {e}", current_file_name)
            return bg_image


# logo_maker_initiate()
