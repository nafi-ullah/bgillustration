import os
from rembg import remove
import requests
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from datetime import datetime
from io import BytesIO
from my_logging_script import log_to_json
from features.core.licenseplateoperations.licenseplateblurfunc import make_the_licenseplate_blur
from features.core.licenseplateoperations.licenseplateimagefunc import addLicensePlateImage
from features.functionalites.backgroundoperations.reflection import addReflection
from features.functionalites.backgroundoperations.addbackground.addwallinitiate import addBackgroundInitiate

current_file_name = "dynamicfuncs.py"

def is_feature_enabled(catalogue_feature_list, cat_id, feature_name):
    # print(f"cat id {cat_id} feature_name {feature_name}  cataluge list: {catalogue_feature_list}")
    try:
    # Convert cat_id to int for consistent comparison
        cat_id = int(cat_id)

        # Find the object with the matching catalogue_id
        catalogue = next((item for item in catalogue_feature_list if item['catalogue_id'] == cat_id), None)
        
        if not catalogue:
            # No matching catalogue found
            return False

        # Find if the feature name matches any cat_features "name"
        feature = next((f for f in catalogue['cat_features'] if f['name'] == feature_name), None)
        
        if not feature:
            # No matching feature name found
            return False

        # Return True if the feature's value is 'true', otherwise False
        return feature['value'].lower() == 'true'
    
    except FileNotFoundError as e:
        log_to_json(f"File not found error {e}", current_file_name)
        return f'Sorry the image is not processed for: File not found: {str(e)}'
    except ValueError as e:
        log_to_json(f"ValueError {e}", current_file_name)
        
        return f'Sorry the image is not processed for: Value error: {str(e)}'
    except Exception as e:
        log_to_json(f"Exceptional error {e}", current_file_name)
        return  f'Sorry the image is not processed for:  An unexpected error occurred: {str(e)}'
    
def get_user_setting(user_settings, user_id, setting_name):
    # Find the user with the given user_id
    try:
        user = next((user for user in user_settings if user.get("user_id") == user_id), None)
        
        if not user:
            return None  # Return None if user_id is not found
        
        # Iterate through the user's settings to find the setting with the given name
        for setting in user.get("settings", []):
            if setting.get("name") == setting_name:
                    values =  validate_setting(setting_name, setting.get("value"))
                    return values
        
        return None 
    
    except FileNotFoundError as e:
        log_to_json(f"File not found error {e}", current_file_name)
        return f'Sorry the image is not processed for: File not found: {str(e)}'
    except ValueError as e:
        log_to_json(f"ValueError {e}", current_file_name)
        
        return f'Sorry the image is not processed for: Value error: {str(e)}'
    except Exception as e:
        log_to_json(f"Exceptional error {e}", current_file_name)
        return  f'Sorry the image is not processed for:  An unexpected error occurred: {str(e)}'
    
def validate_setting(name: str, value: str) -> str:
    try:
        allowed_values = {
            "Default Crop": {"full_crop", "blur_crop"},
            "Background Type": {"photo_box", "basic", "universal"},
            "Default Background": None, 
            "Default Logo": None,        
            "License Plate Image": None, 
            "Interior Crop Type": {"full_crop", "blur_crop"},
            "Blur": {"small_blur", "medium_blur", "high_blur", "ultra_high_blur", "extreme_blur"},
            "Logo Position": {"top_left", "bottom_left", "top_right", "bottom_right", "center", "auto"}
        }

        # Check if the setting name exists
        if name not in allowed_values:
            log_to_json(f"Heyy mamaaa---- you have spell mistake in your setting. Invalid setting name '{name}'.", current_file_name)
            return None

        # Get the allowed values for the given setting name
        possible_values = allowed_values[name]
        print(f"the name : {name} value: {value}")
        # If possible_values is None, handle cases that allow any value
        if possible_values is None:
            # Check if the setting requires URL validation
            if name in { "Default Logo", "License Plate Image"}:
                if isinstance(value, str) and len(value.strip()) > 0:  # Ensure value is a non-empty string
                    print(f"Got the data in before url check for {name}")
                    validated_url = generate_and_check_url(value)
                    if validated_url is None:
                        log_to_json(f"URL validation failed for {name} with value: {value}", current_file_name)
                        return None
                    return validated_url  # Return the validated URL if it exists
                return None
            return value  # For other cases, return the value directly

        # Check if the value is valid for the setting
        if value not in possible_values:
            log_to_json(f"Value '{value}' does not match allowed values for '{name}': {value}", current_file_name)
            return None

        return value
    except Exception as e:
        print("Error uploading file to S3:", e)
        return None


def generate_and_check_url(base_path: str) -> str:

    prefix = "https://vroomview.obs.eu-de.otc.t-systems.com/"
    full_url = f"{prefix}{base_path.lstrip('/')}"  # Ensure no leading slash in base_path

    try:
        # Send a HEAD request to check if the image exists
        response = requests.head(full_url)

        if response.status_code == 200 or response.status_code == 201:
            return full_url
        else:
            print(f"Image does not exist at your logo / license plate path URL: {full_url}")
            return None
    except requests.RequestException as e:
        print(f"Error checking URL {full_url}: {e}")
        return None




def dynamic_licenseplate_blur():
    try:
        image_path = './outputs/cars/all/image01-original (1).jpg'
        model_path = './models/licenseplate/v495images/v495image_best.pt'
        output_path = './outputs/dumtest'
        #convert the image to bytesIO image
        with open(image_path, "rb") as image_file:
            image_bytesio = BytesIO(image_file.read())

        blurred_image = make_the_licenseplate_blur(model_path, image_bytesio)

        #save the image in outputpath
        if blurred_image:
            output_image_path = os.path.join(output_path, "blurred_image.jpg")
            with open(output_image_path, "wb") as output_file:
                output_file.write(blurred_image.getvalue())
            print(f"Blurred image saved at: {output_image_path}")
        else:
            print("Failed to process the image.")
    except Exception as e:
        log_to_json(f"Exceptional error {e}", current_file_name)
        return  f'Sorry : An unexpected error occurred: {str(e)}'

def getLicensePlateImage():
    try:
        licenseplat_path = 'outputs/fill/plate.png'

        with open(licenseplat_path, "rb") as image_file:
            license_image_bytesio = BytesIO(image_file.read())
        return license_image_bytesio
    except Exception as e:
        log_to_json(f"Exceptional error {e}", current_file_name)
        return  f'Sorry : An unexpected error occurred: {str(e)}'


def dynamic_licenseplate_image():
    try:
        image_path = './outputs/cars/all/image01-original (1).jpg'
        licenseplat_path = 'outputs/fill/plate.png'
        model_path = './models/licenseplate/v495images/v495image_best.pt'
        output_path = './outputs/dumtest'

        # Convert the image to bytesIO image
        with open(image_path, "rb") as image_file:
            image_bytesio = BytesIO(image_file.read())
        
        with open(licenseplat_path, "rb") as image_file:
            license_image_bytesio = BytesIO(image_file.read())

        # Call addLicensePlateImage with BytesIO objects
        blurred_image = addLicensePlateImage(model_path, image_bytesio, license_image_bytesio)
        

        ##save the image in outputpath
        if blurred_image:
            output_image_path = os.path.join(output_path, "license_processed_image.png")
            with open(output_image_path, "wb") as output_file:
                output_file.write(blurred_image.getvalue())
            print(f"license_processed saved at: {output_image_path}")
        else:
            print("Failed to process the image.")
    except Exception as e:
        log_to_json(f"Exceptional error {e}", current_file_name)
        return  f'Sorry : An unexpected error occurred: {str(e)}'


def dynamic_addReflection():
    try:
        image_cut_path = 'outputs/cars/processedcars/image_20241206_232031.png'
        image_transparent_path = 'outputs/cars/processedcars/image_20241206_232035.png'
        output_path = 'outputs/dumtest/outputdebug/finalimages'
        os.makedirs(output_path, exist_ok=True)  # Ensure the output directory exists

        # Convert the images to BytesIO format
        with open(image_cut_path, "rb") as image_file:
            image_cut_bytesio = BytesIO(image_file.read())

        with open(image_transparent_path, "rb") as image_file:
            transparent_bytesio = BytesIO(image_file.read())

        # Call the addReflection function
        blurred_image = addReflection(image_cut_bytesio, transparent_bytesio, angle_view='normal')

        # Save the image in output path
        if blurred_image:
            output_image_path = os.path.join(output_path, "blurred_image.png")
            with open(output_image_path, "wb") as output_file:
                output_file.write(blurred_image.getvalue())
            print(f"Blurred image saved at: {output_image_path}")
        else:
            print("Failed to process the image.")
    except Exception as e:
        log_to_json(f"Exceptional error {e}", current_file_name)
        return  f'Sorry : An unexpected error occurred: {str(e)}'






def save_image_with_timestamp(image_bytes_io, output_path, output_file_name):

    try:
        # Ensure the output path exists
        os.makedirs(output_path, exist_ok=True)

        # Append timestamp to the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name, file_extension = os.path.splitext(output_file_name)
        timestamped_file_name = f"{file_name}_{timestamp}{file_extension}"

        # Full path for the output image
        output_image_path = os.path.join(output_path, timestamped_file_name)

        # Save the image
        with open(output_image_path, "wb") as output_file:
            output_file.write(image_bytes_io.getvalue())

        print(f"Image saved at: {output_image_path}")
        return output_image_path

    except Exception as e:
        log_to_json(f"Failed to save the image. Error: {e}", current_file_name)
        return None
    
def save_image_temporary(image_bytes_io, output_path, output_file_name):

    try:
        # Ensure the output path exists
        os.makedirs(output_path, exist_ok=True)

        # Append timestamp to the filename

        file_name, file_extension = os.path.splitext(output_file_name)
        new_file_name = f"{file_name}{file_extension}"

        # Full path for the output image
        output_image_path = os.path.join(output_path, new_file_name)

        # Save the image
        with open(output_image_path, "wb") as output_file:
            output_file.write(image_bytes_io.getvalue())

        print(f"Image saved at: {output_image_path}")
        return output_image_path

    except Exception as e:
        log_to_json(f"Failed to save the image. Error: {e}", current_file_name)
        return None
    
def get_blur_intensity(blur_type: str) -> int:
    try:
        blur_intensity_map = {
            "small_blur": 5,
            "medium_blur": 15,
            "high_blur": 25,
            "ultra_high_blur": 35,
            "extreme_blur": 45,
        }

        # Get the blur intensity based on the blur type, or use medium_blur as default
        intensity = blur_intensity_map.get(blur_type, None)

        if intensity is None:
            print("None of the provided blur types matched. Defaulting to medium blur.")
            return blur_intensity_map["medium_blur"]

        return intensity
    except Exception as e:
        log_to_json(f"Exceptional error {e}", current_file_name)
        return  f'Sorry : An unexpected error occurred: {str(e)}'


# test_licenseplate_image()



# Paths
