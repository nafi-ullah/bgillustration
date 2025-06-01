from flask import Flask,g, request, jsonify, send_from_directory
import requests
import json
import os
from queue import Queue
from threading import Thread
import time
from PIL import Image
from rembg import remove
from io import BytesIO
from datetime import datetime
from flask_cors import CORS 
from features.interiorprocess.interiorbg import  remove_background,  save_image_to_s3,  remove_background_premium
from my_logging_script import log_to_json
from vehicle_save import  detect_and_return_ByteIoImage
from features.laravelconnection.processdoneapi import notify_error_status, notify_success_status
from features.functionalites.addBackground import add_gray_background_solo, dynamic_interior_images
from features.laravelconnection.uploadImageBucket import upload_to_s3
from dynamicfuncs import is_feature_enabled, get_user_setting, save_image_with_timestamp, save_image_temporary ,getLicensePlateImage, get_blur_intensity
from features.core.licenseplateoperations.licenseplateblurfunc import make_the_licenseplate_blur
from features.core.licenseplateoperations.licenseplateimagefunc import addLicensePlateImage
from features.core.licenseplateoperations.licenseplateimageupdated import addLicensePlateImageNewAlgo
from features.functionalites.backgroundoperations.reflection import addReflection
from features.functionalites.backgroundoperations.addbackground.addwallinitiate import addBackgroundInitiate
from backgroundconfs import get_dynamic_wall_coordinates, get_universal_wall_images,  get_dynamic_wall_reverse_coordinates,  get_dynamic_basic_wall_images,get_dynamic_wall_images, get_logo_image, get_wall_coordinates_recverse_angle, get_basic_wall_images, get_dynamic_image
from features.functionalites.backgroundoperations.combinebgwithcar import combine_car_with_bg
from config.configfunc import read_config, write_config, read_config_bool, write_api_key, read_api_key
from config.manageimgprocess import image_process_req_manupulate
from features.functionalites.backgroundoperations.basicbg.basicbgprocess import add_basic_bg
from features.functionalites.imageoperations.basic import resize_image_bytesio, get_image_dimensions, paste_foreground_on_background_bytesio, blur_image_bytesio, resize_1920_image
from features.core.polishing import polish_car
from features.core.rimpolishing import rimpolishingfunc
import cv2
from features.functionalites.backgroundoperations.carbasicoperations import add_padding_to_image, crop_image_by_coordinates
from features.detections.fullcarcoordinates import create_proxy_background_with_car

current_file_name = "features/exterior_process/exteriorprocess.py"
licensepalate_model_path = './models/licenseplate/augmented/best.pt'


def frontleft(image_url, userid, global_user_setting_data, catalogue_feature_list):
    opencv_version = cv2.__version__
    log_to_json(f"---------------version check-----------------opencv: {opencv_version}", current_file_name)

    print("------------got the file in front left function----------")
    # shared_array_data = g.get("shared_array_data", {})

    # catalogue_feature_list = shared_array_data.get("catalogue_feature_list", [])
    # global_user_setting_data = shared_array_data.get("global_user_setting_data", [])


    set_crop_val = get_user_setting(global_user_setting_data, userid, "Default Crop")
    set_bg_type = get_user_setting(global_user_setting_data, userid, "Background Type")
    set_def_background = get_user_setting(global_user_setting_data, userid, "Default Background")
    set_def_logo = get_user_setting(global_user_setting_data, userid, "Default Logo")
    set_license_plate_image = get_user_setting(global_user_setting_data, userid, "License Plate Image")
    set_blur_intesity = get_blur_intensity(get_user_setting(global_user_setting_data, userid, "Blur"))
    set_logo_position = get_user_setting(global_user_setting_data, userid, "Logo Position")

    log_set_value = f"Setting for userid {userid}: Default Crop : {set_crop_val} Background Type: {set_bg_type} Default Background: {set_def_background} default logo: {set_def_logo} License Plate Image: {set_license_plate_image}  Blur intesity:  {set_blur_intesity} Logo Position {set_logo_position}"
    log_to_json(log_set_value, current_file_name)

    if set_def_logo is not None and set_logo_position is None:
        set_logo_position = "auto"
        log_to_json(f"logo poistion set to {set_logo_position}. because of wrong setting", current_file_name)

    if set_crop_val is None:
        set_crop_val = "full_crop"
        log_to_json(f"logo poistion set to {set_crop_val}.", current_file_name)

    try:
        response = requests.get(image_url)
        if response.status_code != 200:
            # log_to_json(f"Failed to download image from {image_url}", current_file_name)
            notify_error_status(catalogue_id, angle_id, filename, f"Failed to get the selected image. Please capture again.", current_file_name)
            return None
        from config.configfunc import get_current_process_values
        current_config = get_current_process_values()
        db_angle_id = current_config.get("angle_id", 0)
        # Extract catalogue_id and angle_id from the URL
        url_parts = image_url.split('/')
        catalogue_id = url_parts[-3]  # Example: 1001
        angle_id = url_parts[-2]      # Example: 7
        filename = url_parts[-1]      # Example: image07-original.jpg
        file_name, file_extension = os.path.splitext(filename)
        if db_angle_id in [18, 19]:
            angle_id = "2"

        if set_crop_val == 'full_crop' and (set_bg_type is None or set_def_background is None):
            notify_error_status(catalogue_id, angle_id, filename, "Default Crop or Background type or Default Background is not selected properly.", current_file_name)
            return None
            
        
        #image processing start by get the image

        image_file = BytesIO(response.content)


        # detect and crop image

       #-----------------remove bg-------------      
        
        rem_back = remove_background_premium(image_file)
  

        
        if rem_back["status"] == "failed":
            # Notify failure
            notify_response = notify_error_status(catalogue_id, angle_id, filename, f"There is a technical difficulty, please try again later.", current_file_name) 
            # foreground and credit er issue alada korte hobe
            return None

        # Background removal was successful; proceed with adding gray background and upload
        transparent_image = rem_back["image"]
        
        img_byte_arr = BytesIO()
        # saving = save_image_with_timestamp(transparent_image, 'outputs/dumtest/outputdebug/finalimages', 'before.png')
        transparent_image.save(img_byte_arr, format='PNG')  # Save the image in PNG format to the BytesIO object
        if angle_id == "2":
            save_image_temporary(img_byte_arr, 'config/assets/angle2_temp', f'{file_name}_transparent.png')
        img_byte_arr.seek(0)  # Get the byte data


        #----------- get cropped original ----------------
        foreground_coordinates = rem_back["foreground"]
        cropped_original = crop_image_by_coordinates(
            image_bytes_io=image_file,
            top=foreground_coordinates["top"],
            left=foreground_coordinates["left"],
            width=foreground_coordinates["width"],
            height=foreground_coordinates["height"],
        )

        if angle_id == "2":
            save_image_temporary(cropped_original, 'config/assets/angle2_temp', f'{file_name}.png')
        #----------- resizing the transparent cropped image---------
        image_file_json = resize_image_bytesio(img_byte_arr, "width", 1920 )
        img_byte_arr = image_file_json['retruned_image']

        cropped_original_json = resize_image_bytesio(cropped_original, "width", 1920 )
        cropped_original_resized = cropped_original_json['retruned_image']

        # save_image_with_timestamp(img_byte_arr, 'outputs/lombaprob', 'resized.png')
       

        # add padding----
        # save_image_with_timestamp(img_byte_arr, 'outputs/dumtest/outputdebug/croppedcar', 'beforepadding.png')
        detected_vehicle = add_padding_to_image(img_byte_arr, 0, 215, 215, 180)
        detected_vehicle.seek(0)

        # save_image_with_timestamp(detected_vehicle, 'outputs/indtest/lombaprob', 'transparent.png')

        original_with_padding = add_padding_to_image(cropped_original_resized, 0, 215, 215, 180)
        original_with_padding.seek(0)

        # save_image_with_timestamp(original_with_padding, 'outputs/indtest/lombaprob', 'original_with_padding.png')


        img_byte_arr = detected_vehicle
        img_byte_arr.seek(0)


        image = Image.open(img_byte_arr)

        # Get the dimensions
        car_width, car_height = image.size  
        print(f"image width x height : {car_width} x {car_height}")
        # save_image_with_timestamp(img_byte_arr, 'outputs/indtest/lombaprob', 'removebg_resized_pic.png')


        with open("assets/proxy/proxybg.png", "rb") as bg_file:
            bg_bytes = BytesIO(bg_file.read())
     
        detected_vehicle =  create_proxy_background_with_car(bg_bytes, detected_vehicle) # original_with_padding
       
        # save_image_with_timestamp(detected_vehicle, 'outputs/indtest/lombaprob', 'with_proxy_image.png')
        #----------- previous detect --------------------

        # detected_cropped = detect_and_return_ByteIoImage(image_file, angle_id)

        # if detected_cropped is None or detected_cropped['detected_vehicle'] is None:
        #     notify_error_status(catalogue_id, angle_id, filename, "Car is not detected in your image. Please provide correct image.", current_file_name)
        #     return None


        # detected_vehicle = detected_cropped['detected_vehicle']
        # detected_vehicle.seek(0)

        after_detected_car = img_byte_arr


        #----------------add blurry license plate-----------------------
        is_licenseplate_blur = is_feature_enabled(catalogue_feature_list, cat_id=catalogue_id,  feature_name='License Plate Blur')
        log_to_json(f"Is the license plate blur selected : {is_licenseplate_blur}", current_file_name)
        if is_licenseplate_blur == True : 
            after_detected_car = make_the_licenseplate_blur(licensepalate_model_path, detected_vehicle, after_detected_car)
            if after_detected_car is None:
                notify_error_status(catalogue_id, angle_id, filename, "License Plate blur failed. License Plate could not be detected.", current_file_name, notify=1)
                after_detected_car = img_byte_arr
          #  reflected_image_path = save_image_with_timestamp(after_bg_image, 'outputs/dumtest/outputdebug/finalimages', 'blurred_license_image.png')
        else:
            print("License plate blur doesnt selected")
        # save_image_with_timestamp(after_detected_car, 'outputs/indtest/lombaprob', 'before_after_detected_car_lcns.png')
        # save_image_with_timestamp(detected_vehicle, 'outputs/indtest/lombaprob', 'before_detected_vehicle_lcns.png')
        print(f"License plate image path {set_license_plate_image}")
        #----------------add license plate image-----------------------
        is_licenseplate_image = is_feature_enabled(catalogue_feature_list, cat_id=catalogue_id,  feature_name='License Plate Image')
        log_to_json(f"Is the license plate image selected : {is_licenseplate_image}", current_file_name)
        print(set_license_plate_image)
        if is_licenseplate_image == True : 
            if set_license_plate_image is not None:
                licenseplate_image = get_dynamic_image(set_license_plate_image)
                print(f"licenseplate image type {type(licenseplate_image)}")
                after_detected_car_updated_algo = addLicensePlateImageNewAlgo(licensepalate_model_path, detected_vehicle, after_detected_car, licenseplate_image)
                if after_detected_car_updated_algo is None:
                    log_to_json(f"Licenseplate image failed with new algo. trying previous approach", current_file_name)
                    after_detected_car = addLicensePlateImage(licensepalate_model_path, detected_vehicle, after_detected_car, licenseplate_image)
                else:
                    after_detected_car = after_detected_car_updated_algo
                # save_image_with_timestamp(after_detected_car, 'outputs/indtest/lombaprob', 'imageadded_license_image.png')
                if after_detected_car is None:
                    after_detected_car = img_byte_arr
                    notify_error_status(catalogue_id, angle_id, filename, "License Plate image filling failed. License Plate could not be detected.", current_file_name, notify=1)
            else:
                log_to_json(f"You have selected license plate image but no image found ", current_file_name)
                notify_error_status(catalogue_id, angle_id, filename, "License Plate image filling failed. License Plate image could not be found.", current_file_name, notify=1)
           
          #  reflected_image_path = save_image_with_timestamp(after_bg_image, 'outputs/dumtest/outputdebug/finalimages', 'imageadded_license_image.png')
        else:
            print("License plate Image doesnt selected")

        after_detected_car.seek(0)
        #----------------rim polishing---------------------
        is_rim_shines = is_feature_enabled(catalogue_feature_list, cat_id=catalogue_id,  feature_name='Rim Polishing')
        log_to_json(f"Is the Rim Polishing selected: {is_rim_shines}", current_file_name)
        if is_rim_shines == True : 
            before_rim_polish_car = after_detected_car

            after_detected_car = rimpolishingfunc(detected_vehicle, after_detected_car)
            if after_detected_car is None:
                notify_error_status(catalogue_id, angle_id, filename, "Rim polishing failed. Rim could not be detected.", current_file_name, notify=1)
                after_detected_car = before_rim_polish_car
            else:
                after_detected_car.seek(0)
        # save_image_with_timestamp(after_detected_car, 'outputs/indtest/lombaprob', 'after_rim_polish.png')



        #------------- brightness polish car-----------------
        is_add_carpolish = is_feature_enabled(catalogue_feature_list, cat_id=catalogue_id,  feature_name='Polishing')
        log_to_json(f"Polish Car selected to be: {is_add_carpolish}", current_file_name)
        if is_add_carpolish == True:
            before_car_polish = after_detected_car
            after_detected_car = polish_car(after_detected_car)
            
            if after_detected_car is None:
                notify_error_status(catalogue_id, angle_id, filename, "Car polishing failed.", current_file_name, notify=1)
                after_detected_car = before_car_polish
            else:
                after_detected_car.seek(0)
            

        # save_image_with_timestamp(after_detected_car, 'outputs/indtest/lombaprob', 'after_polishing.png')
        #-------------------full crop or blur crop--------------
        print(f"Crop state value --- {set_crop_val}")
        if set_crop_val == 'blur_crop':
            blurcropped_image = paste_foreground_on_background_bytesio(blur_image_bytesio(image_file, set_blur_intesity),after_detected_car, foreground_coordinates["left"], foreground_coordinates["top"] ) # detected_cropped['x'], detected_cropped['y'] )
            final_image_link = save_image_to_s3(blurcropped_image, filename, catalogue_id, angle_id)
            if final_image_link:
                log_to_json(f"Processed and uploaded image: {final_image_link}", current_file_name)
                
                # Extract filename from final_image_link
                processed_filename = final_image_link.split('/')[-1]
                notify_success_status(catalogue_id, angle_id, processed_filename, current_file_name)
                # print(f"angle id is {angle_id} type {type(angle_id)}")
                # --------- image 5 and 6 process -------------
                print(f"angle id of front left type is {type(angle_id)}") 
            else:
                log_to_json(f"Failed to upload the processed image to S3.", current_file_name)
            
            return final_image_link


        #---------- angle setting-------------
        angle_view = None
    
        if angle_id == "4" or angle_id == "8":
            angle_view = 'reverse'
        else:
            angle_view = 'normal'
        
        #-------------------- reflection-----------------------
        is_add_reflection = is_feature_enabled(catalogue_feature_list, cat_id=catalogue_id,  feature_name='Car Reflection')
        is_add_reflection = False
        print(f"{catalogue_feature_list}")
        log_to_json(f"Is the image need to add refelction : {is_add_reflection}", current_file_name)
        print(f"feature selected is {catalogue_id}")
        after_ref_image = after_detected_car
        if is_add_reflection == True :
            after_ref_image = addReflection(detected_vehicle, after_detected_car,angle_id, angle_view)
            # save_image_with_timestamp(after_ref_image, 'outputs/reflectiondebug', 'reflected_image.png')
            if after_ref_image is None:
                notify_error_status(catalogue_id, angle_id, filename, "Car reflection could not be added.", current_file_name, notify=1)
                after_ref_image = after_detected_car
           # reflected_image_path = save_image_with_timestamp(after_ref_image, 'outputs/dumtest/outputdebug/finalimages', 'reflected_image.png')
        else:
            print("Reflection doesnt selected")

        #saving = save_image_with_timestamp(img_byte_arr, 'outputs/dumtest/outputdebug/finalimages', 'after.png')
        #---------add bg configs------------------
        print(f"angle id of front left type is {type(angle_id)}") 


        if angle_id == "4" or angle_id == "8":
            result = get_dynamic_wall_reverse_coordinates(detected_vehicle)

        else:
            result = get_dynamic_wall_coordinates(detected_vehicle)
        
        if result is None:
            notify_error_status(catalogue_id, angle_id, filename, "Wheels could not be detected. Please capture again.", current_file_name)
            # Handle the case where result is None, for example, skip further processing
            floor_coordinates, left_wall_coordinates, right_wall_coordinates, ceiling_coordinates = None, None, None, None
            return None
        else:
            # Unpack the result if it's not None
            floor_coordinates, left_wall_coordinates, right_wall_coordinates, ceiling_coordinates = result
    
        wallImages_bytes = None
        basicWall_images_bytes = None
        if set_bg_type == 'photo_box':
            wallImages_bytes = get_dynamic_wall_images(set_def_background) 
            if wallImages_bytes is None:
                notify_error_status(catalogue_id, angle_id, filename, "Background images are missing. Please configure again.", current_file_name)
                return None
        elif set_bg_type == 'basic':
            basicWall_images_bytes = get_dynamic_basic_wall_images(set_def_background)
            if basicWall_images_bytes is None:
                notify_error_status(catalogue_id, angle_id, filename, "Background images are missing. Please configure again.", current_file_name)
                return None
        if set_def_logo is not None and set_logo_position is not None:
            logo_bytesio = get_dynamic_image(set_def_logo, angle_id)
        else:
            logo_bytesio = None

        # save_image_with_timestamp(after_ref_image, 'outputs/indtest/lombaprob', 'before_entry_photobox.png')
        if set_bg_type == 'basic':
            final_output_stream = add_basic_bg(
                wallImagesBytes= basicWall_images_bytes, 
                logo_bytes=logo_bytesio, 
                new_image=detected_vehicle, 
                transparent_image=after_detected_car,
                logo_position=set_logo_position)
            if final_output_stream is None:
                notify_error_status(catalogue_id, angle_id, filename, "Background images are missing. Please configure again.", current_file_name)
                return None
        elif set_bg_type == 'photo_box':
            background = addBackgroundInitiate( wallImages_bytes=wallImages_bytes, 
                                                   logo_bytes= logo_bytesio, 
                                                   floor_coordinates =floor_coordinates, 
                                                   left_wall_coordinates= left_wall_coordinates, 
                                                   right_wall_coordinates=right_wall_coordinates, 
                                                   ceiling_coordinates=ceiling_coordinates,
                                                   angle=angle_view , 
                                                   logo_position=set_logo_position)
            if background:
                # save_image_with_timestamp(after_ref_image, 'outputs/indtest/lombaprob', 'before_combine.png')
                final_output_stream = combine_car_with_bg(background, after_ref_image , car_height)
            else:
                notify_error_status(catalogue_id, angle_id, filename, "Background images are missing. Please configure again.", current_file_name)
                return None
            # save_image_with_timestamp(background, 'outputs/dumtest/outputdebug/finalimages', 'generated_bg_image.png')
        else:
            background = get_universal_wall_images(set_def_background) 
            final_output_stream = combine_car_with_bg(background, after_ref_image , car_height)
             
         
        save_image_with_timestamp(final_output_stream, 'outputs/licenseplate', 'final_image.png')

        #final_image_link = save_image_with_timestamp(final_output_stream, 'outputs/dumtest/januarytrials/2/height_adjustment', f'{filename}.png')
        final_image_link = save_image_to_s3(final_output_stream, filename, catalogue_id, angle_id)
        
        if final_image_link:
            log_to_json(f"Processed and uploaded image: {final_image_link}", current_file_name)
            
            from config.configfunc import get_current_process_values

            if db_angle_id in [18, 19]:
                    return final_image_link
            
            
            processed_filename = final_image_link.split('/')[-1]
            notify_success_status(catalogue_id, angle_id, processed_filename, current_file_name)
 
            


        else:
            log_to_json(f"Failed to upload the processed image to S3.", current_file_name)
            
        return final_image_link
    
    except FileNotFoundError as e:
        log_to_json(f"File not found error {e}", current_file_name)
        notify_error_status(catalogue_id, angle_id, filename, f"System error occurred. It will be available soon.", current_file_name)
        return f'Sorry the image is not processed for: File not found: {str(e)}'
    except ValueError as e:
        log_to_json(f"ValueError {e}", current_file_name)
        notify_error_status(catalogue_id, angle_id, filename, f"System error occurred. It will be available soon.", current_file_name)
        return f'Sorry the image is not processed for: Value error: {str(e)}'
    except Exception as e:
        log_to_json(f"Exceptional error {e}", current_file_name)
        notify_error_status(catalogue_id, angle_id, filename, f"System error occurred. It will be available soon.", current_file_name)
        return  f'Sorry the image is not processed for:  An unexpected error occurred: {str(e)}'
    


def save_angle2_temporary(image_url):
   

    log_to_json("------------got the file in save_angle2_temporary----------", current_file_name)


    catalogue_id = angle_id = filename = None

    try:

        response = requests.get(image_url)
        if response.status_code != 200:
            log_to_json(f"Failed to download image from {image_url}", current_file_name)
            return None

                # Extract catalogue_id and angle_id from the URL
        url_parts = image_url.split('/')
        catalogue_id = url_parts[-3]  # Example: 1001
        angle_id = url_parts[-2]      # Example: 7
        filename = url_parts[-1]      # Example: image07-original.jpg
        file_name, file_extension = os.path.splitext(filename)        
        #image processing start by get the image

        image_file = BytesIO(response.content)

        log_to_json(f"catlaogue id is {catalogue_id} angle id is {angle_id} filename is {filename}", current_file_name)
       #-----------------remove bg-------------      
        
        rem_back = remove_background_premium(image_file, for_type='any')
  

        
        if rem_back["status"] == "failed":
            # Notify failure
            notify_response = notify_error_status(catalogue_id, angle_id, filename, f"There is a technical difficulty, please try again later.", current_file_name) 
            # foreground and credit er issue alada korte hobe
            return None

        # Background removal was successful; proceed with adding gray background and upload
        transparent_image = rem_back["image"]
        
        img_byte_arr = BytesIO()
        # saving = save_image_with_timestamp(transparent_image, 'outputs/dumtest/outputdebug/finalimages', 'before.png')
        transparent_image.save(img_byte_arr, format='PNG')  # Save the image in PNG format to the BytesIO object
        if angle_id == "2":
            folder_name = "angle2_temp"
        else:
            folder_name = "indangle"
        save_image_temporary(img_byte_arr, f'config/assets/{folder_name}', f'{file_name}_transparent.png')
        img_byte_arr.seek(0)  # Get the byte data


        #----------- get cropped original ----------------
        foreground_coordinates = rem_back["foreground"]
        cropped_original = crop_image_by_coordinates(
            image_bytes_io=image_file,
            top=foreground_coordinates["top"],
            left=foreground_coordinates["left"],
            width=foreground_coordinates["width"],
            height=foreground_coordinates["height"],
        )


        save_image_temporary(cropped_original, f'config/assets/{folder_name}', f'{file_name}.png')


    except FileNotFoundError as e:
        log_to_json(f"File not found error {e}", current_file_name)
        notify_error_status(catalogue_id, angle_id, filename, f"System error occurred. It will be available soon.", current_file_name)
        return f'Sorry the image is not processed for: File not found: {str(e)}'
    except ValueError as e:
        log_to_json(f"ValueError {e}", current_file_name)
        notify_error_status(catalogue_id, angle_id, filename, f"System error occurred. It will be available soon.", current_file_name)
        return f'Sorry the image is not processed for: Value error: {str(e)}'
    except Exception as e:
        log_to_json(f"Exceptional error {e}", current_file_name)
        notify_error_status(catalogue_id, angle_id, filename, f"System error occurred. It will be available soon.", current_file_name)
        return  f'Sorry the image is not processed for:  An unexpected error occurred: {str(e)}'