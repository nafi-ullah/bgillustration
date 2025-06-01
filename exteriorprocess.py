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
from dynamicfuncs import is_feature_enabled, get_user_setting, save_image_with_timestamp, save_image_temporary ,getLicensePlateImage, get_blur_intensity
from features.core.licenseplateoperations.licenseplateblurfunc import make_the_licenseplate_blur
from features.core.licenseplateoperations.licenseplateimagefunc import addLicensePlateImage
from features.core.licenseplateoperations.licenseplateimageupdated import addLicensePlateImageNewAlgo
from features.functionalites.backgroundoperations.reflection import addReflection
from features.functionalites.backgroundoperations.addbackground.addwallinitiate import addBackgroundInitiate
from backgroundconfs import get_dynamic_wall_coordinates, get_universal_wall_images,  get_dynamic_image
from features.functionalites.backgroundoperations.combinebgwithcar import combine_car_with_bg

from features.functionalites.backgroundoperations.basicbg.basicbgprocess import add_basic_bg
from features.functionalites.imageoperations.basic import resize_image_bytesio, get_image_dimensions, paste_foreground_on_background_bytesio, blur_image_bytesio, resize_1920_image
from features.core.polishing import polish_car
from features.core.rimpolishing import rimpolishingfunc
import cv2
from coordinates import getCoordinates


current_file_name = "features/exterior_process/exteriorprocess.py"
licensepalate_model_path = './models/licenseplate/augmented/best.pt'


def frontleft():
    try:
        
        # Load car image and convert to BytesIO
        car_image_path = "assets/cars/images2_original.jpg"
        with open(car_image_path, "rb") as f:
            detected_vehicle = BytesIO(f.read())

        # Load wall images and prepare as JSON
        wall_images_paths = {
            "ceiling_wall_img": f"assets/walls/7/ceiling.jpg",
            "left_wall_img": f"assets/walls/7/lw.jpg",
            "right_wall_img": f"assets/walls/7/rw.jpg",
            "floor_wall_img": f"assets/walls/7/floor.jpg"
        }
        wallImages_bytes = {}
        for key, path in wall_images_paths.items():
            with open(path, "rb") as f:
                wallImages_bytes[key] = BytesIO(f.read())

        angle_id = "15"
        wallCoordinates = getCoordinates(angle_id, detected_vehicle)

        print(wallCoordinates)

        floor_left_top = wallCoordinates["floor_left_top"]
        floor_left_bottom = wallCoordinates["floor_left_bottom"]
        floor_right_bottom = wallCoordinates["floor_right_bottom"]
        floor_right_top = wallCoordinates["floor_right_top"]
        rwall_top_left = wallCoordinates["rwall_top_left"]
        rwall_right_bottom = wallCoordinates["rwall_right_bottom"]
        rwall_top_right = wallCoordinates["rwall_top_right"]
        lwall_left_top = wallCoordinates["lwall_left_top"]
        lwall_left_bottom = wallCoordinates["lwall_left_bottom"]
        canvas_middle_ref = wallCoordinates["canvas_middle_ref"]
        ceiling_top = wallCoordinates["ceiling_top"]

        floor_coordinates = [floor_left_top, floor_left_bottom, floor_right_bottom, floor_right_top]
        left_wall_coordinates = [rwall_top_left, lwall_left_top, lwall_left_bottom, canvas_middle_ref]
        right_wall_coordinates = [rwall_top_left, canvas_middle_ref, rwall_right_bottom, rwall_top_right]
        ceiling_coordinates = [lwall_left_top, rwall_top_left, rwall_top_right, ceiling_top]



    
    
    
        # Load logo image and convert to BytesIO
        logo_path = "assets/logo/Artboard 47.png"
        with open(logo_path, "rb") as f:
            logo_bytesio = BytesIO(f.read())

        angle = "normal"
        if angle_id in ["1","2", "3","5","6","7","9","12","13","16"]:
            angle = "normal"
        else:
            angle = "reverse"

        background = addBackgroundInitiate(
            wallImages_bytes=wallImages_bytes, 
            logo_bytes=logo_bytesio, 
            floor_coordinates=floor_coordinates, 
            left_wall_coordinates=left_wall_coordinates, 
            right_wall_coordinates=right_wall_coordinates, 
            ceiling_coordinates=ceiling_coordinates,
            angle=angle,
            logo_position="auto"
        )
        # if background:
        final_image_link = save_image_with_timestamp(background, 'outputs/backgrounds', 'newbg.png')

        

            
        return final_image_link
    
    except Exception as e:
        log_to_json(f"Exceptional error {e}", current_file_name)
        return  f'Sorry the image is not processed for:  An unexpected error occurred: {str(e)}'
    



getbg = frontleft()