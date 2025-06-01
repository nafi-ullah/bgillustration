from flask import Flask, request, jsonify
import json
import os
from pathlib import Path
import boto3
from botocore.client import Config as BotoConfig

current_file_name = "config/configfunc.py"
# Path to the config file
CONFIG_FILE_PATH = "config/removebgsetting.json"
config_file_path = 'config/removebgapikey.json'

GLOBAL_SETTINGS_FILE = 'config/config_jsons/global_setting.json'
CATALOGUE_SETTINGS_FILE = 'config/config_jsons/catalogue_feature_setting.json'
CURRENT_PROCESS_FILE = 'config/config_jsons/currentprocess.json'
LOG_DB_FILE = 'config/config_jsons/log_db.json'
Current_Process_Path = Path(f"{CURRENT_PROCESS_FILE}")


# config/configfunc.py


# OBS Credentials and Config
access_key = '4IUMPGRCHXNLAJ5EUUGP'
secret_key = 'LakMU0aI4zTtEd3dzxr2LUD5R9EgvEMtBTgf7ukd'
region = 'eu-de'
bucket_name = 'vroomview'
endpoint_url = 'https://obs.eu-de.otc.t-systems.com'
obs_folder = 'pythonProcessing/config'

# S3 client setup
s3_client = boto3.client(
    's3',
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key,
    region_name=region,
    endpoint_url=endpoint_url,
    config=BotoConfig(s3={'addressing_style': 'path'})
)

# List of required config files and their local paths
CONFIG_FILES = {
    'global_setting.json': f'{GLOBAL_SETTINGS_FILE}',
    'catalogue_feature_setting.json': f'{CATALOGUE_SETTINGS_FILE}',
    'currentprocess.json': f'{CURRENT_PROCESS_FILE}',
    'log_db.json': f'{LOG_DB_FILE}'
}

def ensure_config_files():
    for filename, local_path in CONFIG_FILES.items():
        local_path_obj = Path(local_path)
        if not local_path_obj.exists():
            print(f"{filename} not found. Downloading from OBS...")
            local_path_obj.parent.mkdir(parents=True, exist_ok=True)
            try:
                s3_key = f"{obs_folder}/{filename}"
                with open(local_path, 'wb') as f:
                    s3_client.download_fileobj(bucket_name, s3_key, f)
                print(f"{filename} downloaded and saved to {local_path}")
            except Exception as e:
                print(f"Failed to download {filename} from OBS: {str(e)}")
        else:
            print(f"{filename} already exists. Skipping download.")








# Helper function to read the configuration file
def read_config():
    if not os.path.exists(CONFIG_FILE_PATH):
        return {"isremovebg": "false"}  # Default value if file does not exist
    with open(CONFIG_FILE_PATH, "r") as file:
        return json.load(file)

# Helper function to write to the configuration file
def write_config(data):
    with open(CONFIG_FILE_PATH, "w") as file:
        json.dump(data, file, indent=4)

def read_config_bool():
    if not os.path.exists(CONFIG_FILE_PATH):
        return False  # Default value if file does not exist
    with open(CONFIG_FILE_PATH, "r") as file:
        config = json.load(file)
        # Return True or False based on the "isremovebg" value
        return config.get("isremovebg", "false") == "true"

# API to update the isremovebg value

def read_api_key():
    try:
        with open(config_file_path, 'r') as file:
            config = json.load(file)
            return config.get('rembgapikey', None)
    except FileNotFoundError:
        return ""
    except json.JSONDecodeError:
        return ""

# Function to write the API key to the JSON file
def write_api_key(new_api_key):
    try:
        with open(config_file_path, 'w') as file:
            json.dump({"rembgapikey": new_api_key}, file)
    except Exception as e:
        return str(e)
    return "API key updated successfully!"



def set_current_process_values(new_data):
 
    try:
        # Ensure the input is a dictionary
        if not isinstance(new_data, dict):
            raise ValueError("Input data must be a dictionary.")

        # Write the new data to the file
        with Current_Process_Path.open("w") as file:
            json.dump(new_data, file, indent=4)

        print(f"Configuration updated: {new_data}")


        process_uid = get_process_uid()
        angle_id = new_data.get("angle_id")
        filename = new_data.get("filename")
        catalogue_id = new_data.get("catalogue_id")

        if None in (process_uid, angle_id, filename, catalogue_id):
            raise ValueError("Missing required fields to create processImage log.")

        url = f"https://vroomview.obs.eu-de.otc.t-systems.com/uploads/{catalogue_id}/{angle_id}/{filename}"

        process_image_data = {
            "process_uid": process_uid,
            "angle_id": angle_id,
            "filename": filename,
            "url": url
        }
        from app import create_app
        app = create_app()
        with app.app_context():
            from app.dbfunc import create_process_image_log
            processImage_uid = create_process_image_log(process_image_data)
            update_processImage_uid(processImage_uid)

           

        print(f"ProcessImage log created with UID: {processImage_uid}")


    except Exception as e:
        print(f"Error setting current process values: {e}")


def get_current_process_values():
 
    try:
        # Read the existing data
        with Current_Process_Path.open("r") as file:
            config_data = json.load(file)

        return config_data
    except Exception as e:
        print(f"Error reading current process values: {e}")
    


# Load the JSON data
def load_global_settings():
    with open(GLOBAL_SETTINGS_FILE, 'r') as file:
        return json.load(file)

# Save the JSON data
def save_global_settings(data):
    with open(GLOBAL_SETTINGS_FILE, 'w') as file:
        json.dump(data, file, indent=4)


def load_catalogue_settings():
    with open(CATALOGUE_SETTINGS_FILE, 'r') as file:
        return json.load(file)

# Save the JSON data
def save_catalogue_settings(data):
    with open(CATALOGUE_SETTINGS_FILE, 'w') as file:
        json.dump(data, file, indent=4)


##--------------- log db management ------------------##
def _read_log_file():
    path = Path(LOG_DB_FILE)
    if not path.exists():
        # Initialize file if it doesn't exist
        with open(path, 'w') as f:
            json.dump({"process_uid": 0, "processImage_uid": 0}, f)
    with open(path, 'r') as f:
        return json.load(f)


def _write_log_file(data):
    with open(LOG_DB_FILE, 'w') as f:
        json.dump(data, f, indent=4)


def update_process_uid(new_uid):
    data = _read_log_file()
    data["process_uid"] = new_uid  # Only update process_uid
    _write_log_file(data)


def update_processImage_uid(new_uid):
    data = _read_log_file()
    data["processImage_uid"] = new_uid  # Only update processImage_uid
    _write_log_file(data)


def get_process_uid():
    return _read_log_file().get("process_uid")


def get_processImage_uid():
    return _read_log_file().get("processImage_uid")