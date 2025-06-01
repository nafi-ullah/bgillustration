from datetime import datetime
from io import BytesIO
import os
from PIL import Image
import boto3
from botocore.client import Config
from my_logging_script import log_to_json
from features.laravelconnection.processdoneapi import notify_error_status
access_key = '4IUMPGRCHXNLAJ5EUUGP'
secret_key = 'LakMU0aI4zTtEd3dzxr2LUD5R9EgvEMtBTgf7ukd'
region = 'eu-de'
bucket_name = 'vroomview'
endpoint_url = 'https://obs.eu-de.otc.t-systems.com'
get_url = 'https://vroomview.obs.eu-de.otc.t-systems.com'

current_file_name ="uploadImageBucket.py"

s3_client = boto3.client(
    's3',
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key,
    region_name=region,
    endpoint_url=endpoint_url,
    config=Config(s3={'addressing_style': 'path'})
)


def upload_to_s3(image, original_filename, catalogue_id, angle_id):

    # Create in-memory file
    in_memory_file = BytesIO()
    image.save(in_memory_file, format="PNG")
    in_memory_file.seek(0)  # Reset file pointer to start

    # Generate S3 object path
    base_filename = os.path.splitext(original_filename)[0].replace("_original", "")
    result_filename = f"{base_filename}_processed.png"

    s3_object_name = f"uploads/{catalogue_id}/{angle_id}/{result_filename}"

    try:
        # Upload to S3
        s3_client.upload_fileobj(in_memory_file, bucket_name, s3_object_name)
        final_image_link = f"{get_url}/{s3_object_name}"
        log_to_json(f"Uploaded angle_id {angle_id} processed image to S3: {final_image_link}", current_file_name)
        return final_image_link
    except Exception as e:
        log_to_json(f"Error uploading file to S3 of cat_id {catalogue_id} angle_id {angle_id} filename {original_filename}: {e}", current_file_name)
        # Notify failure via API
        notify_error_status(catalogue_id, angle_id, original_filename, "Upload to bucket failed")
        return None
    

def upload_image_to_s3(image, object_name):
    try:
        # Convert image to RGB mode if it is in RGBA or P mode
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")

        # Prepare the image file
        image_file = BytesIO()
        image.save(image_file, format='JPEG')  # Save the image to BytesIO in JPEG format
        image_file.seek(0)

        # Upload to S3
        s3_client.upload_fileobj(image_file, bucket_name, object_name)
        uploaded_image_url = f"{endpoint_url}/{bucket_name}/{object_name}"
        log_to_json(f"uploaded image url {uploaded_image_url}", current_file_name)
        return uploaded_image_url
    except Exception as e:
        log_to_json(f"Error uploading to S3: {e}", current_file_name)
        return None
