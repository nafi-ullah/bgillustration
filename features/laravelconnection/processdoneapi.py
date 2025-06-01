import requests
from PIL import Image
from io import BytesIO
from my_logging_script import log_to_json
from config.configfunc import get_current_process_values, set_current_process_values

API_BASE_URL = "http://80.158.3.229:5054"
#API_BASE_URL = "http://localhost:5054"

def get_best_german_translation(input_text):
    messages = [
        {"en": "Failed to get the selected image. Please capture again.", "de": "Ausgewähltes Bild konnte nicht abgerufen werden. Bitte erneut aufnehmen."},
        {"en": "Default Crop or Background type or Default Background is not selected properly.", "de": "Standard-Zuschnitt, Hintergrundtyp oder Standardhintergrund wurde nicht korrekt ausgewählt."},
        {"en": "There is a technical difficulty, please try again later.", "de": "Es liegt ein technisches Problem vor, bitte versuchen Sie es später erneut."},
        {"en": "License Plate blur failed. License Plate could not be detected.", "de": "Unschärfe des Kennzeichens fehlgeschlagen. Kennzeichen konnte nicht erkannt werden."},
        {"en": "License Plate image filling failed. License Plate could not be detected.", "de": "Ausfüllen des Kennzeichenbildes fehlgeschlagen. Kennzeichen konnte nicht erkannt werden."},
        {"en": "License Plate image filling failed. License Plate image could not be found.", "de": "Ausfüllen des Kennzeichenbildes fehlgeschlagen. Kennzeichenbild konnte nicht gefunden werden."},
        {"en": "Rim polishing failed. Rim could not be detected.", "de": "Felgenpolitur fehlgeschlagen. Felge konnte nicht erkannt werden."},
        {"en": "Car polishing failed.", "de": "Autopolitur fehlgeschlagen."},
        {"en": "Car reflection could not be added.", "de": "Autospiegelung konnte nicht hinzugefügt werden."},
        {"en": "Wheels could not be detected. Please capture again.", "de": "Räder konnten nicht erkannt werden. Bitte erneut aufnehmen."},
        {"en": "Background images are missing. Please configure again.", "de": "Hintergrundbilder fehlen. Bitte erneut konfigurieren."},
        {"en": "System error occurred. It will be available soon.", "de": "Ein Systemfehler ist aufgetreten. Es wird bald wieder verfügbar sein."},
        {"en": "Wheel auto generation failed. Wheel could not be detected.", "de": "Automatische Raderzeugung fehlgeschlagen. Rad konnte nicht erkannt werden."},
        {"en": "Headlight auto generation failed. Headlight could not be detected.", "de": "Automatische Scheinwerfererzeugung fehlgeschlagen. Scheinwerfer konnte nicht erkannt werden."}
    ]
    
    default_de = "Ein Systemfehler ist aufgetreten. Es wird bald wieder verfügbar sein."
    input_words = set(input_text.lower().split())
    max_score = 0
    best_match = None

    for msg in messages:
        msg_words = set(msg["en"].lower().split())
        score = len(input_words & msg_words)  # number of common words

        if score > max_score:
            max_score = score
            best_match = msg

    # Threshold can be adjusted depending on needs
    if best_match and max_score > 0:
        return best_match["de"]
    else:
        return default_de

def notify_error_status(catalogue_id, angle_id, filename,  message, current_file_name, notify=0):
    log_to_json(f"Failed to handle: for cat_id {catalogue_id} ang_id {angle_id} filename {filename} Error: {message}", current_file_name)
    api_url = f"{API_BASE_URL}/api/process/done"

    current_config = get_current_process_values()
    billing_id = current_config.get("billing_id", 0)
    uploads_id = current_config.get("uploads_id", 0)
    db_catalogue_id = current_config.get("catalogue_id", 0)
    db_angle_id = current_config.get("angle_id", 0)
    db_filename = current_config.get("filename", 0)
    print(f"payload with billing id: {billing_id}")
    payload = {
        "catalogue_id": db_catalogue_id,
        "angle_id": db_angle_id,
        "filename": db_filename,
        "billing_id": billing_id,
        "upload_id": uploads_id,
        "status": 3,
        "message": message,
        "message_de": get_best_german_translation(message),
        "notify": notify
    }

    print(f"payload with billing id: {payload}")
    response = None
    try:
        response = requests.post(api_url, json=payload)
        from admin import send_email_notification
        send_email_notification(f'catalogue_id: {catalogue_id} angle_id: {angle_id} filename: {filename}: message: {message}')
        response.raise_for_status()
        log_to_json(f"Notification sent successfully for payload {payload}.. ,, notification response {response}", current_file_name)
        return response
    except requests.RequestException as e:
        error_message = f"Failed to send notification: payload {payload}.."
        if response is not None:
            error_message += f" response status: {response.status_code}, response text: {response.text}"
        else:
            error_message += f" error: {str(e)}"

        log_to_json(error_message, current_file_name)
        return None

#notify_error_status(catalogue_id, angle_id, filename, "Image download failed", current_file_name)

def notify_success_status(catalogue_id, angle_id, filename, current_file_name):
    api_url = f"{API_BASE_URL}/api/process/done"

    current_config = get_current_process_values()
    billing_id = current_config.get("billing_id", 0)
    uploads_id = current_config.get("uploads_id", 0)
    db_catalogue_id = current_config.get("catalogue_id", 0)
    db_angle_id = current_config.get("angle_id", 0)
    db_filename = current_config.get("filename", 0)
    response = None
    payload = {
        "catalogue_id": db_catalogue_id,
        "angle_id": db_angle_id,
        "filename": filename,
        "billing_id": billing_id,
        "upload_id": uploads_id,
        "status": 1,
        "message": "Image processed successfully",
        "message_de": "Bild erfolgreich verarbeitet"
    }

    #log_to_json(f"payload for notification: {payload}", current_file_name)

    try:
        notify_response = requests.post(api_url, json=payload)
        
        # Check if the notification was successful
        if notify_response.status_code == 201 or notify_response.status_code == 200:
            log_to_json(f"Processing completion API called successfully for payload {payload} ,, notification response {notify_response}", current_file_name)
            
        elif notify_response.status_code == 404:
            response_body = notify_response.json()
            log_to_json(f"Error: API endpoint not found ,, notification response {notify_response}. payload {payload} {response_body.get('message', 'No message provided')}", current_file_name)
            log_to_json(f"Error details: {response_body.get('error', 'No error details provided')}", current_file_name)
            
        else:
            log_to_json(f"Failed to call completion API.,, notification response {notify_response} Status Code: payload {payload} {notify_response.status_code}", current_file_name)
            log_to_json(f"Response text: payload {payload}.. {notify_response.text}", current_file_name)
            
        
        # Raise an exception for 4xx or 5xx status codes if not already handled
        notify_response.raise_for_status()
        return notify_response
    
    except requests.RequestException as e:
        error_message = f"Failed to send notification: payload {payload}.."
        if response is not None:
            error_message += f" response status: {response.status_code}, response text: {response.text}"
        else:
            error_message += f" error: {str(e)}"

        log_to_json(error_message, current_file_name)
        return None
    

def notify_background_success_status(pictures, preset_id, bg_type, filename, current_file_name):
    api_url = f"{API_BASE_URL}/api/setting/bg/processed"
    payload = {
        "pictures": pictures,
        "counter": preset_id,
        "bg_type": bg_type,
        "filename": filename,
        "status": 1,
        "message": "Background processed successfully"
    }
    print(payload)
    try:
        notify_response = requests.post(api_url, json=payload)
        
        # Check if the notification was successful
        if notify_response.status_code == 201 or notify_response.status_code == 200:
            log_to_json(f"Processing completion API called successfully for prictures:{pictures} preset_id: {preset_id} filename: {filename}", current_file_name)
            
        elif notify_response.status_code == 404:
            response_body = notify_response.json()
            log_to_json(f"Error: API endpoint not found.,, notification response {notify_response}.. {response_body.get('message', 'No message provided')}", current_file_name)
            log_to_json(f"Failed to sent api success response . Error details: {response_body.get('error', 'No error details provided')}", current_file_name)
            
        else:
            log_to_json(f"Failed to call completion API.,, notification response {notify_response}.. Status Code: {notify_response.status_code}", current_file_name)
            log_to_json(f"Response text: {notify_response.text}", current_file_name)
            
        
        # Raise an exception for 4xx or 5xx status codes if not already handled
        notify_response.raise_for_status()
        return notify_response
    
    except requests.RequestException as e:
        log_to_json(f"Failed to send notification: pictures {pictures}, presetid {preset_id}, filename {filename}. Error: {e}", current_file_name)
        return None
    

def notify_background_error_status(pictures, preset_id, bg_type, filename, message, current_file_name):
    api_url = f"{API_BASE_URL}/api/setting/bg/processed"
    payload = {
        "pictures": pictures,
        "catalogue_id": preset_id,
        "bg_type": bg_type,
        "filename": filename,
        "status": 0,
        "message": f"Failed to process background image"
    }
    
    try:
        notify_response = requests.post(api_url, json=payload)
        from admin import send_email_notification
        send_email_notification(message)
        
        # Check if the notification was successful
        if notify_response.status_code == 201 or notify_response.status_code == 200:

            log_to_json(f"Processing completion API called successfully for prictures:{pictures} preset_id: {preset_id} filename: {filename}", current_file_name)
            
        elif notify_response.status_code == 404:
            response_body = notify_response.json()
            log_to_json(f"Error: API endpoint not found. {response_body.get('message', 'No message provided')}", current_file_name)
            log_to_json(f"Failed to sent api success response . Error details: {response_body.get('error', 'No error details provided')}", current_file_name)
            
        else:
            log_to_json(f"Failed to call completion API. Status Code: {notify_response.status_code}", current_file_name)
            log_to_json(f"Response text: {notify_response.text}", current_file_name)
            
        
        # Raise an exception for 4xx or 5xx status codes if not already handled
        notify_response.raise_for_status()
        return notify_response
    
    except requests.RequestException as e:
        log_to_json(f"Failed to send notification: pictures {pictures}, presetid {preset_id}, filename {filename}. Error: {e}", current_file_name)
        return None
