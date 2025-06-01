import requests
from PIL import Image
from io import BytesIO
import pytesseract
from my_logging_script import log_to_json

def extract_text_from_image(image_url):
    """
    Extracts text from an image URL using OCR.
    
    Args:
        image_url (str): The URL of the image.
        
    Returns:
        str: The text extracted from the image.
    """
    try:
        # Fetch the image from the URL
        response = requests.get(image_url)
        response.raise_for_status()  # Raise an error for bad HTTP responses
        
        # Open the image
        image = Image.open(BytesIO(response.content))
        
        # Use Tesseract to extract text
        text = pytesseract.image_to_string(image)
        return text
    
    except requests.exceptions.RequestException as e:
        log_to_json(f"Error fetching the image: {e}", "ocranalysis.py")
        return f"Error fetching the image: {e}"
    except Exception as e:
        log_to_json(f"Error processing the image: {e}", "ocranalysis.py")
        return f"Error processing the image: {e}"

# Example usage:
# url = "https://example.com/sample-image.jpg"
# print(extract_text_from_image(url))
