from io import BytesIO
from PIL import Image

def convert_image_to_bytesio(pil_image: Image.Image, format: str = "PNG") -> BytesIO:

    try:
        # Create a BytesIO object
        output = BytesIO()

        # Save the PIL image to the BytesIO object
        pil_image.save(output, format=format)

        # Reset the cursor to the start of the BytesIO object
        output.seek(0)

        return output
    except Exception as e:
        print(f"Error converting image to BytesIO: {str(e)}")
        return None


def convert_image_to_png(image_bytesio: BytesIO) -> BytesIO:
    try:
        # Open the image from BytesIO object
        image = Image.open(image_bytesio)

        # Create a new BytesIO object to store the PNG image
        png_image = BytesIO()

        # Save the image as PNG format to the new BytesIO object
        image.save(png_image, format="PNG")

        # Reset the pointer to the beginning of the BytesIO object
        png_image.seek(0)

        return png_image
    except Exception as e:
        print(f"Error converting image to PNG: {str(e)}")
        return None