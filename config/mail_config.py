from flask_mail import Mail, Message
from my_logging_script import log_to_json
# Create a Mail instance (no app context yet)
mail = Mail()

current_file_name = "config/mail_config.py"

def send_email_notification(error_message):
    try:
        msg = Message(
            "Backend Error Notification",
            recipients=["shafinnafiullah@gmil.com"],
            body=f"An error occurred: {error_message}"
        )
        mail.send(msg)
        print("Error notification email sent.")
    except Exception as e:
        log_to_json(f"Failed to send error mail to Developer: {str(e)}", current_file_name)