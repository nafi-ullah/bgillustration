import os
from my_logging_script import log_to_json
current_file_name = "features/functionalites/osoperation.py"

def delete_all_files(folder_path):

    if not os.path.isdir(folder_path):
        log_to_json(f"'{folder_path}' is not a valid directory.", current_file_name)
        return

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            try:
                os.remove(file_path)
                log_to_json(f"Deleted file: {file_path}", current_file_name)
            except Exception as e:
                log_to_json(f"Error deleting {file_path}: {e}", current_file_name)
        else:
            log_to_json(f"Skipped (not a file): {file_path}", current_file_name)
