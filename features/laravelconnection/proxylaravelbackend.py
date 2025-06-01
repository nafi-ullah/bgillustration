from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/process/done', methods=['POST'])
def process_done():
    try:
        # Get the JSON payload from the request
        payload = request.json
        
        # Extract values
        catalogue_id = payload.get("catalogue_id")
        angle_id = payload.get("angle_id")
        filename = payload.get("filename")
        status = payload.get("status")
        message = payload.get("message")
        
        # Print the values to the console
        print(f"Received Data: Cat_ID: {catalogue_id} Angle_ID: {angle_id} Filename: {filename} Status: {status} Message: {message}")
    
        
        # Return a success response
        return jsonify({"message": "Data received successfully"}), 200

    except Exception as e:
        # Handle errors
        print(f"Error processing request: {e}")
        return jsonify({"error": "Failed to process data"}), 500
    
@app.route('/api/setting/bg/processed', methods=['POST'])
def bg_process_done():
    try:
        # Get the JSON payload from the request
        payload = request.json
        
        # Extract values
        pictures = payload.get("pictures")
        catalogue_id = payload.get("catalogue_id")
        bg_type = payload.get("bg_type")
        filename = payload.get("filename")
        status = payload.get("status")
        message = payload.get("message")
        
        # Print the values to the console
        print(f"Received Data: Cat_ID: {catalogue_id} pictures: {pictures} bg_type: {bg_type} Filename: {filename} Status: {status} Message: {message}")
    
        
        # Return a success response
        return jsonify({"message": "Data received successfully"}), 200

    except Exception as e:
        # Handle errors
        print(f"Error processing request: {e}")
        return jsonify({"error": "Failed to process data"}), 500

if __name__ == '__main__':
    app.run(port=5054, debug=True)
