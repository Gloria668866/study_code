import requests
import json

# FastAPI endpoint URL
url = "http://localhost:8001/process"

# Sample JSON data to be sent in the request
json_data = {
    "name": "杜毅",
    "company": "传智播客",
    "subject": "python大模型人工智能"
}

# Convert dictionary to JSON string for sending in the request
json_data_str = json.dumps(json_data)

# Path to an example audio file to upload (update with your actual file path)
file_path = "/Users/duyi/PycharmProjects/interview_voice_project/__001__data/罗培鑫面试.aac"  # Update this with the correct file path

# Open the file and prepare the request
with open(file_path, 'rb') as file:
    files = {'file': (file_path, file, 'audio')}  # Adjust the MIME type based on the file format
    data = {
        'json_data_str': json_data_str}  # The key 'json_data_str' should match your FastAPI endpoint's expected parameter

    # Send POST request
    response = requests.post(url, data=data, files=files, stream=True)

# Handle the response
if response.status_code == 200:
    print("Request was successful!")
    print("Streaming Response:")
    # Print each line of the streamed response
    for line in response.iter_lines():
        if line:
            print(line.decode('utf-8'))  # Print the line from the server
else:
    print(f"Failed request. Status code: {response.status_code}")
    print(f"Response: {response.text}")
