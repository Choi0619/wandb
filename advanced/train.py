import os
import requests
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Your Hugging Face API token (loaded from .env)
HF_TOKEN = os.getenv("HF_TOKEN")

# API URL for the Hugging Face model
API_URL = "https://api-inference.huggingface.co/models/gemma-2b"

# Headers with the authorization token
headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

# The input prompt for the model
data = {
    "inputs": "Explain about dogs."
}

# Make the POST request to the Hugging Face API
response = requests.post(API_URL, headers=headers, json=data)

# Check for successful response and print the output
if response.status_code == 200:
    output = response.json()
    print("Response from the model:", output[0]['generated_text'])
else:
    print(f"Request failed with status code {response.status_code}")
    print(response.text)
