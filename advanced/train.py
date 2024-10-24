import os
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load the environment variables from .env file
load_dotenv()

# Get the Hugging Face API token from the .env file
hf_token = os.getenv('HF_TOKEN')

# Log into Hugging Face using the token
login(hf_token)

# Define the model name
model_name = "google/gemma-2b"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the prompt
prompt = "Can you describe dogs in detail?"

# Tokenize the input and move it to the same device as the model
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generate the response from the model
outputs = model.generate(**inputs, max_new_tokens=150)

# Decode the output to text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Print the generated text
print(generated_text)
