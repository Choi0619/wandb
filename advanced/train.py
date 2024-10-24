import os
import requests
import json
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load environment variables from .env file
load_dotenv()

# Your Hugging Face API token (loaded from .env)
HF_TOKEN = os.getenv("HF_TOKEN")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Test prompt
prompt = "Explain about cat."

# Tokenize input
inputs = tokenizer(prompt, return_tensors="pt")

# Generate response
outputs = model.generate(**inputs)

# Decode and print the generated text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated text:", generated_text)
