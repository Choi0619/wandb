import os
import torch
import json
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset, load_metric

# Load Hugging Face API token from environment variables
from dotenv import load_dotenv
from huggingface_hub import login

# Load .env file
load_dotenv()
hf_token = os.getenv('HF_TOKEN')

# Hugging Face login using token
login(hf_token)

# Load the tokenizer and model
model_name = "EleutherAI/gpt-neo-1.3B"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPTNeoForCausalLM.from_pretrained(model_name)

# Ensure padding tokens are set
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer))

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load dataset (corpus.json)
with open('corpus.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Prepare dataset
def preprocess_data(data):
    instructions, outputs = [], []
    for i in range(0, len(data), 2):
        if data[i]['role'] == 'user' and data[i+1]['role'] == 'therapist':
            instructions.append(data[i]['content'])
            outputs.append(data[i+1]['content'])
    return instructions, outputs

instructions, outputs = preprocess_data(data)

# Convert to HuggingFace Dataset format
dataset = Dataset.from_dict({
    "instruction": instructions,
    "output": outputs
})

# Split data into training and evaluation sets (80/20 split)
train_test_split = dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

# Tokenize the dataset
def tokenize_function(examples):
    inputs = tokenizer(examples['instruction'], padding="max_length", truncation=True, max_length=512)
    outputs = tokenizer(examples['output'], padding="max_length", truncation=True, max_length=512)
    inputs["labels"] = outputs["input_ids"]
    return inputs

# Apply tokenization to datasets
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,  # Reduce batch size if memory issues occur
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    evaluation_strategy="steps",
    eval_steps=100,
    logging_steps=100,
    save_steps=100,
    save_total_limit=2,
    fp16=True if torch.cuda.is_available() else False,  # Mixed precision training to reduce memory usage
    gradient_accumulation_steps=2,  # Accumulate gradients over multiple steps
    load_best_model_at_end=True,
    report_to=None  # Disable reporting to WandB for now
)

# Data collator for causal language modeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Initialize the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Fine-tune the model
trainer.train()

# Save the model
trainer.save_model("./fine_tuned_model")

# Sample prompt for generation
sample_prompt = "너무 무기력한데 어떻게 해야할지 모르겠어."
input_ids = tokenizer(sample_prompt, return_tensors="pt").input_ids.to(device)
output = model.generate(input_ids, max_length=150)
print("Generated response:", tokenizer.decode(output[0], skip_special_tokens=True))
