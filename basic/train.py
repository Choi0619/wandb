import torch
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
import json
from sklearn.model_selection import train_test_split

# Initialize wandb project
wandb.init(project='LLM_instruction_tuning', name='chatbot-finetuning')

# Load GPT-2 model and tokenizer
print("Loading GPT-2 model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
print(f"GPT-2 model and tokenizer loaded successfully. Using device: {device}")

# Load data
with open("corpus.json", "r", encoding="utf-8") as f:
    corpus = json.load(f)

print("JSON file loaded successfully.")
print(f"Data example: {corpus[0]}")

# Convert data to question-answer pairs
formatted_data = []
for i in range(0, len(corpus), 2):
    if i + 1 < len(corpus):  # Check if there's a matching response
        formatted_data.append({
            "instruction": corpus[i]["content"],
            "response": corpus[i + 1]["content"]
        })

print(f"Converted data example: {formatted_data[0]}")

# Split data into train/validation sets
train_data, valid_data = train_test_split(formatted_data, test_size=0.2, random_state=42)
print(f"Train data size: {len(train_data)}, Validation data size: {len(valid_data)}")

# Convert to Dataset objects
train_dataset = Dataset.from_dict({
    "instruction": [d["instruction"] for d in train_data],
    "response": [d["response"] for d in train_data]
})
valid_dataset = Dataset.from_dict({
    "instruction": [d["instruction"] for d in valid_data],
    "response": [d["response"] for d in valid_data]
})

print("Successfully converted to Dataset objects.")

# Data formatting function
def formatting_prompts_func(example):
    text = f"### Question: {example['instruction']}\n ### Answer: {example['response']}"
    return tokenizer(
        text,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    )

# Response template and data collator
response_template = " ### Answer:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

# Configure trainer
trainer_config = SFTConfig(
    output_dir="./results",
    run_name="gpt2-finetuning",
    evaluation_strategy="steps",
    eval_steps=20,
    save_strategy="steps",
    save_steps=20,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    logging_steps=10,
    learning_rate=5e-5,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    fp16=True,
    use_cache=False,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
)

# Initialize trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    formatting_func=formatting_prompts_func,
    args=trainer_config,
    data_collator=collator,
)

# Start training
print("Starting training...")
try:
    train_result = trainer.train()
    
    # Save model
    trainer.save_model("./trained_model")
    
    # Log metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # Run evaluation
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)
    
    print("Training completed successfully!")

except Exception as e:
    print(f"An error occurred during training: {str(e)}")
    raise

finally:
    wandb.finish()