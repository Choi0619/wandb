import json
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
import wandb

# Initialize WandB
wandb.init(project="therapist-chatbot", name="fine-tuning")

# Load the corpus data
with open('corpus.json', 'r', encoding='utf-8') as f:
    corpus = json.load(f)

# Prepare input-output pairs
data_pairs = []
for i in range(0, len(corpus), 2):
    input_text = corpus[i]['content']  # user input
    output_text = corpus[i + 1]['content']  # therapist response
    data_pairs.append({"input": input_text, "output": output_text})

# Split into train and validation sets (80-20 split)
train_data, val_data = train_test_split(data_pairs, test_size=0.2, random_state=42)

# Convert to Hugging Face datasets
train_dataset = Dataset.from_pandas(pd.DataFrame(train_data))
val_dataset = Dataset.from_pandas(pd.DataFrame(val_data))

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

# Preprocessing function
def preprocess_function(examples):
    inputs = examples['input']  # Access the input and output fields directly
    outputs = examples['output']
    
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")

    labels = tokenizer(outputs, max_length=512, truncation=True, padding="max_length").input_ids
    model_inputs["labels"] = labels
    return model_inputs

# Apply preprocessing
train_dataset = train_dataset.map(preprocess_function, batched=True)
val_dataset = val_dataset.map(preprocess_function, batched=True)

# Define response template and data collator
response_template = " ### Answer:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

# Define SFT configuration and trainer
sft_config = SFTConfig(output_dir="./results", evaluation_strategy="epoch", logging_strategy="epoch")

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    args=sft_config,
    data_collator=collator,
)

# Start training
trainer.train()

# Save the model
trainer.save_model("./fine_tuned_therapist_chatbot")

# Finish WandB logging
wandb.finish()
