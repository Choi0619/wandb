import os
import sys
import math
import torch
import wandb
import logging
import datasets
import argparse
import evaluate
import transformers

from typing import Optional
from itertools import chain
from dataclasses import dataclass, field

from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator
)
from transformers.trainer_utils import get_last_checkpoint

# Wandb initialization
wandb.init(project='gyuhwan')
wandb.run.name = 'gpt-finetuning'

# Arguments for fine-tuning
@dataclass
class Arguments:
    model_name_or_path: Optional[str] = field(default=None)
    torch_dtype: Optional[str] = field(default=None, metadata={'choices': ['auto', 'bfloat16', 'float16', 'float32']})
    dataset_name: Optional[str] = field(default=None)
    dataset_config_name: Optional[str] = field(default=None)
    block_size: int = field(default=1024)
    num_workers: Optional[int] = field(default=None)

# Parsing arguments
parser = HfArgumentParser((Arguments, TrainingArguments))
args, training_args = parser.parse_args_into_dataclasses()

# Setup logger
logger = logging.getLogger()
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log_level = training_args.get_process_log_level()
logger.setLevel(log_level)
datasets.utils.logging.set_verbosity(log_level)
transformers.utils.logging.set_verbosity(log_level)
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()

logger.info(f"Training/evaluation parameters {training_args}")

# Load dataset
raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)

# Load pre-trained model and tokenizer
config = AutoConfig.from_pretrained(args.model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    config=config,
    torch_dtype=args.torch_dtype
)

# Set pad token ID and resize token embeddings
tokenizer.pad_token_id = tokenizer.eos_token_id
embedding_size = model.get_input_embeddings().weight.shape[0]
if len(tokenizer) > embedding_size:
    model.resize_token_embeddings(len(tokenizer))

# Tokenize dataset
def tokenize_function(examples):
    output = tokenizer(examples['text'])
    return output

with training_args.main_process_first(desc="dataset map tokenization"):
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=args.num_workers,
        remove_columns=raw_datasets["train"].column_names
    )

# Validation 데이터셋 10%로 분리
train_test_split = raw_datasets["train"].train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
validation_dataset = train_test_split["test"]

# Group text for causal language modeling
def group_texts(examples):
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // args.block_size) * args.block_size
    result = {
        k: [t[i:i + args.block_size] for i in range(0, total_length, args.block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

with training_args.main_process_first(desc="grouping texts together"):
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=args.num_workers
    )

# Define Trainer with validation data
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,  # validation 데이터 추가
    tokenizer=tokenizer,
    data_collator=default_data_collator
)

# Resume from checkpoint if available
checkpoint = None
last_checkpoint = get_last_checkpoint(training_args.output_dir)
if training_args.resume_from_checkpoint is not None:
    checkpoint = training_args.resume_from_checkpoint
else:
    checkpoint = last_checkpoint

# Train the model
train_result = trainer.train(resume_from_checkpoint=checkpoint)
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()

# Evaluate on validation set
eval_result = trainer.evaluate(eval_dataset=validation_dataset)
trainer.log_metrics("eval", eval_result)
trainer.save_metrics("eval", eval_result)

# Log train and eval loss to Wandb
wandb.log({"train/loss": metrics["train_loss"], "eval/loss": eval_result["eval_loss"]})
