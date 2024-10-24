import os
import torch
import wandb
import logging
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset

# Wandb 초기화
wandb.init(project="maum_shelter")

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 데이터셋 로드
dataset = load_dataset("json", data_files="corpus.json")

# datasets의 train_test_split을 사용하여 데이터 분리
dataset = dataset['train'].train_test_split(test_size=0.2)
train_data = dataset['train']
val_data = dataset['test']

# 토크나이저 및 모델 로드
model_name = "facebook/opt-350m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 데이터 전처리
def preprocess_function(examples):
    inputs = [f"### User: {item['content']}\n" for item in examples["role"] if item == "user"]
    responses = [f"### Therapist: {item['content']}\n" for item in examples["role"] if item == "therapist"]
    model_inputs = tokenizer(inputs, padding="max_length", truncation=True)
    model_inputs["labels"] = tokenizer(responses, padding="max_length", truncation=True)["input_ids"]
    return model_inputs

train_data = train_data.map(preprocess_function, batched=True)
val_data = val_data.map(preprocess_function, batched=True)

# 학습 설정
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_steps=10,  # 로그를 10 스텝마다 찍음
    evaluation_strategy="steps",  # 스텝마다 평가
    eval_steps=50,  # 평가를 매 50 스텝마다 수행
    save_steps=100,
    load_best_model_at_end=True,
    logging_dir='./logs',
)

# Data collator 설정
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Trainer 설정
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# 학습 및 평가
for epoch in range(int(training_args.num_train_epochs)):
    logger.info(f"Epoch {epoch+1} 시작")
    
    # 학습
    train_result = trainer.train()
    train_metrics = train_result.metrics
    wandb.log({"train_loss": train_metrics['train_loss'], "epoch": epoch})
    
    # 평가
    eval_metrics = trainer.evaluate()
    wandb.log({"eval_loss": eval_metrics['eval_loss'], "epoch": epoch})
    
    # 모델 저장
    trainer.save_model()

wandb.finish()

