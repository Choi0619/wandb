import os
import json
import torch
import wandb
import logging
from sklearn.model_selection import train_test_split
from datasets import Dataset, load_metric
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

# Wandb 프로젝트 초기화
wandb.init(project='LLM_instruction_tuning')  # 프로젝트 이름 설정
wandb.run.name = 'gpt2-instruction-tuning'  # Wandb 실행 이름 설정

# 로깅 설정
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# GPT-2 모델과 토크나이저 로드
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# 데이터 로드
with open('corpus.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 데이터셋 분할 (8:2 비율로 train과 validation 나누기)
train_data, valid_data = train_test_split(data, test_size=0.2)

# 데이터를 Hugging Face Dataset 형태로 변환
def preprocess_data(data):
    texts = []
    for pair in data:
        role, content = pair['role'], pair['content']
        if role == 'user':
            instruction = f"User: {content}\nTherapist:"
        else:
            instruction = f"Therapist: {content}"
        texts.append(instruction)
    return texts

train_texts = preprocess_data(train_data)
valid_texts = preprocess_data(valid_data)

train_dataset = Dataset.from_dict({"text": train_texts})
valid_dataset = Dataset.from_dict({"text": valid_texts})

# 토크나이즈 및 텍스트 그룹화
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=1024)

train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
valid_dataset = valid_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Data collator 설정
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# TrainingArguments 설정
training_args = TrainingArguments(
    output_dir="./output",
    evaluation_strategy="steps",
    eval_steps=500,
    logging_steps=100,
    save_steps=1000,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    save_total_limit=1,
    report_to="wandb",  # wandb로 로깅
    logging_dir="./logs",
    logging_first_step=True,
    logging_dir="./logs",
    fp16=True,  # GPU에서 mixed precision 사용 (속도 향상)
)

# Trainer 설정
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# 학습 실행
train_result = trainer.train()
trainer.save_model()

# 학습 결과 저장 및 로깅
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)

# 평가 실행
eval_metrics = trainer.evaluate()
trainer.log_metrics("eval", eval_metrics)
trainer.save_metrics("eval", eval_metrics)

# Trainer state 저장
trainer.save_state()

# Wandb 학습 종료
wandb.finish()
