import os
import json
import wandb
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, DataCollatorForLanguageModeling
from trl import SFTTrainer

# Wandb 초기화
wandb.init(project='maum_shelter')
wandb.run.name = 'gpt-instruction-tuning'

# 모델과 토크나이저 로드
model_name = "facebook/opt-350m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# corpus.json 불러오기
with open('corpus.json', 'r', encoding='utf-8') as f:
    corpus = json.load(f)

# 사용자 입력과 상담사의 응답을 추출하여 학습 데이터 준비
data = []
for i in range(len(corpus) - 1):
    if corpus[i]['role'] == 'user' and corpus[i + 1]['role'] == 'therapist':
        prompt = corpus[i]['content']
        response = corpus[i + 1]['content']
        data.append({"prompt": prompt, "response": response})

# 데이터셋을 Hugging Face Datasets 포맷으로 변환
dataset = Dataset.from_dict({"prompt": [d["prompt"] for d in data], "response": [d["response"] for d in data]})

# Train/Validation split (Hugging Face Datasets 메서드 사용)
dataset = dataset.train_test_split(test_size=0.2)
train_dataset = dataset['train']
val_dataset = dataset['test']

# 토크나이저로 데이터 전처리
def tokenize(batch):
    return tokenizer(batch["prompt"], padding="max_length", truncation=True, max_length=512)

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)

# 라벨 생성
def create_labels(batch):
    labels = tokenizer(batch["response"], padding="max_length", truncation=True, max_length=512)["input_ids"]
    batch["labels"] = labels
    return batch

train_dataset = train_dataset.map(create_labels, batched=True)
val_dataset = val_dataset.map(create_labels, batched=True)

# 필요한 컬럼만 남기기
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# 데이터 Collator 설정 (기본적인 DataCollatorForLanguageModeling 사용)
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Trainer 설정
training_args = TrainingArguments(
    output_dir="./output",
    overwrite_output_dir=True,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy="steps",
    eval_steps=500,
    logging_steps=100,
    save_steps=1000,
    save_total_limit=2,
    num_train_epochs=3,
    report_to="wandb"
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=collator,
)

# 학습 수행
trainer.train()

# 모델 및 결과 저장
trainer.save_model()

# 평가 결과 로깅
metrics = trainer.evaluate()
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)
