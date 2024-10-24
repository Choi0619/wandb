import os
import json
import wandb
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from sklearn.model_selection import train_test_split

# Wandb 초기화
wandb.init(project='maum_shelter')  # 프로젝트 이름 설정
wandb.run.name = 'gpt-instruction-tuning'  # wandb run 이름 설정

# 모델과 토크나이저 로드
model_name = "facebook/opt-350m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# corpus.json 불러오기
with open('corpus.json', 'r', encoding='utf-8') as f:
    corpus = json.load(f)

# Instruction tuning에 맞게 데이터셋을 구성
data = []
for i in range(0, len(corpus), 2):
    if i + 1 < len(corpus):
        prompt = f"### User: {corpus[i]['content']}\n### Therapist: {corpus[i+1]['content']}"
        data.append({"prompt": prompt})

# 데이터셋을 Hugging Face Datasets 포맷으로 변환
dataset = Dataset.from_dict({"text": [d["prompt"] for d in data]})

# Train/Validation split (8:2 비율로 나누기)
train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, shuffle=True)

# 데이터 Collator (Completion 전용)
collator = DataCollatorForCompletionOnlyLM(tokenizer=tokenizer)

# Trainer 설정
training_args = TrainingArguments(
    output_dir="./output",
    overwrite_output_dir=True,
    per_device_train_batch_size=4,  # GPU 메모리에 맞게 설정하세요
    per_device_eval_batch_size=4,
    evaluation_strategy="steps",  # 몇 스텝마다 평가를 수행할지 설정
    eval_steps=500,  # 500 스텝마다 평가
    logging_steps=100,  # 100 스텝마다 로깅
    save_steps=1000,  # 1000 스텝마다 체크포인트 저장
    save_total_limit=2,  # 체크포인트 개수 제한
    num_train_epochs=3,  # 학습 에포크 수
    report_to="wandb"  # wandb로 로깅
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=collator,
    formatting_func=lambda x: x["text"],  # 데이터셋에서 텍스트를 가져오는 함수
)

# 학습 수행
trainer.train()

# 모델 및 결과 저장
trainer.save_model()

# wandb에서 로깅
metrics = trainer.evaluate()
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)
