import os
import torch
import wandb
import json
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from sklearn.model_selection import train_test_split
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM

# .env 파일에서 환경 변수 로드
load_dotenv()

# Wandb 프로젝트 초기화 - 'wrtyu0603'은 너의 실제 Wandb 계정 이름
wandb.init(project='gyuhwan', entity='wrtyu0603')  # 프로젝트 이름을 'gyuhwan'으로 설정
wandb.run.name = 'chatbot-finetuning'  # Wandb 실행 이름 설정

# 모델 및 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")  # 경량 모델 사용
model = AutoModelForCausalLM.from_pretrained("distilgpt2", device_map="auto")

# 데이터 로드
with open("corpus.json", "r", encoding="utf-8") as f:
    corpus = json.load(f)

# 데이터를 instruction-response 형식으로 변환
formatted_data = []
for i in range(0, len(corpus), 2):
    formatted_data.append({
        "instruction": corpus[i]["content"],
        "response": corpus[i+1]["content"]
    })

# Train/Validation Split (8:2)
train_data, valid_data = train_test_split(formatted_data, test_size=0.2)

# 데이터셋으로 변환
train_dataset = Dataset.from_dict({
    "instruction": [d["instruction"] for d in train_data],
    "response": [d["response"] for d in train_data]
})
valid_dataset = Dataset.from_dict({
    "instruction": [d["instruction"] for d in valid_data],
    "response": [d["response"] for d in valid_data]
})

# 데이터 포맷팅
def formatting_prompts_func(example):
    text = f"### Question: {example['instruction']}\n ### Answer: {example['response']}"
    return {"input_ids": tokenizer(text, padding="max_length", max_length=512, truncation=True)["input_ids"]}

# 데이터 콜레이터 정의 (답변 부분에만 Loss 적용)
response_template = " ### Answer:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

# SFT Trainer 설정
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset.map(formatting_prompts_func),
    eval_dataset=valid_dataset.map(formatting_prompts_func),
    args=SFTConfig(
        output_dir="./results",
        evaluation_strategy="steps",
        eval_steps=100,
        per_device_train_batch_size=4,  # 배치 크기를 줄임
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        logging_steps=10,
        gradient_accumulation_steps=4,  # Gradient Accumulation 적용
        fp16=True,  # Mixed Precision 사용
    ),
    data_collator=collator,
)

# 학습 및 평가
train_result = trainer.train()
trainer.save_model()

# 학습 결과 로깅
train_metrics = train_result.metrics
trainer.log_metrics("train", train_metrics)
trainer.save_metrics("train", train_metrics)

# 평가 수행 및 로깅
eval_metrics = trainer.evaluate()
trainer.log_metrics("eval", eval_metrics)
trainer.save_metrics("eval", eval_metrics)

# Wandb 종료
wandb.finish()

# GPU 캐시 정리
torch.cuda.empty_cache()
