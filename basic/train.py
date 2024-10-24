import os
import torch
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
import json
from sklearn.model_selection import train_test_split

# Wandb 프로젝트 초기화
wandb.init(project='LLM_instruction_tuning', name='chatbot-finetuning')  # 프로젝트 및 실행 이름 설정

# GPT-2 모델과 토크나이저 불러오기
print("GPT-2 모델과 토크나이저를 로드하는 중입니다...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# pad_token 설정 (eos_token으로 설정)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained("gpt2", device_map="auto")
print("GPT-2 모델과 토크나이저가 성공적으로 로드되었습니다.")

# GPU 캐시 정리
torch.cuda.empty_cache()

# Gradient checkpointing 활성화 (메모리 절약)
model.gradient_checkpointing_enable()

# 데이터 로드
with open("corpus.json", "r", encoding="utf-8") as f:
    corpus = json.load(f)

print("JSON 파일이 성공적으로 로드되었습니다.")
print(f"데이터 예시: {corpus[0]}")

# 데이터를 질문과 답변의 쌍으로 변환
formatted_data = []
for i in range(0, len(corpus), 2):
    formatted_data.append({
        "instruction": corpus[i]["content"],  # 질문
        "response": corpus[i+1]["content"]    # 답변
    })

print(f"변환된 데이터 예시: {formatted_data[0]}")

# 데이터를 8:2로 나누어 train/validation dataset 만들기
train_data, valid_data = train_test_split(formatted_data, test_size=0.2)
print(f"Train 데이터 수: {len(train_data)}, Validation 데이터 수: {len(valid_data)}")

# train과 validation 데이터를 Dataset 객체로 변환
train_dataset = Dataset.from_dict({
    "instruction": [d["instruction"] for d in train_data],
    "response": [d["response"] for d in train_data]
})
valid_dataset = Dataset.from_dict({
    "instruction": [d["instruction"] for d in valid_data],
    "response": [d["response"] for d in valid_data]
})

print("Dataset 객체로 변환 성공.")
print(f"Dataset 예시: {train_dataset[0]}")

# Data formatting
def formatting_prompts_func(example):
    text = f"### Question: {example['instruction']}\n ### Answer: {example['response']}"
    return {
        "input_ids": tokenizer(text, padding="max_length", max_length=512, truncation=True)["input_ids"]
    }

# 데이터 콜레이터 정의 (답변 부분에만 Loss가 적용되도록)
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
        per_device_train_batch_size=4,  # 배치 크기를 늘림
        per_device_eval_batch_size=4,    # 평가 배치 크기도 늘림
        num_train_epochs=5,               # 에폭 수를 늘림
        logging_steps=10,
        gradient_accumulation_steps=4,    # Gradient Accumulation 적용
        fp16=True,                         # Mixed Precision 사용
    ),
    data_collator=collator,
)

# 학습 시작
print("SFT Trainer 설정 성공. 학습을 시작합니다.")
train_result = trainer.train()

# 모델 저장
trainer.save_model("./trained_model")

# 학습 및 평가 결과 로깅
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)

# 평가 데이터셋으로 평가 실행
eval_metrics = trainer.evaluate()
trainer.log_metrics("eval", eval_metrics)
trainer.save_metrics("eval", eval_metrics)

# Wandb 로그 종료
wandb.finish()

# GPU 캐시 정리
torch.cuda.empty_cache()
