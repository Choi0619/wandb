import os
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
import json
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments

# .env 파일에서 환경 변수 로드
load_dotenv()

# 모델과 토크나이저 불러오기 (HF_TOKEN 생략)
print("모델과 토크나이저를 로드하는 중입니다...")
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")  # 가벼운 모델 사용
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
print("모델과 토크나이저가 성공적으로 로드되었습니다.")

# GPU 캐시 정리
torch.cuda.empty_cache()

# Gradient checkpointing을 활성화하여 메모리 절약
model.gradient_checkpointing_enable()

# 데이터 로드
with open("corpus.json", "r", encoding="utf-8") as f:
    corpus = json.load(f)

print("JSON 파일이 성공적으로 로드되었습니다.")
print(f"데이터 예시: {corpus[0]}")

# 데이터를 질문과 답변의 쌍으로 형식화
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
    return {"input_ids": tokenizer(text, padding="max_length", max_length=512, truncation=True)["input_ids"]}

# 데이터 콜레이터 정의 (답변 부분에만 Loss가 적용되도록)
response_template = " ### Answer:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

# 학습 설정
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    eval_steps=100,
    per_device_train_batch_size=4,  # 배치 크기 늘림
    per_device_eval_batch_size=4,   # 평가 배치 크기도 늘림
    num_train_epochs=3,
    logging_steps=10,
    gradient_accumulation_steps=2,  # Gradient Accumulation
    fp16=True,  # Mixed Precision
    save_steps=500,
    save_total_limit=2,
)

# Trainer 설정
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset.map(formatting_prompts_func),
    eval_dataset=valid_dataset.map(formatting_prompts_func),
    data_collator=collator,
)

# 학습 및 평가 수행
train_result = trainer.train()
trainer.save_model()

# 학습 및 평가 결과 로깅
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)

# 평가 데이터셋으로 평가 실행
eval_metrics = trainer.evaluate()
trainer.log_metrics("eval", eval_metrics)
trainer.save_metrics("eval", eval_metrics)

trainer.save_state()

# 샘플 테스트 출력
sample_input = "어떻게 하면 더 집중을 잘할 수 있을까요?"
inputs = tokenizer(f"### Question: {sample_input}\n ### Answer:", return_tensors="pt")
output = model.generate(**inputs)
print(f"샘플 입력: {sample_input}")
print(f"모델 출력: {tokenizer.decode(output[0], skip_special_tokens=True)}")

# GPU 캐시 정리
torch.cuda.empty_cache()
