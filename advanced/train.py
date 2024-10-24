import os
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, Dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
import json
from sklearn.model_selection import train_test_split

# 캐시 정리 및 메모리 상태 확인 함수 추가
def clear_gpu_memory():
    print("GPU 캐시를 정리하고 메모리 상태를 확인합니다...")
    torch.cuda.empty_cache()
    torch.cuda.memory_summary(device=None, abbreviated=False)

# GPU 캐시 정리 및 메모리 상태 확인
clear_gpu_memory()

# .env 파일에서 환경 변수 로드
load_dotenv()

# 환경 변수에서 HF_TOKEN 불러오기
hf_token = os.getenv("HF_TOKEN")

# Hugging Face 로그인
login(hf_token)

# gemma-2b 모델과 토크나이저 불러오기
print("Gemma 2B 모델과 토크나이저를 로드하는 중입니다...")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b", device_map="auto")
print("Gemma 2B 모델과 토크나이저가 성공적으로 로드되었습니다.")

# GPU 캐시 다시 정리
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
    return {"input_ids": tokenizer(text, padding="max_length", max_length=1024, truncation=True)["input_ids"]}

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
        per_device_train_batch_size=1,  # 배치 크기를 줄임
        per_device_eval_batch_size=1,   # 평가 배치 크기도 줄임
        num_train_epochs=3,
        logging_steps=10,
        gradient_accumulation_steps=4,  # Gradient Accumulation 적용
        fp16=True,  # Mixed Precision 사용
    ),
    data_collator=collator,
)

# 학습 시작
print("SFT Trainer 설정 성공. 학습을 시작합니다.")
trainer.train()

# 모델 저장
trainer.save_model("./trained_model")

# 학습 후 GPU 캐시 다시 정리
clear_gpu_memory()
