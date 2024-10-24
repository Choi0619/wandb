import os
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
import json
from sklearn.model_selection import train_test_split
import wandb

# wandb 초기화
wandb.init(project="LLM_instruction_tuning", entity="wrtyu0603")  # entity에 자신의 wandb 계정 이름 입력

# GPT-2 모델과 토크나이저 불러오기
print("GPT-2 모델과 토크나이저를 로드하는 중입니다...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# pad_token 설정 (eos_token으로 설정)
tokenizer.pad_token = tokenizer.eos_token

# 모델 로드 (GPU로 이동)
model = AutoModelForCausalLM.from_pretrained("gpt2").to('cuda')
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
    return {"input_ids": tokenizer(text, padding="max_length", max_length=512, truncation=True)["input_ids"]}

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
        report_to="wandb",  # wandb로 결과 보고
        save_strategy="steps",  # 모델 저장 주기 설정
        save_steps=100,  # 모델 저장할 step 수
    ),
    data_collator=collator,
)

# 학습 시작
print("SFT Trainer 설정 성공. 학습을 시작합니다.")
trainer.train()

# 모델 저장
trainer.save_model("./trained_model")

# 학습 종료 후 GPU 캐시 정리
torch.cuda.empty_cache()

# 샘플 데이터로 모델 테스트
def generate_answer(instruction):
    inputs = tokenizer(f"### Question: {instruction}\n ### Answer:", return_tensors="pt").input_ids.to('cuda')  # CUDA로 전송
    outputs = model.generate(inputs, max_length=100, pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 예시 질문
sample_question = "요즘 아무 이유 없이 눈물이 나요. 이런 기분을 어떻게 해결할 수 있을까요?"
generated_response = generate_answer(sample_question)
print(f"샘플 질문: {sample_question}")
print(f"모델의 답변: {generated_response}")
