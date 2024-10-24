import os
import json
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from sklearn.model_selection import train_test_split

# .env 파일에서 환경 변수 로드
load_dotenv()

# 환경 변수에서 HF_TOKEN 불러오기
hf_token = os.getenv("HF_TOKEN")

# Hugging Face 로그인
login(hf_token)

# gemma-2b 모델과 토크나이저 불러오기
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b", device_map="auto")

# 데이터 로드
with open("corpus.json", "r", encoding="utf-8") as f:
    corpus = json.load(f)

# 데이터를 user(질문)과 therapist(답변)의 쌍으로 변환
formatted_data = []
for i in range(0, len(corpus), 2):
    if corpus[i]["role"] == "user" and corpus[i + 1]["role"] == "therapist":
        formatted_data.append({
            "instruction": corpus[i]["content"],  # user 질문
            "response": corpus[i + 1]["content"]  # therapist 답변
        })

# Train/Validation 데이터를 8:2 비율로 나누기
train_data, valid_data = train_test_split(formatted_data, test_size=0.2)

# Dataset 객체로 변환
train_dataset = Dataset.from_dict({
    "instruction": [d["instruction"] for d in train_data],
    "response": [d["response"] for d in train_data]
})
valid_dataset = Dataset.from_dict({
    "instruction": [d["instruction"] for d in valid_data],
    "response": [d["response"] for d in valid_data]
})

# 데이터 포맷팅 함수 정의
def formatting_prompts_func(example):
    # 질문과 답변을 텍스트로 합침
    text = f"### Question: {example['instruction']}\n ### Answer: {example['response']}"
    return {
        "input_ids": tokenizer(text, padding="max_length", max_length=1024, truncation=True)["input_ids"]
    }

# 데이터 콜레이터 정의 (답변 부분에만 Loss가 적용되도록)
response_template = " ### Answer:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

# SFT Trainer 설정
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    args=SFTConfig(
        output_dir="./results",
        evaluation_strategy="steps",
        eval_steps=100,          # 몇 스텝마다 validation할지
        per_device_train_batch_size=4,  # 배치 사이즈
        per_device_eval_batch_size=4,
        num_train_epochs=3,      # 학습할 epoch 수
        logging_steps=10,        # 로그 기록 빈도
    ),
    formatting_func=formatting_prompts_func,
    data_collator=collator,
)

# 학습 시작
trainer.train()

# 모델 저장
trainer.save_model("./trained_model")
