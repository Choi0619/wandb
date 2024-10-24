import os
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
import torch
import json
from datasets import Dataset

# .env 파일에서 환경 변수 불러오기
load_dotenv()

# Hugging Face API 토큰 불러오기
hf_token = os.getenv('HF_TOKEN')

# Hugging Face에 로그인
login(hf_token)

# 사용할 모델 이름 정의
model_name = "google/gemma-2b"

# 토크나이저와 모델 불러오기
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
model.gradient_checkpointing_enable()  # Gradient checkpointing 사용으로 메모리 최적화

# 모델을 GPU로 이동
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# corpus.json 파일에서 데이터 로드
with open('corpus.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 데이터셋 준비
def preprocess_data(data):
    instructions, outputs = [], []
    for i in range(0, len(data), 2):  # user와 therapist가 번갈아 나오는 구조로 가정
        if data[i]['role'] == 'user' and data[i+1]['role'] == 'therapist':
            instructions.append(data[i]['content'])
            outputs.append(data[i+1]['content'])
    return instructions, outputs

instructions, outputs = preprocess_data(data)

# HuggingFace Dataset 형식으로 변환
dataset = Dataset.from_dict({
    "instruction": instructions,
    "output": outputs
})

# Train, Validation 데이터셋 나누기 (8:2 비율)
train_test_split = dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

# 데이터 포맷팅 함수 정의 (리스트 반환)
def formatting_prompts_func(example):
    return [f"### Question: {example['instruction']}\n ### Answer: {example['output']}"]

# Data Collator 정의
response_template = " ### Answer:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

# Fine-tuning 설정
config = SFTConfig(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=1,
    max_seq_length=512,  # max_seq_length 설정
    gradient_accumulation_steps=4,
    logging_steps=50,
    evaluation_strategy="steps",
    eval_steps=50,
    save_steps=100,
    fp16=True
)


# SFTTrainer 설정
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=config,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
)

# 학습 실행
trainer.train()

# 모델 저장
trainer.save_model("./fine_tuned_model")

# 테스트: 샘플 프롬프트로 모델 결과 확인
test_prompt = "너무 무기력한데 어떻게 해야할지 모르겠어."
inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("모델 응답:", response)
