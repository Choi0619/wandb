import os
import json
import wandb
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

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
for i in range(len(corpus) - 1):
    if corpus[i]['role'] == 'user' and corpus[i + 1]['role'] == 'therapist':
        prompt = f"### User: {corpus[i]['content']}\n"
        answer = f"### Therapist: {corpus[i + 1]['content']}"
        data.append({"prompt": prompt, "response": answer})

# 데이터셋을 Hugging Face Datasets 포맷으로 변환
dataset = Dataset.from_dict({"prompt": [d["prompt"] for d in data], "response": [d["response"] for d in data]})

# Train/Validation split (Hugging Face의 train_test_split 사용)
dataset = dataset.train_test_split(test_size=0.2, shuffle=True)
train_dataset = dataset["train"]
val_dataset = dataset["test"]

# 템플릿 설정 (Therapist 답변을 기준으로 함)
response_template = "### Therapist:"

# 데이터 Collator (Completion 전용)
collator = DataCollatorForCompletionOnlyLM(tokenizer=tokenizer, response_template=response_template)

# 텍스트 포맷 함수 정의 (프롬프트와 응답을 합침)
def formatting_prompts_func(example):
    return example['prompt'] + example['response']

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
    formatting_func=formatting_prompts_func,  # 프롬프트와 응답을 함께 처리하는 함수
)

# 학습 수행
trainer.train()

# 모델 및 결과 저장
trainer.save_model()

# wandb에서 로깅
metrics = trainer.evaluate()
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)
