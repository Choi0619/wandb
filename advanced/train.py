import os
import json
import wandb
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

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

# Train/Validation split (Hugging Face의 train_test_split 사용)
dataset = dataset.train_test_split(test_size=0.2, shuffle=True)
train_dataset = dataset["train"]
val_dataset = dataset["test"]

# 데이터 Collator (Completion 전용)
collator = DataCollatorForCompletionOnlyLM(tokenizer=tokenizer)

# 텍스트 포맷 함수 정의 (프롬프트와 응답을 합침)
def formatting_prompts_func(example):
    return example['prompt'] + example['response']

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
    formatting_func=formatting_prompts_func,
)

# 학습 수행
trainer.train()

# 모델 및 결과 저장
trainer.save_model()

# 평가 결과 로깅
metrics = trainer.evaluate()
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)
