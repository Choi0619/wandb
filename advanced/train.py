import os
import json
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig
import wandb

# 1. Wandb 설정
wandb.init(project='Mind-Shelter')  # 프로젝트 이름 설정
wandb.run.name = 'sft-instruction-tuning'

# 2. 데이터 로드 및 Train/Validation Split
with open("corpus.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 데이터를 8:2 비율로 나눔
train_data, valid_data = train_test_split(data, test_size=0.2)

# Dataset 형태로 변환
train_dataset = Dataset.from_list(train_data)
valid_dataset = Dataset.from_list(valid_data)

# HuggingFace `DatasetDict`로 묶음
dataset = DatasetDict({
    'train': train_dataset,
    'validation': valid_dataset
})

# 3. 모델과 토크나이저 로드
model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

# 4. Formatting 및 Collator 설정
response_template = "### Answer:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

def formatting_prompts_func(example):
    formatted_texts = []
    for i in range(len(example['role'])):
        if example['role'][i] == "user":
            formatted_texts.append(f"### Question: {example['content'][i]}")
        else:
            formatted_texts.append(f"### Answer: {example['content'][i]}")
    return {"text": formatted_texts}

# 5. Trainer 설정
training_args = TrainingArguments(
    output_dir="/tmp/clm-instruction-tuning",
    evaluation_strategy="steps",  # 주기적으로 validation 실행
    per_device_train_batch_size=4,  # 배치 크기
    per_device_eval_batch_size=4,
    logging_steps=10,
    save_steps=10,
    save_total_limit=2,  # 저장할 체크포인트 수 제한
    num_train_epochs=3,
    report_to="wandb",  # wandb로 결과 보고
    load_best_model_at_end=True,
    logging_dir="/tmp/logs",
    eval_steps=50,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    args=training_args,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
)

# 6. 학습 시작
trainer.train()

# 7. 학습 후 모델 및 결과 저장
trainer.save_model("/tmp/clm-instruction-tuning-final")
wandb.finish()

# 평가 결과 로깅
metrics = trainer.state.log_history[-1]  # 마지막 로그에서 평가 결과 가져오기
trainer.log_metrics("final_metrics", metrics)
