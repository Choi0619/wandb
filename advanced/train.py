import torch
import wandb
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

# Wandb 프로젝트 초기화
wandb.init(project="LLM_instruction_tuning", name="gpt2-medium-instruction-tuning")

# 사용할 모델 정의 (GPT-2 Medium)
model_name = "gpt2-medium"

# 모델과 토크나이저 불러오기
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# Add padding token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
model.resize_token_embeddings(len(tokenizer))

# 데이터 로드 (corpus.json 파일에서)
with open('corpus.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 데이터셋 준비 (user의 발언은 input, therapist의 발언은 output으로 처리)
def preprocess_data(data):
    instructions, outputs = [], []
    for i in range(0, len(data), 2):
        if data[i]['role'] == 'user' and data[i+1]['role'] == 'therapist':
            instructions.append(data[i]['content'])
            outputs.append(data[i+1]['content'])
    return instructions, outputs

instructions, outputs = preprocess_data(data)

# HuggingFace Dataset으로 변환
dataset = Dataset.from_dict({
    "instruction": instructions,
    "output": outputs
})

# Train, Validation 데이터셋 나누기 (8:2 비율)
train_test_split = dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

# 데이터 포맷팅 함수 정의
def formatting_prompts_func(example):
    return [f"### Question: {example['instruction']}\n ### Answer: {example['output']}"]

# Data Collator 정의 (Completion Only)
response_template = " ### Answer:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

# SFTConfig 설정
config = SFTConfig(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=2,  # Batch size of 2 to manage memory
    per_device_eval_batch_size=2,
    logging_steps=100,
    evaluation_strategy="steps",
    eval_steps=100,
    save_steps=100,
    max_seq_length=512,  # Adjust sequence length
    gradient_accumulation_steps=4,  # Accumulate gradients
    fp16=True  # Mixed precision training for memory efficiency
)

# SFTTrainer 설정
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=config,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
    tokenizer=tokenizer,
)

# 학습 실행
trainer.train()

# 모델 저장
trainer.save_model("./fine_tuned_model")

# 학습 중간 및 최종 결과 Wandb에 기록
wandb.watch(model)

# 학습 결과 및 평가 로깅
train_result = trainer.train()
wandb.log({"train_loss": train_result.training_loss})

# 평가 실행
eval_result = trainer.evaluate()
wandb.log({"eval_loss": eval_result['eval_loss']})

# Wandb 종료
wandb.finish()
