import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
import json
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

# 사용할 모델 정의
model_name = "EleutherAI/gpt-neo-1.3B"

# 모델과 토크나이저 불러오기
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# Add padding token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
model.resize_token_embeddings(len(tokenizer))  # Resize embeddings to account for the new token

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
    num_train_epochs=3,  # 적절히 늘릴 수 있음
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    logging_steps=100,
    evaluation_strategy="steps",
    eval_steps=100,
    save_steps=100,
    max_seq_length=512,  # 설정 추가
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

# 샘플 프롬프트로 테스트
test_prompt = "너무 무기력한데 어떻게 해야할지 모르겠어."
inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

# 모델 예측
outputs = model.generate(**inputs, max_new_tokens=50)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
