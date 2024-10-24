import json
import torch
import wandb
import logging
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM

# Wandb 프로젝트 초기화
wandb.init(project='LLM_instruction_tuning')
wandb.run.name = 'gpt2-instruction-tuning'

# 로깅 설정
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# GPT-2 모델과 토크나이저 로드
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 데이터 로드 (corpus.json 파일에서 직접 로드)
with open('corpus.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 데이터셋 준비
def preprocess_data(data):
    instructions, outputs = [], []
    for i in range(0, len(data), 2):  # user와 therapist가 번갈아 나오는 구조라 가정
        if data[i]['role'] == 'user' and data[i+1]['role'] == 'therapist':
            instructions.append(data[i]['content'])
            outputs.append(data[i+1]['content'])
    return instructions, outputs

instructions, outputs = preprocess_data(data)

# 데이터셋을 HuggingFace Dataset 형태로 변환
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
    return f"### Question: {example['instruction']}\n ### Answer: {example['output']}"

# Data Collator 정의 (SFT용)
response_template = " ### Answer:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

# SFTTrainer 설정
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # 평가 데이터 추가
    args=SFTConfig(output_dir="./output", num_train_epochs=3, per_device_train_batch_size=2),
    formatting_func=formatting_prompts_func,
    data_collator=collator,
)

# 학습 실행
trainer.train()

# Wandb 학습 종료
wandb.finish()
