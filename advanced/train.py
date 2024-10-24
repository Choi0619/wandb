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
    for entry in data:
        if entry['role'] == 'user':
            instruction = entry['content']
        else:
            output = entry['content']
            instructions.append(instruction)
            outputs.append(output)
    return instructions, outputs

instructions, outputs = preprocess_data(data)

# 데이터셋을 HuggingFace Dataset 형태로 변환
dataset = Dataset.from_dict({
    "instruction": instructions,
    "output": outputs
})

# 데이터 포맷팅 함수 정의
def formatting_prompts_func(example):
    return f"### Question: {example['instruction']}\n ### Answer: {example['output']}"

# Data Collator 정의 (SFT용)
response_template = " ### Answer:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

# SFTTrainer 설정
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=SFTConfig(output_dir="./output"),
    formatting_func=formatting_prompts_func,
    data_collator=collator,
)

# 학습 실행
trainer.train()

# Wandb 학습 종료
wandb.finish()
