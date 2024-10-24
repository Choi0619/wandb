import json
import torch
import wandb
import logging
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
import argparse

# 명령어 인자 설정
parser = argparse.ArgumentParser(description="Fine-tuning GPT-2 model")
parser.add_argument('--output_dir', type=str, required=True, help='Model output directory')
args = parser.parse_args()

# Wandb 프로젝트 초기화 (sync_tensorboard는 설정 X)
wandb.init(project='LLM_instruction_tuning', name='gpt2-instruction-tuning')

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

# Add padding token for GPT-2
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model = AutoModelForCausalLM.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))  # resize the token embeddings to include pad_token

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

# 데이터 포맷팅 함수 정의 (리스트 반환)
def formatting_prompts_func(example):
    return [f"### Question: {example['instruction']}\n ### Answer: {example['output']}"]

# Data Collator 정의 (SFT용)
response_template = " ### Answer:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

# SFTTrainer 설정
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # 평가 데이터 추가
    args=SFTConfig(
        output_dir=args.output_dir, 
        num_train_epochs=3, 
        per_device_train_batch_size=2, 
        logging_steps=100, 
        evaluation_strategy="steps", 
        eval_steps=100
    ),
    formatting_func=formatting_prompts_func,
    data_collator=collator,
)

# 학습 실행
train_result = trainer.train()

# 학습 중간 및 최종 결과 Wandb에 기록 (매 스텝마다 기록)
for log in trainer.state.log_history:
    if 'loss' in log:
        wandb.log({"train_loss": log['loss']})
    if 'eval_loss' in log:
        wandb.log({"eval_loss": log['eval_loss']})

# 모델 저장
trainer.save_model(args.output_dir)

# 평가 실행 및 결과 로그
eval_result = trainer.evaluate()
wandb.log({"eval_loss": eval_result['eval_loss']})  # log eval loss

# Wandb 학습 종료
wandb.finish()
