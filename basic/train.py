import os
import sys
import math
import torch
import wandb
import logging
import datasets
import argparse
import evaluate
import transformers

from typing import Optional
from itertools import chain
from dataclasses import dataclass, field

from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    EarlyStoppingCallback,
    IntervalStrategy,
    WandbCallback  # Added WandbCallback
)
from transformers.trainer_utils import get_last_checkpoint

# Wandb 프로젝트 초기화
wandb.init(project='gyuhwan', entity='wrtyu0603-illinois-institute-of-technology')  # 프로젝트와 엔터티 이름 설정
wandb.run.name = 'gpt-finetuning'  # Wandb 실행 이름 설정

@dataclass
class Arguments:
    model_name_or_path: Optional[str] = field(default=None)
    torch_dtype: Optional[str] = field(default=None, metadata={'choices': ['auto', 'bfloat16', 'float16', 'float32']})
    dataset_name: Optional[str] = field(default=None)
    dataset_config_name: Optional[str] = field(default=None)
    block_size: int = field(default=1024)
    num_workers: Optional[int] = field(default=None)

parser = HfArgumentParser((Arguments, TrainingArguments))
args, training_args = parser.parse_args_into_dataclasses()

# 로깅 설정
logger = logging.getLogger()
logging.basicConfig(
    format="%(asctime)s - %(levellevelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

if training_args.should_log:
    transformers.utils.logging.set_verbosity_info()

log_level = training_args.get_process_log_level()
logger.setLevel(log_level)
datasets.utils.logging.set_verbosity(log_level)
transformers.utils.logging.set_verbosity(log_level)

logger.info(f"Training/evaluation parameters {training_args}")

# 데이터셋 로드
raw_datasets = load_dataset(
    args.dataset_name,
    args.dataset_config_name
)

# 모델과 토크나이저 로드
config = AutoConfig.from_pretrained(args.model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    config=config,
    torch_dtype=args.torch_dtype
)

# 토크나이저 및 모델 조정
tokenizer.pad_token_id = tokenizer.eos_token_id
embedding_size = model.get_input_embeddings().weight.shape[0]
if len(tokenizer) > embedding_size:
    model.resize_token_embeddings(len(tokenizer))

column_names = list(raw_datasets["train"].features)
text_column_name = "text" if "text" in column_names else column_names[0]

# 토크나이즈 함수 정의
def tokenize_function(examples):
    output = tokenizer(examples[text_column_name])
    return output

# 데이터셋 토크나이즈
with training_args.main_process_first(desc="dataset map tokenization"):
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=args.num_workers,
        remove_columns=column_names
    )

# 입력 길이에 맞게 블록 크기 조정
block_size = args.block_size if tokenizer.model_max_length is None else min(args.block_size, tokenizer.model_max_length)

# 텍스트 그룹화 함수 정의
def group_texts(examples):
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

# 텍스트를 그룹화하여 토크나이즈된 데이터셋 생성
with training_args.main_process_first(desc="grouping texts together"):
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=args.num_workers
    )

# 학습 및 평가용 데이터셋 분리
train_dataset = lm_datasets["train"]
eval_dataset = lm_datasets["validation"]

# Trainer 정의
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=default_data_collator,
    callbacks=[WandbCallback()]  # WandbCallback 추가하여 자동으로 Wandb에 로깅
)

# 체크포인트 설정
checkpoint = None
last_checkpoint = get_last_checkpoint(training_args.output_dir)
if training_args.resume_from_checkpoint is not None:
    checkpoint = training_args.resume_from_checkpoint
else:
    checkpoint = last_checkpoint

# 학습 및 평가 수행
train_result = trainer.train(resume_from_checkpoint=checkpoint)
trainer.save_model()

# 학습 및 평가 결과 Wandb에 기록
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)

# 평가 데이터셋으로 평가 실행
eval_metrics = trainer.evaluate()
trainer.log_metrics("eval", eval_metrics)
trainer.save_metrics("eval", eval_metrics)

trainer.save_state()

# Wandb 종료
wandb.finish()
