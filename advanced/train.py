import json
import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset

# Wandb 설정
wandb.init(project="LLM_instruction_tuning", name="gpt2-medium-instruction-tuning")

# 모델과 토크나이저 불러오기
model_name = "gpt2-medium"  # GPT2-Medium 모델 사용
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 패딩 토큰 추가 (GPT-2 모델의 경우 기본적으로 패딩 토큰이 없기 때문에 추가)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer))

# 데이터 로드
with open('corpus.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 데이터 전처리
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

# 토크나이즈 함수 정의
def tokenize_function(examples):
    # 각 텍스트가 다차원이 아닌지 확인
    inputs = [text if isinstance(text, str) else " ".join(text) for text in examples['instruction']]
    outputs = [text if isinstance(text, str) else " ".join(text) for text in examples['output']]
    
    # 토큰화
    inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=512)
    outputs = tokenizer(outputs, padding="max_length", truncation=True, max_length=512)
    
    # 라벨을 output의 input_ids로 설정
    inputs["labels"] = outputs["input_ids"]
    return inputs

# 데이터셋을 토크나이즈
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)

# TrainingArguments 설정
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,  # 에포크 수는 조절 가능
    logging_steps=100,
    evaluation_strategy="steps",
    eval_steps=100,
    save_steps=500,
    save_total_limit=2,
    remove_unused_columns=False,
    report_to="wandb",  # Wandb에 로그 기록
    load_best_model_at_end=True
)

# Trainer 설정
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    tokenizer=tokenizer
)

# 학습 실행
trainer.train()

# 평가 실행
eval_results = trainer.evaluate()

# Wandb 종료
wandb.finish()
