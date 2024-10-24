import os
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch
import json
from datasets import Dataset

# .env 파일에서 환경 변수 불러오기
load_dotenv()

# .env 파일에서 Hugging Face API 토큰 가져오기
hf_token = os.getenv('HF_TOKEN')

# Hugging Face에 로그인
login(hf_token)

# 사용할 모델 이름 정의
model_name = "EleutherAI/gpt-neo-1.3B"

# 토크나이저와 모델 불러오기
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# Add padding token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
model.resize_token_embeddings(len(tokenizer))  # Resize embeddings to account for the new token

# 모델을 GPU로 이동 (가능한 경우)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# corpus.json 파일에서 데이터 로드
with open('corpus.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 데이터셋 준비 (preprocessing)
def preprocess_data(data):
    instructions, outputs = [], []
    for i in range(0, len(data), 2):  # 대화가 user와 therapist가 번갈아 나오는 것으로 가정
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

# Fine-tuning용 TrainingArguments 설정
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=5,  # 더 많은 epoch으로 설정
    evaluation_strategy="steps",
    eval_steps=100,
    logging_steps=100,
    save_steps=100,
    save_total_limit=2,
    remove_unused_columns=False,
    report_to=None,  # If you don't want to use wandb right now
    load_best_model_at_end=True
)

# 데이터셋에서 instruction과 output을 input_ids로 변환하는 함수
def tokenize_function(examples):
    inputs = tokenizer(examples['instruction'], padding="max_length", truncation=True, max_length=512)
    outputs = tokenizer(examples['output'], padding="max_length", truncation=True, max_length=512)
    inputs["labels"] = outputs["input_ids"]
    return inputs

# 데이터셋을 토크나이즈
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)

# Trainer 설정
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    tokenizer=tokenizer
)

# 모델 학습
trainer.train()

# 모델 저장
trainer.save_model("./fine_tuned_model")

# 샘플 프롬프트로 테스트
test_prompt = "너무 무기력한데 어떻게 해야할지 모르겠어."
inputs = tokenizer(test_prompt, return_tensors="pt").to(device)

# 모델 예측
outputs = model.generate(**inputs, max_new_tokens=50)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
