import json
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
import wandb

# WandB 초기화
wandb.init(project="therapist-chatbot", name="fine-tuning")

# corpus.json 데이터 로드
with open('corpus.json', 'r', encoding='utf-8') as f:
    corpus = json.load(f)

# 입력-출력 쌍 준비
data_pairs = []
for i in range(0, len(corpus), 2):
    input_text = corpus[i]['content']  # 사용자 입력
    output_text = corpus[i + 1]['content']  # 치료사 응답
    data_pairs.append({"input": input_text, "output": output_text})

# 학습 및 검증 세트로 분할 (80-20 비율)
train_data, val_data = train_test_split(data_pairs, test_size=0.2, random_state=42)

# Hugging Face 데이터셋으로 변환
train_dataset = Dataset.from_pandas(pd.DataFrame(train_data))
val_dataset = Dataset.from_pandas(pd.DataFrame(val_data))

# 모델과 토크나이저 로드
model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

# 전처리 함수 정의
def preprocess_function(examples):
    inputs = examples['input']  # 입력 필드에 직접 접근
    outputs = examples['output']  # 출력 필드에 직접 접근
    
    # 토크나이저를 사용하여 입력과 출력 토큰화
    model_inputs = tokenizer(inputs, max_length=256, truncation=True, padding="max_length")

    labels = tokenizer(outputs, max_length=256, truncation=True, padding="max_length").input_ids
    model_inputs["labels"] = labels  # 라벨 추가
    return model_inputs

# 전처리 적용
train_dataset = train_dataset.map(preprocess_function, batched=True)
val_dataset = val_dataset.map(preprocess_function, batched=True)

# 응답 템플릿 및 데이터 콜레이터 정의
response_template = " ### Answer:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

# SFT 설정 및 트레이너 정의
sft_config = SFTConfig(
    output_dir="./results",
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    per_device_train_batch_size=1,  # 배치 크기 줄이기
    per_device_eval_batch_size=1,    # 배치 크기 줄이기
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    args=sft_config,
    data_collator=collator,
)

# 학습 시작
trainer.train()

# 모델 저장
trainer.save_model("./fine_tuned_therapist_chatbot")

# WandB 로깅 종료
wandb.finish()
