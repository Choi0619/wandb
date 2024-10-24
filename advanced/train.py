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
for i in range(0, len(corpus)-1, 2):  # user와 therapist 쌍으로 진행
    if corpus[i]['role'] == 'user' and corpus[i+1]['role'] == 'therapist':
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
    inputs = examples['input']  # 사용자 입력에 접근
    outputs = examples['output']  # 치료사 응답에 접근
    
    # 토크나이저를 사용하여 입력과 출력 토큰화
    model_inputs = tokenizer(inputs, max_length=256, truncation=True, padding="max_length")

    # 라벨: 치료사 응답만 토크나이즈하여 라벨로 사용
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(outputs, max_length=256, truncation=True, padding="max_length").input_ids

    # <pad> 토큰 제거 (손실 함수에서 무시하도록 처리)
    model_inputs["labels"] = [[(label if label != tokenizer.pad_token_id else -100) for label in labels] for labels in labels]

    return model_inputs

# 전처리 적용
train_dataset = train_dataset.map(preprocess_function, batched=True)
val_dataset = val_dataset.map(preprocess_function, batched=True)

# 데이터 콜레이터 정의 (응답 템플릿 제거)
collator = DataCollatorForCompletionOnlyLM(tokenizer=tokenizer)

# SFT 설정 및 트레이너 정의
sft_config = SFTConfig(
    output_dir="./results",
    evaluation_strategy="epoch",  # 매 epoch마다 평가
    logging_strategy="steps",  # steps 단위로 로그 남기기
    logging_steps=100,  # 100 스텝마다 로깅
    eval_steps=500,  # 500 스텝마다 평가
    per_device_train_batch_size=8,  # 배치 크기 설정
    per_device_eval_batch_size=8,
    num_train_epochs=3,  # 에폭 수 3으로 설정
    save_total_limit=1,
    fp16=False  # FP16 비활성화
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    args=sft_config,
    data_collator=collator,
)

# 학습 시작
train_result = trainer.train()

# 모델 저장
trainer.save_model("./fine_tuned_therapist_chatbot")

# WandB에 train 결과 로깅
train_metrics = train_result.metrics
wandb.log({"train/loss": train_metrics['loss'], "train/epoch": train_metrics['epoch']})

# 평가 데이터셋으로 평가 실행
eval_metrics = trainer.evaluate()

# WandB에 eval 결과 로깅
wandb.log({"eval/loss": eval_metrics['eval_loss'], "eval/epoch": eval_metrics['epoch']})

# 학습 및 평가 결과 로깅
trainer.log_metrics("train", train_metrics)
trainer.save_metrics("train", train_metrics)

trainer.log_metrics("eval", eval_metrics)
trainer.save_metrics("eval", eval_metrics)

# WandB 로깅 종료
wandb.finish()
