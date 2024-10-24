import os
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, Dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
import json
from sklearn.model_selection import train_test_split

# 1. .env 파일에서 HF_TOKEN 환경 변수 로드
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

# 2. Hugging Face에 로그인
login(hf_token)

# 3. Gemma-2B 모델과 토크나이저 불러오기
try:
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
    model = AutoModelForCausalLM.from_pretrained("google/gemma-2b", device_map="auto")
    print("Gemma 2B 모델과 토크나이저가 성공적으로 로드되었습니다.")
except Exception as e:
    print(f"모델 로딩 중 오류 발생: {e}")

# 4. corpus.json 파일 로드 및 구조 확인
file_path = "corpus.json"
try:
    with open(file_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)
        print("JSON 파일이 성공적으로 로드되었습니다.")
        print(f"데이터 예시: {corpus[0]}")
except Exception as e:
    print(f"JSON 파일 로드 중 오류 발생: {e}")

# 5. 질문과 답변 쌍으로 데이터를 변환
formatted_data = []
try:
    for i in range(0, len(corpus), 2):
        if corpus[i]["role"] == "user" and corpus[i + 1]["role"] == "therapist":
            formatted_data.append({
                "instruction": corpus[i]["content"],
                "response": corpus[i + 1]["content"]
            })
    print(f"변환된 데이터 예시: {formatted_data[0]}")
except Exception as e:
    print(f"데이터 변환 중 오류 발생: {e}")

# 6. 데이터를 8:2로 나누어 train과 validation 데이터셋 생성
try:
    train_data, valid_data = train_test_split(formatted_data, test_size=0.2)
    print(f"Train 데이터 수: {len(train_data)}, Validation 데이터 수: {len(valid_data)}")
except Exception as e:
    print(f"데이터 분할 중 오류 발생: {e}")

# 7. Dataset 객체로 변환
try:
    train_dataset = Dataset.from_dict({
        "instruction": [d["instruction"] for d in train_data],
        "response": [d["response"] for d in train_data]
    })
    valid_dataset = Dataset.from_dict({
        "instruction": [d["instruction"] for d in valid_data],
        "response": [d["response"] for d in valid_data]
    })
    print("Dataset 객체로 변환 성공.")
    print(f"Dataset 예시: {train_dataset[0]}")
except Exception as e:
    print(f"Dataset 변환 중 오류 발생: {e}")

# 8. 데이터 포맷팅 및 토크나이징 수정
def formatting_prompts_func(example):
    text = f"### Question: {example['instruction']}\n ### Answer: {example['response']}"
    # input_ids와 attention_mask를 반환하여 tokenizer가 올바르게 동작하도록 수정
    tokenized = tokenizer(text, padding="max_length", max_length=1024, truncation=True)
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"]
    }

# 9. 데이터 콜레이터 정의
response_template = " ### Answer:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

# 10. SFT Trainer 설정 및 학습
try:
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        args=SFTConfig(
            output_dir="./results",
            evaluation_strategy="steps",
            eval_steps=100,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=3,
            logging_steps=10,
        ),
        formatting_func=formatting_prompts_func,
        data_collator=collator,
    )
    print("SFT Trainer 설정 성공. 학습을 시작합니다.")
    trainer.train()
    
    # 11. 모델 저장
    trainer.save_model("./trained_model")
    print("모델이 성공적으로 저장되었습니다.")
except Exception as e:
    print(f"학습 중 오류 발생: {e}")
