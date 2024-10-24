import json
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig

# Step 1: 데이터 로드 및 분리
with open("corpus.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 사용자 질문과 상담사 답변 쌍으로 분리
dialogs = []
dialog = {"instruction": "", "answer": ""}

for i in range(0, len(data), 2):  # 2개씩 묶음
    if data[i]["role"] == "user" and data[i+1]["role"] == "therapist":
        dialog["instruction"] = data[i]["content"]  # 사용자 질문
        dialog["answer"] = data[i+1]["content"]  # 상담사 답변
        dialogs.append(dialog.copy())

# 8:2 비율로 Train/Validation 데이터 나누기
train_data, val_data = train_test_split(dialogs, test_size=0.2)

# Step 2: 포맷팅 함수 정의
def formatting_prompts_func(example):
    output_texts = []
    for data in example:
        text = f"### Question: {data['instruction']}\n ### Answer: {data['answer']}"
        output_texts.append(text)
    return output_texts

# Step 3: 모델과 토크나이저 로드
model = AutoModelForCausalLM.from_pretrained("gpt2")  # gpt2 모델 사용 예시
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 답변 구분자 설정
response_template = " ### Answer:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

# Step 4: 학습 설정
training_args = SFTConfig(
    output_dir="./sft_results",  # 결과 저장 경로
    evaluation_strategy="steps",  # 평가 주기
    eval_steps=500,  # 평가 간격
    logging_steps=500,  # 로그 간격
    save_steps=1000,  # 모델 저장 간격
    num_train_epochs=3,  # 학습 epoch 수
    per_device_train_batch_size=4,  # 훈련 배치 크기
    per_device_eval_batch_size=4  # 평가 배치 크기
)

# Step 5: Trainer 정의 및 학습 시작
trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=training_args,
    formatting_func=formatting_prompts_func,  # 데이터 포맷팅 함수
    data_collator=collator  # 데이터 구분자 설정
)

# 학습 시작
trainer.train()

# Step 6: 학습된 모델로 테스트
model_path = './sft_results'
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 테스트용 질문
test_input = "요즘 불안해서 잠을 잘 못 자요. 어떻게 해야 할까요?"

# 토크나이즈 및 답변 생성
input_ids = tokenizer(test_input, return_tensors="pt").input_ids
output = model.generate(input_ids=input_ids, max_length=100)
decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

print(f"챗봇 답변: {decoded_output}")
