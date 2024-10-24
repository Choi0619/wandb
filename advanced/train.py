import json
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding
from transformers import AdamW, get_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import wandb

# WandB 초기화
wandb.init(project="therapist-chatbot", name="fine-tuning")

# corpus.json 데이터 로드
with open('corpus.json', 'r', encoding='utf-8') as f:
    corpus = json.load(f)

# 입력-출력 쌍 준비
data_pairs = []
for i in range(0, len(corpus) - 1, 2):  # user와 therapist 쌍으로 진행
    if corpus[i]['role'] == 'user' and corpus[i + 1]['role'] == 'therapist':
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
    # padding=True, truncation=True 옵션을 추가하여 데이터 길이를 맞추어 줍니다.
    inputs = tokenizer(examples['input'], max_length=256, truncation=True, padding="max_length")
    labels = tokenizer(text_target=examples['output'], max_length=256, truncation=True, padding="max_length").input_ids
    
    # <pad> 토큰을 -100으로 설정하여 손실 계산에서 제외
    labels = [[(label if label != tokenizer.pad_token_id else -100) for label in label_list] for label_list in labels]
    
    inputs["labels"] = labels
    return inputs

# 전처리 적용
train_dataset = train_dataset.map(preprocess_function, batched=True)
val_dataset = val_dataset.map(preprocess_function, batched=True)

# DataLoader 준비
collator = DataCollatorWithPadding(tokenizer=tokenizer)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8, collate_fn=collator)
eval_dataloader = DataLoader(val_dataset, batch_size=8, collate_fn=collator)

# 옵티마이저와 스케줄러 설정
optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

# 모델을 GPU로 이동
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 학습 루프
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        # 매 스텝마다 train loss 로그
        train_loss += loss.item()
        wandb.log({"train/loss": loss.item(), "step": step + epoch * len(train_dataloader)})

    # 에폭 당 평균 train loss 계산
    avg_train_loss = train_loss / len(train_dataloader)

    # 평가 루프
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            eval_loss += outputs.loss.item()

    avg_eval_loss = eval_loss / len(eval_dataloader)

    # WandB에 eval loss 및 train loss 기록
    wandb.log({"eval/loss": avg_eval_loss, "train/epoch": epoch, "eval/epoch": epoch})

    print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss}, Eval Loss = {avg_eval_loss}")

# 모델 저장
model.save_pretrained("./fine_tuned_therapist_chatbot")
tokenizer.save_pretrained("./fine_tuned_therapist_chatbot")

# WandB 로깅 종료
wandb.finish()
