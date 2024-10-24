import json
from sklearn.model_selection import train_test_split

# corpus.json 파일 로드
with open("corpus.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# role에 따라 데이터를 분리하여 질문-답변 쌍 만들기
dialogs = []
dialog = {"instruction": "", "answer": ""}

for i in range(0, len(data), 2):  # 2개씩 가져와서 사용자-상담사 쌍을 만듦
    if data[i]["role"] == "user" and data[i+1]["role"] == "therapist":
        dialog["instruction"] = data[i]["content"]  # 질문
        dialog["answer"] = data[i+1]["content"]  # 답변
        dialogs.append(dialog.copy())

# 데이터를 8:2 비율로 train과 validation으로 분할
train_data, val_data = train_test_split(dialogs, test_size=0.2)
