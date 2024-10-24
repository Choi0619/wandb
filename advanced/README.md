# LLM Instruction-Tuning 과제 설명

이번 과제에서는 **LLM (Large Language Model)**을 사용자 지정 텍스트 데이터로 **Instruction-Tuning**을 진행하고, 학습 과정에서 발생하는 `train/loss`와 `eval/loss` 값을 **WandB**를 활용하여 기록 및 시각화하였습니다.

## Instruction-Tuning 과정

### 1. **데이터 준비**
- **Corpus 데이터**: `corpus.json` 파일을 사용하였으며, 해당 데이터는 사용자(`user`)와 치료사(`therapist`) 간의 상담 대화를 포함하고 있습니다. 각 대화 쌍은 `"role": "user"`와 `"role": "therapist"`로 구분되며, 100개의 대화 쌍이 존재합니다.
- **데이터 분할**: 데이터는 **80%** 학습용, **20%** 검증용으로 분할하여 **train-test split**을 진행하였습니다.

### 2. **모델 및 토크나이저 준비**
- **모델**: `facebook/opt-350m` 모델을 사용하였습니다. 이 모델은 자연어 생성을 위한 Causal Language Model입니다.
- **토크나이저**: 같은 `facebook/opt-350m` 모델의 토크나이저를 사용하여 데이터를 처리하였습니다.

### 3. **Fine-tuning 및 학습 과정**
- **모델 학습**: Hugging Face의 `SFTTrainer`를 사용하여 모델을 학습시켰습니다. `train/loss`와 `eval/loss`는 각각의 스텝마다 기록되며, 각 에폭(epoch)마다 WandB로 시각화하여 추적할 수 있도록 설정하였습니다.
- **학습 중 로깅**: 매 스텝(step)마다 WandB로 손실(loss) 값을 기록하여 실시간으로 학습 상태를 확인할 수 있었습니다.

### 4. **결과**
- **Train Loss**: 학습 중 `train/loss`는 점진적으로 감소하는 양상을 보였습니다. 아래 첨부된 그래프에서 학습 손실이 꾸준히 줄어드는 모습을 확인할 수 있습니다.

[![image](https://github.com/user-attachments/assets/6440020a-0e60-4ffd-94dc-072998383aad)](https://wandb.ai/wrtyu0603-illinois-institute-of-technology/therapist-chatbot/reports/train-loss-24-10-24-17-40-42---Vmlldzo5ODc1MDA4)

- **Eval Loss**: 검증 데이터에 대한 `eval/loss`는 에폭이 진행됨에 따라 감소하다가, 마지막 에폭에서 약간 증가하는 경향을 보였습니다. 이는 **overfitting**의 경우로 보여집니다.

![image](https://github.com/user-attachments/assets/7b93f417-0691-45ae-bb54-4a61cbbb159b)

## 사용된 도구 및 환경
- **모델**: Facebook OPT-350M (`AutoModelForCausalLM`)
- **프레임워크**: Hugging Face Transformers, Datasets, WandB
- **데이터**: 사용자-치료사 대화 데이터 (`corpus.json`)
- **학습 환경**: GPU 기반 학습, **10 Epochs**
- **배치 크기**: 8

## 결론
이번 실습에서는 자신만의 .json에 저장된 **텍스트 데이터**를 사용하여 **LLM Instruction-Tuning**을 수행하였습니다. 학습 과정에서 `train/loss`와 `eval/loss`를 기록하고, 이를 WandB를 통해 시각화하여 학습 성능을 모니터링할 수 있었습니다. 모델의 학습 성능은 전체적으로 개선되었으며, Eval 일부 에폭에서 **overfitting**의 경향을 확인할 수 있었습니다.

## WandB 로그

- **Train Loss** WandB 로그: [Train Loss 링크](https://wandb.ai/wrtyu0603-illinois-institute-of-technology/therapist-chatbot/reports/train-loss-24-10-24-17-40-42---Vmlldzo5ODc1MDA4)
- **Eval Loss** WandB 로그: [Eval Loss 링크](https://wandb.ai/wrtyu0603-illinois-institute-of-technology/therapist-chatbot/reports/eval-loss-24-10-24-17-40-50---Vmlldzo5ODc1MDA5)
