# GPT Fine-tuning 과제 설명

이번 과제에서는 **GPT 모델**을 Fine-tuning 하면서 `train/loss`와 `eval/loss`에 대한 validation 데이터를 측정하는 과정을 수행했습니다. 또한, 학습 진행 중 **WandB**를 활용하여 각 에폭(epoch)마다 발생하는 손실(loss) 값들을 추적하고, 그래프 형태로 시각화했습니다.

## Fine-tuning 과정

- **Validation Data 준비**: 
  사용자와 치료사 간의 대화 데이터를 JSON 형식으로 로드한 후, 이를 학습용 데이터셋과 검증용 데이터셋으로 나누었습니다. 학습 데이터는 전체 데이터의 80%, 검증 데이터는 20%로 설정했습니다.

- **Train 및 Evaluation 과정**:
  모델 학습 중 각 에폭마다 `train/loss`와 `eval/loss` 값을 WandB에 기록하여 그래프로 확인했습니다. 학습 손실 값과 평가 손실 값이 감소하는 것을 통해 모델이 점차 성능을 개선하고 있다는 것을 알 수 있습니다.

## 결과
- **Train Loss**: 학습 진행 중 `train/loss` 값은 점진적으로 감소하는 양상을 보였습니다. 아래 첨부된 그래프에서 학습 손실이 꾸준히 줄어드는 모습을 확인할 수 있습니다.
  
![train loss](https://github.com/user-attachments/assets/9d2274b9-5100-4bb2-a8a6-40b4b67cdbe1)

- **Eval Loss**: 검증 데이터에 대한 `eval/loss` 값 역시 에폭이 진행됨에 따라 점차 감소하였으며, 이는 모델이 과적합되지 않고 일반화된 성능을 보이고 있음을 나타냅니다.
  
![eval loss](https://github.com/user-attachments/assets/818996e8-9e8d-4350-84bb-ed1b27d66258)

## 사용한 도구 및 환경
- **모델**: Facebook OPT-350M (AutoModelForCausalLM)
- **데이터**: JSON 파일로 제공된 사용자-치료사 대화 데이터
- **프레임워크**: Hugging Face Transformers, Datasets, WandB
- **학습 환경**: GPU 기반 학습, 10 Epochs
- **배치 크기**: 8
- **에폭 수**: 10으로 설정

## 결론

이번 실습에서는 GPT 모델의 Fine-tuning 과정에서 검증 데이터셋을 활용하여 학습 중 `train/loss`와 `eval/loss`를 추적하고, 이를 시각적으로 WandB에서 확인할 수 있도록 했습니다. Fine-tuning 과정이 성공적으로 완료되었으며, 모델이 점차 성능을 개선해가는 모습을 확인할 수 있었습니다.

## WandB 로그

- **Train Loss** WandB 로그: [Train Loss 링크](https://api.wandb.ai/links/wrtyu0603-illinois-institute-of-technology/xkig1j60)
- **Eval Loss** WandB 로그: [Eval Loss 링크](https://wandb.ai/wrtyu0603-illinois-institute-of-technology/gyuhwan/reports/eval-loss-24-10-23-21-43-00---Vmlldzo5ODU4Njgw)
