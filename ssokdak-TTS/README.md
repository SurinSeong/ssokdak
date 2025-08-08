# 🎤 Korean TTS Fine-tuning & IoT Deployment (XTTS-v2)

이 저장소는 **Coqui TTS의 XTTS-v2**를 기반으로  
한국어 발음과 억양을 개선하기 위해 파인튜닝하고,  
경량화 후 IoT 기기에 배포한 과정을 기록합니다.

---

## 📌 프로젝트 개요
- **목표**: 범용 TTS 모델을 한국어 친화적으로 개선
- **모델**: Coqui TTS - XTTS-v2
- **데이터셋**: Kaggle 한국어 단일 화자 + 개인 음성(유튜브 동화책 읽기)
- **결과**:
  - 한국어 발음·억양 향상
  - Reference voice 기반 화자 스타일 모사 가능
  - 모델 크기 **5.8GB → 1.8GB**로 축소 (GPT2 + HiFi-GAN 양자화)

---

## 📊 TTS Fine-tuning Workflow

![TTS Fine-tuning Flow](./tts_finetune_flow.png)

---

## 🛠️ 환경 설정

```bash
# 1. Conda 가상환경 생성 (Python 3.10 권장)
conda create -n xtts python=3.10
conda activate xtts

# 2. PyTorch + CUDA 설치 (CUDA 버전 확인 후 맞춰 설치)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu{알맞은 버전}

# 3. Coqui TTS 설치
pip install coqui-tts

# 4. FFmpeg 설치 (오디오 전처리용)
sudo apt install ffmpeg
```

## 📂 데이터셋 구성

```
dataset/
├── wavs/
│   ├── 0001.wav
│   ├── 0002.wav
│   └── ...
└── metadata.csv   # 형식: wav_path|transcript|speaker_name

```

- 전처리
    - 무음 제거(0.5초 이상), 노이즈 최소화
    - 22.05kHz / mono 변환
    - 발화 길이 12~15초 이하로 분할

- 출처
    - [Kaggle - Korean Single Speaker Speech Dataset](https://www.kaggle.com/datasets/bryanpark/korean-single-speaker-speech-dataset)
    - 유튜브 개인 음성(저작권 허락 가정)

## 🚀 학습 방법

### 주요 하이퍼파라미터

| 파라미터           | 값                 |
| -------------- | ----------------- |
| sample\_rate   | 22050             |
| batch\_size    | 8\~16             |
| learning\_rate | 1e-4 → 1e-5 decay |
| warmup\_steps  | 2000              |
| max\_steps     | 50k\~200k         |
| text\_cleaners | korean\_cleaners  |

## 🧪 추론

- Reference Voice: 10~30초 클립 권장 (톤·발음 일관성 ↑)
- Tip: 긴 문장은 문장 단위로 나누어 합성 후 연결

## 📦 경량화

- 결과 용량
    - Before: 5.8 GB
    - After: 1.8 GB

- 효과
    - IoT 보드(ARM, 저전력 CPU)에서도 실시간 합성 가능
    - 로딩 속도 및 메모리 사용량 감소

## 📊 품질 평가
- CER/WER: ASR 역변환으로 발음 오류율 측정
- MOS(Mean Opinion Score): 5점 척도 청취 평가
- Speaker Similarity: 임베딩 cosine similarity