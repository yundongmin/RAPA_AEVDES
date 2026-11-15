# 한국어 공감 표현 멀티모달 데이터셋 검증 및 베이스라인 파이프라인 (AEVDES)

아주대학교 공감 표현 멀티모달 데이터셋(**AEVDES**, *Ajou Empathy-Visual Database of Expression Speech*) 초기 코호트(30명)를 대상으로 데이터 정합성 검수(QC) 및 다중 라벨 공감 전략 분류를 수행한 연구 프로젝트입니다.

한국전파진흥협회(RAPA) 및 대학ICT연구센터(ITRC) 연계 과제로 수행했으며, 원시 멀티모달 아카이브(Audio · Video · EEG · BPM/GSR)의 신호 품질 검증부터 참가자 독립 교차검증(Participant-wise Cross Validation) 베이스라인 구축까지의 전체 파이프라인을 포함합니다.

---

## 1. Project Overview

자유 발화 기반의 심리상담·공감 인터뷰 데이터셋에서는 **복수 라벨 발화의 파일 중복 저장**, **녹음 결측(무음)**, **생체신호 센서 접촉 불량**, 그리고 **발화 길이(Duration)와 다중 라벨 수 간의 교란 효과(Confounding Effect)**가 모델 평가를 크게 왜곡할 수 있습니다.

본 프로젝트에서는 무분별한 딥러닝 모델 적용에 앞서 원시 멀티모달 데이터의 무결성을 정량적으로 검증하고, 소규모 표본 환경에서 과적합을 통제한 모달리티별 독립 베이스라인과 교란 변수 대조군(Ablation)을 설계해 데이터의 실질적인 변별력을 규명했습니다.

### Dataset & Modality Summary

| 항목 | 내용 |
| :--- | :--- |
| **Dataset** | AEVDES (초기 코호트 30명 보관본) |
| **Raw Audio / Video** | 388개 세그먼트 (피드백 구간 자유 발화) |
| **Deduplicated Utterances** | 333개 고유 발화 (SHA-256 바이트 해시 기준 중복 통합) |
| **QC-Passed Valid Audio** | **323개 유효 발화** (완전 무음 10개 제외) |
| **Bio-signal Survival (QC)** | EEG **99 / 111건** (89.19%) · BPM/GSR **58 / 72건** (80.56%) |
| **Target Labels (Multi-label)** | `03` 도와주기 (Helping), `04` 지지하기 (Supporting), `05` 격려하기 (Encouraging) |
| **Validation Scheme** | Participant-wise 5-Fold `GroupKFold` (참가자 독립 분할) |
| **Tech Stack** | Python, FFmpeg, Librosa, Scikit-learn, PyTorch, Torchvision, OpenCV |

> 전체 5개 공감 유형 중 표본 수가 극소수인 `01 Neutral`(18건)과 `02 Clarifying`(7건) 단독 발화는 데이터에서 임의 삭제하지 않고 `[0, 0, 0]` 음성 사례(Negative Sample)로 유지하여 실제 데이터 분포를 보존했습니다.

---

## 2. Validation & Modeling Pipeline

### Step 01. SHA-256 Deduplication & Multi-label Restoration
- 단일 라벨 코드별로 중복 저장된 388개 원시 MP3 파일을 `SHA-256` 바이트 해시 기준으로 대조
- 동일 발화임이 확인된 중복 그룹(50그룹)을 **333개 고유 발화**로 통합하고 다중 라벨(Multi-hot) 타깃 벡터로 복원
- 참가자 간 교차 중복(Cross-participant duplicate) 무결성 검증 자동화

### Step 02. Multi-modal Signal Quality Control (QC)
- **Audio QC:** 디코딩 파형 기준 완전 무음(`RMS < -80 dBFS` 또는 `Peak < -60 dBFS`) 녹음 결측 10건을 스크리닝하여 **323개 유효 발화** 확정
- **EEG QC (`src/eeg_quality_check.py`):** 수치형 채널 유무, 최소 샘플 길이, 결측률(`>30%`), 전극 접촉 불량에 따른 Flat 채널(`std < 1e-6`) 정량 판별
- **BPM/GSR QC (`src/bpm_quality_check.py`):** 심박 컬럼 자동 탐색, 결측률(`>30%`), 생리적 정상 범위(`30~220 BPM`) 이탈 비율 검증

### Step 03. Modality-Specific Feature Extraction
- **Audio Baseline (`src/audio_baseline.py`):**
  - 디스크 임시 WAV 생성 없이 `FFmpeg` 메모리 파이프(`pipe:0 → pipe:1`)를 통해 16kHz Mono 파형 직접 디코딩
  - 25ms 윈도우 / 10ms 홉 기준 프레임별 17종 음향 지표(13 MFCC + RMS + ZCR + Spectral Centroid + Spectral Flatness)의 평균 및 표준편