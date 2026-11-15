# AEVDES Multimodal Empathy Classification Baselines

아주대학교 공감 표현 멀티모달 데이터셋(**AEVDES**) 초기 코호트(30명)를 대상으로 다중 라벨 공감 전략 분류를 검증하기 위해 작성한 **Audio / Video 베이스라인 코드**입니다.

---

## 1. 파이프라인 구성

### Audio Baseline (`src/audio_baseline.py`)
- **입력:** 16kHz Mono 변환 음성 (FFmpeg 파이프 디코딩)
- **특징 추출 (34D):** 25ms 윈도우 / 10ms 홉 기준 13 MFCC + RMS + ZCR + Spectral Centroid + Spectral Flatness의 프레임 평균 및 표준편차 (17 × 2 = 34차원)
- **분류기:** `StandardScaler` → `OneVsRestClassifier(LogisticRegression(class_weight="balanced"))`
- **검증:** 참가자 기준 5-Fold `GroupKFold` (동일 참가자 발화의 Train/Test 분리)

### Video Baseline (`src/video_baseline.py`)
- **입력:** 발화 구간 MP4 영상에서 균등 간격으로 추출한 16개 프레임 (`Uniform Frame Sampling`)
- **특징 추출 (512D):** ImageNet 사전학습 `ResNet18` 특징 추출 후 시간축 평균 풀링 (`Temporal Mean Pooling`)
- **분류기 및 검증:** Audio 베이스라인과 동일한 `StandardScaler` + `OvR Logistic Regression` 및 참가자 독립 5-Fold `GroupKFold` 적용

---

## 2. 초기 아카이브 검증 결과 (Initial 30-Participant Archive)

- **데이터 정제(QC):** 전체 MP3 388개 → SHA-256 해시 중복 통합 후 **333개 고유 발화** → 녹음 결측(완전 무음) 10개 제외 후 **323개 유효 발화**
- **타깃 라벨:** Feedback 단계 공감 전략 3종 (`03` 도와주기, `04` 지지하기, `05` 격려하기)
  - 표본 수가 적은 `01`(18건), `02`(7건) 단독 발화는 데이터에서 임의 삭제하지 않고 `[0, 0, 0]` 음성 사례(Negative sample)로 유지해 평가했습니다.

| 구분 | 유효 표본 | Macro-F1 | Micro-F1 | Exact Match | Hamming Acc. | 비고 |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Audio: Dummy (Prevalence)** | 323 | 23.90% | - | - | - | 학습 폴드 빈도 기준선 |
| **Audio: Duration-only (LR)** | 323 | 44.70% | - | - | - | $\log(1+\text{초})$ 단일 변수 대조군 |
| **Audio: 34D MFCC + OvR LR** | 323 | **42.98%** | **46.15%** | **22.60%** | **55.93%** | 실측 완료 |
| **Video: ResNet18 + OvR LR** | - | - | - | - | - | 재현용 실험 파이프라인 |

> **참고:** 현재 결과 수치는 초기 보관본에서 검증이 완료된 Audio 베이스라인 기준입니다. `video_baseline.py`는 동일한 참가자 분할 조건에서 시각 특징을 비교하기 위해 구성한 재현용 코드로, 로컬 원본 MP4 아카이브에서 별도로 실행해 결과를 저장하는 구조입니다.

---

## 3. 설치 및 실행 환경

### 시스템 의존성 (필수)
`audio_baseline.py`는 디스크에 임시 WAV를 생성하지 않고 `subprocess`로 `ffmpeg`를 직접 호출해 메모리에서 디코딩합니다. 실행 전 시스템 `PATH`에 `ffmpeg`가 설치되어 있어야 합니다.

```bash
# Ubuntu / Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows (winget)
winget install Gyan.FFmpeg

### Bio-signal Quality Control (`src/eeg_quality_check.py`, `src/bpm_quality_check.py`)
- **목적:** 생체신호(EEG, BPM/GSR)는 파일이 존재하더라도 전극 접촉 불량(Flat signal), 센서 포화, 참가자별 수집 범위 불일치가 존재하므로 무분별한 멀티모달 융합(Fusion)에 앞서 **신호 유효성 정량 QC**를 수행합니다.
- **검증 항목:**
  - **EEG (99 / 111 유효):** 수치형 채널 유무, 최소 샘플 수, 결측률(>30%), Flat 채널(`std < 1e-6`) 판별
  - **BPM/GSR (58 / 72 유효):** 심박 컬럼 자동 탐색, 결측률(>30%), 생리적 정상 범위(30~220 BPM) 이탈 비율 판별

```bash
# 4) EEG 및 BPM 생체신호 데이터 품질(QC) 점검
python src/eeg_quality_check.py --input /path/to/EEG_Folder --output results/eeg_qc
python src/bpm_quality_check.py --input /path/to/BPM_Folder --output results/bpm_qc
```