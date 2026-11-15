#!/usr/bin/env python3
"""AEVDES Audio-only Baseline (34D acoustic features + OvR Logistic Regression).

초기 30명 보관본 검증 기준:
  - 원본 388개 MP3 -> SHA-256 중복 통합 333개 -> 완전 무음 10개 제외 323개
  - 34차원 음향 특징: 13 MFCC + RMS + ZCR + Centroid + Flatness (각 평균/표준편차)
  - 비교 모델: Dummy, Duration-only(log1p 초), MFCC_34D_LR
"""

import os
import sys
from pathlib import Path

for _env in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_env] = "1"

# 프로젝트 루트에서 `python src/audio_baseline.py`로 실행해도 같은 폴더 모듈을 찾도록 보정
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import json
import subprocess
import zipfile

import librosa
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.fft import dct, rfft
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from evaluation import (
    TARGET_LABELS,
    encode_target_labels,
    evaluate_multilabel,
    load_deduplicated_audio_zip,
    load_metadata_csv,
    print_results,
)

SAMPLE_RATE = 16000
N_FFT = 512
FRAME_LENGTH = 400  # 25ms (16kHz)
HOP_LENGTH = 160    # 10ms (16kHz)
N_SPLITS = 5


def decode_mp3_bytes(mp3_bytes: bytes, sr: int = SAMPLE_RATE) -> np.ndarray:
    """ffmpeg 파이프로 MP3 바이트를 16kHz mono float64 파형으로 변환."""
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", "pipe:0", "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "pipe:1",
    ]
    try:
        proc = subprocess.run(
            cmd, input=mp3_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg 실행 파일을 찾을 수 없습니다. 시스템 PATH에 ffmpeg가 설치되어 있는지 확인하세요."
        ) from exc

    if proc.returncode != 0:
        err_msg = proc.stderr.decode("utf-8", errors="replace")[:200]
        raise ValueError(f"ffmpeg decode error: {err_msg}")

    wav = np.frombuffer(proc.stdout, dtype="<f4").astype("float64")
    if len(wav) < sr // 4:
        raise ValueError("Audio shorter than 0.25 seconds.")
    return np.clip(wav, -1.0, 1.0)


def is_near_silent(wav: np.ndarray) -> bool:
    """녹음 에러로 발생한 무음 파일 필터링 (RMS < -80 dBFS 또는 peak < -60 dBFS)."""
    rms = float(np.sqrt(np.mean(wav * wav)))
    peak = float(np.max(np.abs(wav)))
    return rms < 1e-4 or peak < 1e-3


def extract_34d_acoustic_features(wav: np.ndarray, mel_filter: np.ndarray) -> np.ndarray:
    """프레임 단위 음향 특징 17종의 평균/표준편차 추출 (총 34차원).

    librosa.feature.mfcc 기본 설정 대신 윈도우/홉 크기와 정규화 방식을 고정하기 위해
    mel filterbank 행렬곱 + ortho DCT-II로 직접 계산함.
    """
    if len(wav) < FRAME_LENGTH:
        wav = np.pad(wav, (0, FRAME_LENGTH - len(wav)))

    frames = sliding_window_view(wav, FRAME_LENGTH)[::HOP_LENGTH]
    frames = frames * np.hanning(FRAME_LENGTH)[None, :]

    spec = np.abs(rfft(frames, n=N_FFT, axis=1))
    power = spec * spec
    mel_energy = np.maximum(power @ mel_filter.T, 1e-10)
    mfcc_13 = dct(np.log(mel_energy), type=2, norm="ortho", axis=1)[:, :13]

    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    zcr = np.mean((frames[:, 1:] * frames[:, :-1]) < 0, axis=1)

    freqs = np.fft.rfftfreq(N_FFT, 1.0 / SAMPLE_RATE)
    centroid = (spec @ freqs) / np.maximum(spec.sum(axis=1), 1e-8)
    flatness = np.exp(np.mean(np.log(np.maximum(spec, 1e-10)), axis=1)) / np.maximum(
        spec.mean(axis=1), 1e-10
    )

    frame_features = np.column_stack((mfcc_13, rms, zcr, centroid / 8000.0, flatness))
    feature_vector = np.r_[frame_features.mean(axis=0), frame_features.std(axis=0)]

    if not np.isfinite(feature_vector).all():
        raise ValueError("Non-finite values in extracted acoustic features.")
    return feature_vector


def run_participant_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray, mode: str = "linear") -> dict:
    cv = GroupKFold(n_splits=N_SPLITS)
    y_pred = np.zeros_like(y, dtype=int)
    y_prob = np.zeros_like(y, dtype=float)
    fold_ids = np.zeros(len(y), dtype=int)
    fold_metrics = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y, groups), 1):
        fold_ids[test_idx] = fold_idx
        if mode == "dummy":
            prevalence = y[train_idx].mean(axis=0)
            y_prob[test_idx] = prevalence[None, :]
            y_pred[test_idx] = (prevalence >= 0.5).astype(int)[None, :]
        else:
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                (
                    "clf",
                    OneVsRestClassifier(
                        LogisticRegression(
                            C=1.0, max_iter=2000, class_weight="balanced", random_state=42
                        )
                    ),
                ),
            ])
            pipeline.fit(X[train_idx], y[train_idx])
            probs = pipeline.predict_proba(X[test_idx])
            y_prob[test_idx] = probs
            y_pred[test_idx] = (probs >= 0.5).astype(int)

        fold_eval = evaluate_multilabel(y[test_idx], y_pred[test_idx])
        fold_metrics.append({
            "fold": fold_idx,
            "n_samples": int(len(test_idx)),
            "n_participants": int(len(set(groups[test_idx]))),
            "Macro-F1": fold_eval["Macro-F1"],
            "Micro-F1": fold_eval["Micro-F1"],
            "Exact Match": fold_eval["Exact Match"],
            "Hamming Accuracy": fold_eval["Hamming Accuracy"],
        })

    overall = evaluate_multilabel(y, y_pred)
    overall["fold_metrics"] = fold_metrics
    return {
        "summary": overall,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "fold_ids": fold_ids,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="AEVDES Audio-only Multi-label Baseline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--zip", help="로컬 Audio_Participant_01-30.zip 경로")
    group.add_argument("--metadata", help="audio_path, participant_id, labels가 포함된 CSV 경로")
    parser.add_argument(
        "--include-silent",
        action="store_true",
        help="무음 10개 파일을 제외하지 않고 전체 333개로 평가할 때 사용",
    )
    parser.add_argument("--out", default="results/audio_metrics.json", help="결과 JSON 저장 경로")
    parser.add_argument("--save-oof", default=None, help="샘플별 OOF 예측 결과 CSV 저장 경로 (선택)")
    args = parser.parse_args()

    mel_filter = librosa.filters.mel(
        sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=40, fmin=50, fmax=7600
    ).astype("float64")
    features, durations, valid_flags = [], [], []

    if args.zip:
        with zipfile.ZipFile(args.zip) as zf:
            records = load_deduplicated_audio_zip(zf)
            for rec in records:
                wav = decode_mp3_bytes(zf.read(rec["member"]))
                features.append(extract_34d_acoustic_features(wav, mel_filter))
                durations.append(len(wav) / SAMPLE_RATE)
                valid_flags.append(not is_near_silent(wav))
        y = encode_target_labels([r["labels"] for r in records])
        groups = np.array([r["participant"] for r in records])
    else:
        df, y, groups = load_metadata_csv(Path(args.metadata))
        for audio_path in df["audio_path"]:
            raw_bytes = Path(audio_path).read_bytes()
            wav = decode_mp3_bytes(raw_bytes)
            features.append(extract_34d_acoustic_features(wav, mel_filter))
            durations.append(len(wav) / SAMPLE_RATE)
            valid_flags.append(not is_near_silent(wav))

    X = np.asarray(features)
    durations = np.asarray(durations)
    valid_mask = np.asarray(valid_flags, dtype=bool)

    if not args.include_silent:
        X = X[valid_mask]
        durations = durations[valid_mask]
        y = y[valid_mask]
        groups = groups[valid_mask]

    print(
        f"Evaluated Samples: {len(y)} | Participants: {len(set(groups))} | Feature Dim: {X.shape[1]}"
    )

    runs = {
        "Dummy": run_participant_cv(X, y, groups, mode="dummy"),
        "Duration_only": run_participant_cv(np.log1p(durations)[:, None], y, groups, mode="linear"),
        "MFCC_34D_LR": run_participant_cv(X, y, groups, mode="linear"),
    }

    for model_name, run_data in runs.items():
        print_results(f"Audio Baseline - {model_name}", run_data["summary"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "modality": "Audio-only",
        "n_samples": int(len(y)),
        "n_participants": int(len(set(groups))),
        "target_labels": TARGET_LABELS,
        "feature_dim": int(X.shape[1]),
        "cv_scheme": f"Participant-wise {N_SPLITS}-Fold GroupKFold",
        "results": {k: v["summary"] for k, v in runs.items()},
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nSaved audio evaluation metrics to {out_path}")

    if args.save_oof:
        oof_path = Path(args.save_oof)
        oof_path.parent.mkdir(parents=True, exist_ok=True)
        oof_df = pd.DataFrame({
            "participant": groups,
            "duration_sec": np.round(durations, 3),
            "fold": runs["MFCC_34D_LR"]["fold_ids"],
        })
        for idx, code in enumerate(TARGET_LABELS):
            oof_df[f"y_{code}"] = y[:, idx]
            oof_df[f"pred_{code}"] = runs["MFCC_34D_LR"]["y_pred"][:, idx]
            oof_df[f"prob_{code}"] = np.round(runs["MFCC_34D_LR"]["y_prob"][:, idx], 4)
        oof_df.to_csv(oof_path, index=False, encoding="utf-8-sig")
        print(f"Saved OOF predictions to {oof_path}")


if __name__ == "__main__":
    main()