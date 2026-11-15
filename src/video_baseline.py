#!/usr/bin/env python3
"""AEVDES Video-only Baseline (Uniform Frame Sampling + ResNet18 + OvR LR).

NOTE:
  이 스크립트는 Audio 베이스라인과 동일한 참가자 분할(5-Fold GroupKFold) 및 분류기 구조에서
  영상 모달리티 성능을 비교하기 위해 작성한 실험 파이프라인입니다.
  초기 30명 아카이브의 원본 MP4(388개)는 용량 및 얼굴 노출 문제로 로컬에서만 관리하며,
  실제 MP4 경로가 매핑된 metadata CSV를 준비한 뒤 별도로 실행해 결과를 기록합니다.
"""

import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import json

import cv2
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torchvision import models, transforms
from tqdm import tqdm

from evaluation import (
    TARGET_LABELS,
    evaluate_multilabel,
    load_metadata_csv,
    print_results,
)

N_SPLITS = 5
DEFAULT_NUM_FRAMES = 16


def build_resnet18_extractor(device: torch.device) -> Tuple[nn.Module, transforms.Compose]:
    """ImageNet 사전학습 ResNet18의 마지막 FC 레이어를 제거해 512차원 특징 추출기로 사용."""
    weights = models.ResNet18_Weights.DEFAULT
    backbone = models.resnet18(weights=weights)
    backbone.fc = nn.Identity()
    backbone.to(device)
    backbone.eval()

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return backbone, preprocess


def sample_uniform_frames(video_path: Path, num_frames: int = DEFAULT_NUM_FRAMES) -> List[Image.Image]:
    """영상 전체 길이에서 num_frames개의 프레임을 등간격으로 샘플링."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise ValueError(f"Video contains no readable frames: {video_path}")

    indices = np.linspace(0, max(total_frames - 1, 0), num=num_frames, dtype=int)
    frames: List[Image.Image] = []

    for frame_idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, bgr_frame = cap.read()
        if not ret or bgr_frame is None:
            continue
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(rgb_frame))

    cap.release()

    # 깨진 프레임이 너무 많으면 평균 임베딩이 왜곡되므로 최소 절반 이상 읽혔는지 확인
    min_required = max(1, num_frames // 2)
    if len(frames) < min_required:
        raise RuntimeError(
            f"Decoded only {len(frames)}/{num_frames} frames from {video_path}. "
            "Check video codec or file integrity."
        )
    return frames


@torch.no_grad()
def extract_video_embedding(
    video_path: Path,
    backbone: nn.Module,
    preprocess: transforms.Compose,
    device: torch.device,
    num_frames: int = DEFAULT_NUM_FRAMES,
) -> np.ndarray:
    """샘플링된 프레임들의 ResNet18 임베딩(T, 512)을 시간축 평균 풀링하여 512차원 벡터 반환.

    TODO: 추후 얼굴 영역 크롭(face crop) 전처리나 EfficientFace 백본과 비교 실험할 때
          이 함수의 preprocess/backbone 부분만 교체하면 됨.
    """
    pil_frames = sample_uniform_frames(video_path, num_frames=num_frames)
    batch_tensor = torch.stack([preprocess(img) for img in pil_frames]).to(device)
    frame_embeddings = backbone(batch_tensor).cpu().numpy()
    return frame_embeddings.mean(axis=0)


def run_video_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict:
    cv = GroupKFold(n_splits=N_SPLITS)
    y_pred = np.zeros_like(y, dtype=int)
    fold_metrics = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y, groups), 1):
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
        y_pred[test_idx] = (probs >= 0.5).astype(int)

        fold_eval = evaluate_multilabel(y[test_idx], y_pred[test_idx])
        fold_metrics.append({
            "fold": fold_idx,
            "n_samples": int(len(test_idx)),
            "n_participants": int(len(set(groups[test_idx]))),
            "Macro-F1": fold_eval["Macro-F1"],
            "Micro-F1": fold_eval["Micro-F1"],
        })

    overall = evaluate_multilabel(y, y_pred)
    overall["fold_metrics"] = fold_metrics
    return overall


def main() -> None:
    parser = argparse.ArgumentParser(description="AEVDES Video-only Multi-label Baseline")
    parser.add_argument(
        "--metadata",
        required=True,
        help="video_path, participant_id, labels 컬럼이 포함된 CSV 경로",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=DEFAULT_NUM_FRAMES,
        help="영상당 균등 샘플링할 프레임 수 (기본값: 16)",
    )
    parser.add_argument("--out", default="results/video_metrics.json", help="결과 JSON 저장 경로")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone, preprocess = build_resnet18_extractor(device)

    df, y, groups = load_metadata_csv(Path(args.metadata))
    if "video_path" not in df.columns:
        raise ValueError("Metadata CSV must contain a 'video_path' column.")

    embeddings = []
    for video_path in tqdm(df["video_path"], desc="Extracting ResNet18 video features"):
        emb = extract_video_embedding(
            Path(video_path), backbone, preprocess, device, num_frames=args.num_frames
        )
        embeddings.append(emb)

    X = np.asarray(embeddings)
    results = run_video_cv(X, y, groups)
    print_results("Video Baseline - ResNet18 + OvR LR", results)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "modality": "Video-only",
        "backbone": "ResNet18 (ImageNet pretrained, 512D temporal mean pooling)",
        "num_frames": args.num_frames,
        "n_samples": int(len(y)),
        "n_participants": int(len(set(groups))),
        "target_labels": TARGET_LABELS,
        "cv_scheme": f"Participant-wise {N_SPLITS}-Fold GroupKFold",
        "results": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nSaved video evaluation metrics to {out_path}")


if __name__ == "__main__":
    main()