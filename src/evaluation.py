import collections
import hashlib
import zipfile
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    hamming_loss,
    precision_recall_fscore_support,
)

# 01(Neutral), 02(Clarifying)는 극소수 클래스라 타깃에서는 제외하되 [0, 0, 0] 샘플로 유지함
TARGET_LABELS = ("03", "04", "05")
VALID_LABEL_CODES = ("01", "02", "03", "04", "05")


def evaluate_multilabel(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, object]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    results: Dict[str, object] = {
        "Macro-F1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "Micro-F1": round(float(f1_score(y_true, y_pred, average="micro", zero_division=0)), 4),
        "Exact Match": round(float(accuracy_score(y_true, y_pred)), 4),
        "Hamming Accuracy": round(float(1.0 - hamming_loss(y_true, y_pred)), 4),
    }

    pr, rc, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    per_label = {}
    for idx, code in enumerate(TARGET_LABELS):
        per_label[f"label_{code}"] = {
            "support": int(support[idx]),
            "precision": round(float(pr[idx]), 4),
            "recall": round(float(rc[idx]), 4),
            "f1": round(float(f1[idx]), 4),
            "balanced_accuracy": round(
                float(balanced_accuracy_score(y_true[:, idx], y_pred[:, idx])), 4
            ),
        }
    results["per_label"] = per_label
    return results


def print_results(title: str, results: Dict[str, object]) -> None:
    print(f"\n[{title}]")
    print(f"  Macro-F1         : {results['Macro-F1'] * 100:.2f}% ({results['Macro-F1']:.4f})")
    print(f"  Micro-F1         : {results['Micro-F1'] * 100:.2f}% ({results['Micro-F1']:.4f})")
    print(f"  Exact Match      : {results['Exact Match'] * 100:.2f}% ({results['Exact Match']:.4f})")
    print(f"  Hamming Accuracy : {results['Hamming Accuracy'] * 100:.2f}% ({results['Hamming Accuracy']:.4f})")


def encode_target_labels(label_sets: List[Set[str]]) -> np.ndarray:
    """라벨 코드 집합을 ('03', '04', '05') 기준 multi-hot 행렬로 변환."""
    return np.array(
        [[int(code in labels) for code in TARGET_LABELS] for labels in label_sets],
        dtype=int,
    )


def parse_label_string(raw_val: object, row_idx: int = -1) -> Set[str]:
    """'03|04' 형태의 문자열 파싱. 빈 값이나 이상한 코드가 들어오면 예외 발생."""
    if pd.isna(raw_val) or not str(raw_val).strip():
        raise ValueError(f"Empty label field at row {row_idx}")

    tokens = {t.strip().zfill(2) for t in str(raw_val).split("|") if t.strip()}
    if not tokens:
        raise ValueError(f"No valid label tokens parsed at row {row_idx}: {raw_val!r}")

    invalid = tokens - set(VALID_LABEL_CODES)
    if invalid:
        raise ValueError(f"Unknown label code(s) {invalid} at row {row_idx}. Allowed: {VALID_LABEL_CODES}")
    return tokens


def load_metadata_csv(csv_path: Path) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path, dtype=str)
    required_cols = {"participant_id", "labels"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")

    label_sets = [parse_label_string(val, idx) for idx, val in enumerate(df["labels"])]
    y = encode_target_labels(label_sets)
    groups = df["participant_id"].astype(str).str.strip().to_numpy()
    return df, y, groups


def load_deduplicated_audio_zip(zip_obj: zipfile.ZipFile) -> List[Dict]:
    """ZIP 내부의 MP3를 읽고 SHA-256 해시 기준으로 동일 음성을 하나로 묶음.

    초기 보관본에서 복수 라벨 발화가 라벨코드만 다른 별도 MP3로 중복 저장되어 있어
    해시가 같은 파일들의 2번째 토큰(라벨)을 하나의 set으로 병합한다.
    """
    records = {}
    participants_per_hash = collections.defaultdict(set)

    for info in zip_obj.infolist():
        if info.is_dir() or not info.filename.lower().endswith(".mp3"):
            continue
        parts = Path(info.filename).parts
        if len(parts) < 2 or not parts[0].startswith("Participant_"):
            continue

        participant = parts[0]
        tokens = Path(info.filename).stem.split("-")
        if len(tokens) != 7 or tokens[0] != "02" or tokens[1] not in VALID_LABEL_CODES:
            raise ValueError(f"Unexpected filename format: {Path(info.filename).name}")

        raw_bytes = zip_obj.read(info)
        sha256_hash = hashlib.sha256(raw_bytes).hexdigest()
        participants_per_hash[sha256_hash].add(participant)

        key = (participant, sha256_hash)
        if key not in records:
            records[key] = {
                "participant": participant,
                "sha256": sha256_hash,
                "member": info.filename,
                "labels": set(),
            }
        records[key]["labels"].add(tokens[1])

    # 서로 다른 참가자 폴더에 완전히 동일한 음성 파일이 들어있는지 체크
    cross_dups = {k: list(v) for k, v in participants_per_hash.items() if len(v) > 1}
    if cross_dups:
        raise ValueError(
            f"Identical audio hash found across multiple participants ({len(cross_dups)} groups)."
        )

    return list(records.values())