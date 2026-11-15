#!/usr/bin/env python3
"""AEVDES EEG Signal Quality Control (QC) Script.

초기 코호트 EEG CSV/TXT 파일의 무결성 및 신호 유효성 점검:
  - 빈 파일 및 로드 실패 여부
  - 수치형 채널 존재 여부 및 샘플 길이(min_samples) 확인
  - 결측치(NaN) 비율 및 전극 접촉 불량으로 인한 Flat 채널(표준편차 < 1e-6) 탐지
  (단순히 행이 존재한다고 합격 처리하지 않고, 실제 신호 변동성을 기준으로 유효성 판정)
"""

import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


def load_tabular_file(file_path: Path) -> Optional[pd.DataFrame]:
    """구분자(쉼표, 탭, 세미콜론)를 순회하며 EEG CSV/TXT 파일 로드."""
    for sep in (",", "\t", ";"):
        try:
            df = pd.read_csv(file_path, sep=sep)
            if df.shape[1] > 1:
                return df
        except Exception:
            continue
    return None


def analyze_eeg_file(
    file_path: Path,
    min_samples: int = 100,
    max_missing_ratio: float = 0.3,
) -> Dict[str, object]:
    result: Dict[str, object] = {
        "file": file_path.name,
        "path": str(file_path),
        "status": "INVALID",
        "rows": 0,
        "numeric_channels": 0,
        "missing_ratio": np.nan,
        "mean_std": np.nan,
        "flat_channels": 0,
        "reason": "",
    }

    df = load_tabular_file(file_path)
    if df is None:
        result["reason"] = "LOAD_FAILED"
        return result

    result["rows"] = int(len(df))
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.empty:
        result["reason"] = "NO_NUMERIC_CHANNEL"
        return result

    n_channels = int(numeric_df.shape[1])
    result["numeric_channels"] = n_channels

    missing_ratio = float(numeric_df.isna().mean().mean())
    result["missing_ratio"] = round(missing_ratio, 4)

    channel_std = numeric_df.std(skipna=True)
    result["mean_std"] = round(float(channel_std.mean()), 6)

    flat_channels = int((channel_std < 1e-6).sum())
    result["flat_channels"] = flat_channels

    reasons = []
    if len(df) < min_samples:
        reasons.append("TOO_SHORT")
    if missing_ratio > max_missing_ratio:
        reasons.append("HIGH_MISSING")
    if n_channels == 0:
        reasons.append("NO_CHANNEL")
    if flat_channels == n_channels:
        reasons.append("ALL_FLAT")

    if reasons:
        result["reason"] = "|".join(reasons)
    else:
        result["status"] = "VALID"
        result["reason"] = "OK"

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="AEVDES EEG Data Quality Check")
    parser.add_argument("--input", required=True, help="EEG CSV/TXT 데이터 폴더 경로")
    parser.add_argument("--output", default="results/eeg_qc", help="QC 결과 저장 폴더")
    parser.add_argument("--min-samples", type=int, default=100, help="최소 유효 샘플(행) 수")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = []
    for ext in ("*.csv", "*.txt", "*.tsv"):
        files.extend(sorted(input_dir.rglob(ext)))

    print(f"EEG files found: {len(files)}")
    results = []

    for idx, file_path in enumerate(files, start=1):
        res = analyze_eeg_file(file_path, min_samples=args.min_samples)
        results.append(res)
        if idx % 20 == 0 or idx == len(files):
            print(f"[{idx}/{len(files)}] Processed {file_path.name} -> {res['status']}")

    result_df = pd.DataFrame(results)
    detail_path = output_dir / "eeg_file_qc.csv"
    result_df.to_csv(detail_path, index=False, encoding="utf-8-sig")

    total = len(result_df)
    valid = int((result_df["status"] == "VALID").sum()) if total > 0 else 0
    invalid = total - valid
    survival_rate = (valid / total * 100.0) if total > 0 else 0.0

    summary = pd.DataFrame([{
        "modality": "EEG",
        "total_files": total,
        "valid_files": valid,
        "invalid_files": invalid,
        "survival_rate_percent": round(survival_rate, 2),
    }])

    summary_path = output_dir / "eeg_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n========== EEG QC SUMMARY ==========")
    print(summary.to_string(index=False))
    print(f"\nSaved:\n  - {detail_path}\n  - {summary_path}")


if __name__ == "__main__":
    main()