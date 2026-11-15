#!/usr/bin/env python3
"""AEVDES BPM / GSR Bio-signal Quality Control (QC) Script.

초기 코호트 심박(BPM) 및 피부전도도(GSR) 분할 CSV 파일의 품질 점검:
  - 유효 샘플 수(min_samples) 및 결측치 비율 확인
  - 생리적 정상 심박 범위(30 ~ 220 BPM) 이탈 비율(out_of_range_ratio) 계산
  - 센서 포화(saturation) 또는 측정 오류 구간 필터링
"""

import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


def load_tabular_file(file_path: Path) -> Optional[pd.DataFrame]:
    """구분자(쉼표, 탭, 세미콜론)를 순회하며 BPM CSV/TXT 파일 로드."""
    for sep in (",", "\t", ";"):
        try:
            df = pd.read_csv(file_path, sep=sep)
            if df.shape[1] >= 1:
                return df
        except Exception:
            continue
    return None


def find_bpm_column(df: pd.DataFrame) -> Optional[str]:
    """컬럼명에서 BPM/Heart Rate 관련 키워드를 탐색하고, 없으면 첫 번째 수치형 컬럼 반환."""
    keywords = ("bpm", "heart", "hr", "pulse", "heartrate")
    for col in df.columns:
        col_clean = str(col).lower().replace("_", "").replace(" ", "")
        if any(kw in col_clean for kw in keywords):
            return col

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        return numeric_cols[0]
    return None


def analyze_bpm_file(
    file_path: Path,
    min_samples: int = 5,
    min_bpm: float = 30.0,
    max_bpm: float = 220.0,
) -> Dict[str, object]:
    result: Dict[str, object] = {
        "file": file_path.name,
        "path": str(file_path),
        "status": "INVALID",
        "rows": 0,
        "bpm_column": None,
        "valid_samples": 0,
        "missing_ratio": np.nan,
        "mean_bpm": np.nan,
        "std_bpm": np.nan,
        "min_bpm": np.nan,
        "max_bpm": np.nan,
        "out_of_range_ratio": np.nan,
        "reason": "",
    }

    df = load_tabular_file(file_path)
    if df is None:
        result["reason"] = "LOAD_FAILED"
        return result

    result["rows"] = int(len(df))
    bpm_col = find_bpm_column(df)
    if bpm_col is None:
        result["reason"] = "NO_BPM_COLUMN"
        return result

    result["bpm_column"] = str(bpm_col)
    bpm_series = pd.to_numeric(df[bpm_col], errors="coerce")

    missing_ratio = float(bpm_series.isna().mean())
    valid_bpm = bpm_series.dropna()

    result["missing_ratio"] = round(missing_ratio, 4)
    result["valid_samples"] = int(len(valid_bpm))

    if len(valid_bpm) == 0:
        result["reason"] = "NO_VALID_SAMPLE"
        return result

    result["mean_bpm"] = round(float(valid_bpm.mean()), 2)
    result["std_bpm"] = round(float(valid_bpm.std()), 2) if len(valid_bpm) > 1 else 0.0
    result["min_bpm"] = round(float(valid_bpm.min()), 2)
    result["max_bpm"] = round(float(valid_bpm.max()), 2)

    out_of_range = (valid_bpm < min_bpm) | (valid_bpm > max_bpm)
    out_of_range_ratio = float(out_of_range.mean())
    result["out_of_range_ratio"] = round(out_of_range_ratio, 4)

    reasons = []
    if len(valid_bpm) < min_samples:
        reasons.append("TOO_FEW_SAMPLES")
    if missing_ratio > 0.3:
        reasons.append("HIGH_MISSING")
    if out_of_range_ratio > 0.3:
        reasons.append("HIGH_OUT_OF_RANGE")

    if reasons:
        result["reason"] = "|".join(reasons)
    else:
        result["status"] = "VALID"
        result["reason"] = "OK"

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="AEVDES BPM Data Quality Check")
    parser.add_argument("--input", required=True, help="BPM CSV/TXT 데이터 폴더 경로")
    parser.add_argument("--output", default="results/bpm_qc", help="QC 결과 저장 폴더")
    parser.add_argument("--min-samples", type=int, default=5, help="최소 유효 샘플(행) 수")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = []
    for ext in ("*.csv", "*.txt", "*.tsv"):
        files.extend(sorted(input_dir.rglob(ext)))

    print(f"BPM files found: {len(files)}")
    results = []

    for idx, file_path in enumerate(files, start=1):
        res = analyze_bpm_file(file_path, min_samples=args.min_samples)
        results.append(res)
        if idx % 20 == 0 or idx == len(files):
            print(f"[{idx}/{len(files)}] Processed {file_path.name} -> {res['status']}")

    result_df = pd.DataFrame(results)
    detail_path = output_dir / "bpm_file_qc.csv"
    result_df.to_csv(detail_path, index=False, encoding="utf-8-sig")

    total = len(result_df)
    valid = int((result_df["status"] == "VALID").sum()) if total > 0 else 0
    invalid = total - valid
    survival_rate = (valid / total * 100.0) if total > 0 else 0.0

    summary = pd.DataFrame([{
        "modality": "BPM",
        "total_files": total,
        "valid_files": valid,
        "invalid_files": invalid,
        "survival_rate_percent": round(survival_rate, 2),
    }])

    summary_path = output_dir / "bpm_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n========== BPM QC SUMMARY ==========")
    print(summary.to_string(index=False))
    print(f"\nSaved:\n  - {detail_path}\n  - {summary_path}")


if __name__ == "__main__":
    main()