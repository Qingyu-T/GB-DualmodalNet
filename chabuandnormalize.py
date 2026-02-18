# impute_and_scale_simple.py
# 使用示例：
#   python impute_and_scale_simple.py \
#       --input_file /mnt/data/tqy/GP_classificationYong/ABC_cleaned_filtered1104.xlsx \
#       --id_col patient_id \
#       --out_dir /mnt/data/tqy/GP_classificationYong/imputed2 \
#       --n_imputations 5 \
#       --scale zscore \
#       --exclude_cols ""   # 需要排除的非特征列（逗号分隔），没有就空着

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

def read_table(p: Path) -> pd.DataFrame:
    if p.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(p, dtype=object)
    elif p.suffix.lower() in [".csv", ".txt"]:
        return pd.read_csv(p, dtype=object)
    else:
        raise ValueError(f"Unsupported file type: {p.suffix}")

def to_float_df(df: pd.DataFrame) -> pd.DataFrame:
    # 仅将数值特征列转为 float，保留 NaN
    out = pd.DataFrame(index=df.index)
    for c in df.columns:
        out[c] = pd.to_numeric(df[c], errors="coerce")
    return out

def compute_scale_params(X: pd.DataFrame, method: str):
    """对每列忽略 NaN 计算中心与尺度。"""
    centers = {}
    scales = {}
    if method == "zscore":
        for c in X.columns:
            col = X[c]
            mu = col.mean(skipna=True)
            sd = col.std(skipna=True, ddof=0)
            if pd.isna(sd) or sd == 0:
                sd = 1.0
            centers[c] = float(mu) if not pd.isna(mu) else 0.0
            scales[c] = float(sd)
    elif method == "robust":
        for c in X.columns:
            col = X[c]
            med = col.median(skipna=True)
            q1 = col.quantile(0.25, interpolation="linear")
            q3 = col.quantile(0.75, interpolation="linear")
            iqr = (q3 - q1) if pd.notna(q3) and pd.notna(q1) else np.nan
            if pd.isna(iqr) or iqr == 0:
                iqr = 1.0
            centers[c] = float(med) if not pd.isna(med) else 0.0
            scales[c] = float(iqr)
    else:
        raise ValueError("scale method must be 'zscore' or 'robust'")
    return centers, scales

def apply_standardize(X: pd.DataFrame, centers: dict, scales: dict) -> pd.DataFrame:
    Xs = X.copy()
    for c in X.columns:
        mu = centers[c]
        sd = scales[c]
        Xs[c] = (X[c] - mu) / sd
    return Xs

def inverse_standardize(Xs: pd.DataFrame, centers: dict, scales: dict) -> pd.DataFrame:
    X = Xs.copy()
    for c in Xs.columns:
        mu = centers[c]
        sd = scales[c]
        X[c] = Xs[c] * sd + mu
    return X

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_file", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--id_col", default="patient_id")
    ap.add_argument("--exclude_cols", default="", help="逗号分隔的列名，这些列将被保留但不作为特征参与标准化/插补")
    ap.add_argument("--n_imputations", type=int, default=5)
    ap.add_argument("--scale", choices=["zscore", "robust"], default="zscore")
    ap.add_argument("--random_seed", type=int, default=42)
    args = ap.parse_args()

    in_path = Path(args.input_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_raw = read_table(in_path)
    if args.id_col not in df_raw.columns:
        raise ValueError(f"ID column '{args.id_col}' not found in input file.")

    # 保留 ID 为字符串（含前导0）
    df_raw[args.id_col] = df_raw[args.id_col].astype(str).str.strip()

    # 解析要排除但保留的列
    exclude = [c.strip() for c in args.exclude_cols.split(",") if c.strip()]
    keep_cols = [args.id_col] + exclude

    # 特征列 = 除 keep_cols 外的所有列（假定已是数值或 0/1）
    feat_cols = [c for c in df_raw.columns if c not in keep_cols]
    if len(feat_cols) == 0:
        raise ValueError("No feature columns found after excluding keep_cols.")

    # 转 float（不做任何“字符串清洗→NaN”，只做安全的 to_numeric）
    X = to_float_df(df_raw[feat_cols])

    # 标准化参数（忽略 NaN）
    centers, scales = compute_scale_params(X, args.scale)
    X_std = apply_standardize(X, centers, scales)

    # 多重插补：在标准化空间进行
    imputations_std = []
    for k in range(args.n_imputations):
        rs = args.random_seed + k
        imp = IterativeImputer(
            estimator=BayesianRidge(),
            sample_posterior=True,
            max_iter=20,
            random_state=rs,
            skip_complete=True,
            tol=1e-3
        )
        X_imp_std_np = imp.fit_transform(X_std)
        X_imp_std = pd.DataFrame(X_imp_std_np, index=X_std.index, columns=X_std.columns)
        imputations_std.append((k+1, X_imp_std))

    # 导出：标准化后的插补结果 & 反标准化回原量纲的插补结果；以及各自的长表
    long_std_rows, long_orig_rows = [], []
    meta = {
        "id_col": args.id_col,
        "exclude_cols": exclude,
        "feature_cols": feat_cols,
        "scale": args.scale,
        "n_imputations": args.n_imputations,
        "random_seed": args.random_seed,
        "centers": centers,
        "scales": scales
    }

    for k, X_imp_std in imputations_std:
        X_imp_orig = inverse_standardize(X_imp_std, centers, scales)

        out_std = pd.concat([df_raw[keep_cols], X_imp_std], axis=1)
        out_orig = pd.concat([df_raw[keep_cols], X_imp_orig], axis=1)

        p_std = out_dir / f"imputed_std_k{k}.csv"
        p_orig = out_dir / f"imputed_orig_k{k}.csv"
        out_std.to_csv(p_std, index=False)
        out_orig.to_csv(p_orig, index=False)

        tmp_std = out_std.copy()
        tmp_std.insert(1, "imputation_id", k)
        long_std_rows.append(tmp_std)

        tmp_orig = out_orig.copy()
        tmp_orig.insert(1, "imputation_id", k)
        long_orig_rows.append(tmp_orig)

    long_std = pd.concat(long_std_rows, axis=0, ignore_index=True)
    long_orig = pd.concat(long_orig_rows, axis=0, ignore_index=True)
    long_std.to_csv(out_dir / "imputed_std_long.csv", index=False)
    long_orig.to_csv(out_dir / "imputed_orig_long.csv", index=False)

    with open(out_dir / "impute_scale_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved to: {out_dir}")
    print(f"  - standardized CSVs: imputed_std_k*.csv + imputed_std_long.csv")
    print(f"  - original-scale CSVs: imputed_orig_k*.csv + imputed_orig_long.csv")
    print(f"  - meta: impute_scale_meta.json")
    # 简要汇总
    na_before = X.isna().sum().sum()
    print(f"Missing cells before impute: {int(na_before)}")
    print(f"Feature columns ({len(feat_cols)}): {feat_cols[:12]}{' ...' if len(feat_cols)>12 else ''}")

if __name__ == "__main__":
    main()
