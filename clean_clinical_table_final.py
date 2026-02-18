# -*- coding: utf-8 -*-
"""
数据清洗脚本（按用户规范）
- 以 FilePath 的文件名去扩展名主干作为 patient_id
- 特殊字符串: "——" "—" "无" "" -> NaN
- 全角不等号 ＜ ＞ / 半角 < > 处理为浮点；"<1"/"＜1" -> 0.5（可调）
- 二分类: 性别 男=1 女=0
- 数值：年龄、生化、形态学、时间（月）
- 多分类（有序映射）：息肉个数、基底情况、回声强度、息肉形状
- 多选（multi-hot）：部位、合并消化系统症状
"""
import csv
import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

# -----------------------
# 配置：列名与词表
# -----------------------

ID_COL = "FilePath"

BIN_COL = "性别"  # 男=1 女=0

NUM_COLS = [
    "年龄(y)",
    "ALT(U/L)", "AST(U/L)", "ALP(U/L)", "GGT(U/L)", "TB(umol/L)", "DB(umol/L)", "ALB(g/l)",
    "CEA(u/l)", "CA199(u/l)", "CA125(u/l)",
    "息肉大小（长径）", "息肉大小（短径）",
    "胆囊壁厚度",
    "入院时发现胆囊息肉时间（月）",
]

# 多分类（有序）映射 —— 按你的要求（注意：有意保留你给的数值编码）
MAP_COUNT_ORDERED = {  # 息肉个数（单发或个数）
    "单发": 0, "1": 0,
    # 按要求：2=2, 3=3（即保留不连续编码）
    "2": 2, "３": 3, "3": 3,
    "数个": 4, "多个": 4, ">3": 4, "＞3": 4, "gt3": 4,
}
MAP_BASE_ORDERED = {   # 基底情况
    "带蒂": 0, "窄基底": 1, "宽基底": 2
}
MAP_ECHO_ORDERED = {   # 回声强度
    "低": 0, "弱": 0,
    "中等": 1, "等回声": 1,
    "稍强": 2, "稍高": 2,
    "强": 3,
}
MAP_SHAPE_ORDERED = {  # 息肉形状
    "球状": 0, "桑葚状": 1, "乳头状": 2, "结节状": 3,
}

# 多选（multi-hot）词表
SITES_VOCAB = ["底部", "体部", "颈部", "胆囊管"]  # 部位
SYMPTOMS_VOCAB = ["腹痛", "腹胀", "腹泻", "消化不良", "胀痛"]  # 合并消化系统症状

# 视为缺失的字符串（清洗前统一成 NaN）
NA_STRINGS = {"", "无", "—", "——", None}

# "<1" 的替代值（可调）
LESS_THAN_ONE_VALUE = 0.5

# -----------------------
# 工具函数
# -----------------------

_re_num = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
_re_full_lt = re.compile(r"^\s*[＜<]\s*([0-9]+(?:\.[0-9]+)?)\s*$")
_re_full_gt = re.compile(r"^\s*[＞>]\s*([0-9]+(?:\.[0-9]+)?)\s*$")


def is_na_like(x) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and math.isnan(x):
        return True
    s = str(x).strip()
    return s in NA_STRINGS


# 追加：更全的不等号变体集合
_LT_CHARS = set("<＜﹤⟨")
_GT_CHARS = set(">＞﹥⟩")

def _normalize_spaces(s: str) -> str:
    # 把多种空白统一成半角空格，并去掉首尾
    return (
        s.replace("\u00A0", " ")  # NBSP
        .replace("\u3000", " ")  # 全角空格
        .replace("\t", " ")
        .replace("\r", " ")
        .replace("\n", " ")
        .strip()
    )

def normalize_str(s):
    """更强：空值→''；全角/特殊标点统一；空白归一化；去掉多余标点"""
    if is_na_like(s):
        return ""
    s = str(s)
    s = _normalize_spaces(s)
    # 统一中文标点与分隔
    s = (s.replace("，", ",").replace("；", ";").replace("：", ":")
           .replace("—", "-").replace("——", "-")
           .replace("（", "(").replace("）", ")")
           .replace("？", "?"))
    # 统一不等号到半角
    s = "".join("<" if ch in _LT_CHARS else (">" if ch in _GT_CHARS else ch) for ch in s)
    # 特殊缺失词
    if s in NA_STRINGS or s in {"-", "--"}:
        return ""
    return s

_re_num = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
_re_any_lt = re.compile(r"^\s*[<]\s*([0-9]+(?:\.[0-9]+)?)\s*$")
_re_any_gt = re.compile(r"^\s*[>]\s*([0-9]+(?:\.[0-9]+)?)\s*$")

def parse_numeric_cell(x):
    s = normalize_str(x)
    if s == "":
        return np.nan
    # <X
    m = _re_any_lt.match(s)
    if m:
        v = float(m.group(1))
        return LESS_THAN_ONE_VALUE if abs(v - 1.0) < 1e-9 else v
    # >X
    m = _re_any_gt.match(s)
    if m:
        return float(m.group(1))
    # 提取首个数字
    m = _re_num.search(s)
    return float(m.group(0)) if m else np.nan

def map_ordered(value, mapping: dict, *, priority=None):
    """
    更鲁棒的有序映射：
    - 先标准化
    - 支持在长句中抓取第一个命中的关键词（或用 priority 顺序）
    - 显式处理宽/窄/带蒂的优先级冲突
    """
    s = normalize_str(value)
    if s == "":
        return np.nan

    # 显式优先级（若给定），例如基底：窄 > 宽 > 带蒂
    if priority:
        for token in priority:
            if token in s:
                return mapping.get(token, np.nan)

    # 常规：精确命中
    if s in mapping:
        return mapping[s]

    # 常见变体统一
    s_norm = s.replace("３", "3").replace("＞", ">")
    if s_norm in mapping:
        return mapping[s_norm]

    # 裸数字
    if s_norm.isdigit():
        return mapping.get(s_norm, np.nan)

    # 包含匹配：命中第一个出现的关键词
    best_pos, best_key = 10**9, None
    for k in mapping.keys():
        if not k:
            continue
        pos = s.find(k)
        if pos != -1 and pos < best_pos:
            best_pos, best_key = pos, k
    if best_key is not None:
        return mapping[best_key]

    return np.nan


def extract_patient_id(path_like):
    s = normalize_str(path_like)
    if s == "":
        return ""
    try:
        stem = Path(s).stem
        # 仅数字的场景左侧补零到 4 位；若已有 4+ 位会保持原样
        return stem.zfill(4)
    except Exception:
        return ""


def map_gender(x):
    s = normalize_str(x)
    if s in ("男", "male", "Male", "M", "m"):
        return 1
    if s in ("女", "female", "Female", "F", "f"):
        return 0
    return np.nan  # 先置 NaN，稍后用众数填充


def multi_hot(text, vocab):
    """
    将文本拆成 multi-hot：
    - 任何连接方式（空格/逗号/顿号/直接连写）都尽量用词表匹配
    - 返回一个长度=len(vocab) 的 0/1 列表
    """
    flags = [0] * len(vocab)
    s = normalize_str(text)
    if s == "" or s == "无":
        return flags
    # 做一个“包含匹配”：一旦命中词表中的词，就置 1
    for i, token in enumerate(vocab):
        if token and token in s:
            flags[i] = 1
    return flags


# -----------------------
# 主流程
# -----------------------

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        # 兜底：若不小心传入了 dict（多表），取第一个
    if isinstance(df, dict):
        first_key = next(iter(df.keys()))
        print(f"[WARN] clean_dataframe 收到 dict，使用第一个工作表: {first_key}")
        df = df[first_key]
        
    df = df.copy()

    # 1) patient_id
    df["patient_id"] = df[ID_COL].apply(extract_patient_id)

    # 2) 性别（二分类）
    df["性别_bin"] = df[BIN_COL].apply(map_gender)
    # 众数填充（若全空则保持）
    if df["性别_bin"].notna().any():
        mode_val = df["性别_bin"].mode(dropna=True)
        if len(mode_val) > 0:
            df["性别_bin"] = df["性别_bin"].fillna(mode_val.iloc[0])

    # 3) 数值列
    for col in NUM_COLS:
        if col in df.columns:
            if col == "入院时发现胆囊息肉时间（月）":
                df[col + "_num"] = df[col].apply(parse_numeric_cell)
                # 对 "<1" / "＜1" 已经在 parse_numeric_cell 做了 0.5；其余按数值
            elif col == "胆囊壁厚度":
                # "无" / 空 -> NaN；其它转数值
                df[col + "_num"] = df[col].apply(parse_numeric_cell)
            else:
                df[col + "_num"] = df[col].apply(parse_numeric_cell)
        else:
            df[col + "_num"] = np.nan  # 缺列也补出来，避免后续报错

    # 4) 多分类（有序 -> 整数编码，按你的要求）
    # ---- 先定义列名变量（一定要在使用前定义）----
    col_cnt   = "息肉个数（单发或个数）"
    col_base  = "基底情况（带蒂、宽基地）"
    col_echo  = "回声强度（强、中等、弱）"
    col_shape = "息肉形状（球状、桑葚状、乳头状、结节状）"
    if col_shape not in df.columns and "息肉形状" in df.columns:
        col_shape = "息肉形状"

    # 4.1 息肉个数（单发或个数）
    if col_cnt in df.columns:
        df["息肉个数_ord"] = df[col_cnt].apply(
            lambda x: map_ordered(
                x, MAP_COUNT_ORDERED,
                priority=[">3", "＞3", "多个", "数个", "3", "２", "2", "1", "单发"]
            )
        )
    else:
        df["息肉个数_ord"] = np.nan

    # 4.2 基底情况（带优先级：窄 > 宽 > 带蒂）
    if col_base in df.columns:
        df["基底情况_ord"] = df[col_base].apply(
            lambda x: map_ordered(
                x, MAP_BASE_ORDERED,
                priority=["窄基底", "宽基底", "带蒂"]
            )
        )
    else:
        df["基底情况_ord"] = np.nan

    # 4.3 回声强度（强 > 稍强/稍高 > 中等/等回声 > 低/弱）
    if col_echo in df.columns:
        df["回声强度_ord"] = df[col_echo].apply(
            lambda x: map_ordered(
                x, MAP_ECHO_ORDERED,
                priority=["强", "稍强", "稍高", "中等", "等回声", "低", "弱"]
            )
        )
    else:
        df["回声强度_ord"] = np.nan

    # 4.4 息肉形状
    if col_shape in df.columns:
        df["息肉形状_ord"] = df[col_shape].apply(
            lambda x: map_ordered(x, MAP_SHAPE_ORDERED)
        )
    else:
        df["息肉形状_ord"] = np.nan

    # 4.5 息肉部位
    SITE2IDX = {"底部": 0, "体部": 1, "颈部": 2, "胆囊管": 3}
    SITE_PRIORITY = ["胆囊管", "颈部", "体部", "底部"]  # 多部位时的选择优先级（高->低）
    def site_onehot_index(text):
        s = normalize_str(text)
        if s == "" or s == "无":
            return np.nan
        # 按优先级寻找第一个命中的部位，输出对应的索引编号
        for token in SITE_PRIORITY:
            if token in s:
                return SITE2IDX[token]
        # 若没有命中词表，返回缺失
        return np.nan
    # 原：部位 multi-hot 生成的若干列 —— 整段删除
    # 新：单列 one-hot 索引
    col_site = "部位（底部、体部、颈部、胆囊管）"
    if col_site in df.columns:
        df["部位_onehot"] = df[col_site].apply(site_onehot_index)
    else:
        df["部位_onehot"] = np.nan


    # 5) 多选（multi-hot）

    # 5.2 合并消化系统症状 multi-hot
    col_sym = "合并消化系统症状（无、腹痛、腹胀、腹泻、消化不良）"
    sym_mh_cols = [f"症状_{t}" for t in SYMPTOMS_VOCAB]
    if col_sym in df.columns:
        def _sym_mh(s):
            s_norm = normalize_str(s)
            if s_norm in ("", "无"):
                return [0] * len(SYMPTOMS_VOCAB)
            return multi_hot(s_norm, SYMPTOMS_VOCAB)

        mh2 = df[col_sym].apply(lambda s: pd.Series(_sym_mh(s), index=sym_mh_cols))
        df = pd.concat([df, mh2], axis=1)
    else:
        for c in sym_mh_cols:
            df[c] = 0

    # 仅输出处理后的列
    processed_num_cols = [col + "_num" for col in NUM_COLS]
    sym_mh_cols = [f"症状_{t}" for t in SYMPTOMS_VOCAB]  # 原脚本已有

    out_cols = (
        ["patient_id", "性别_bin"]
        + processed_num_cols
        + ["息肉个数_ord", "基底情况_ord", "回声强度_ord", "息肉形状_ord"]
        + ["部位_onehot"]
        + sym_mh_cols
    )

    # 过滤存在列（防止 KeyError）
    out_cols = [c for c in out_cols if c in df.columns]

    def _to_na_or_norm_text(x):
        s = normalize_str(x)
        return np.nan if s == "" else s
    # —— 原始列 NaN 友好化（这些列会出现在 out_cols 里）——
    candidate_text_cols = set()

    # 原始二分类/文本列
    candidate_text_cols.add(BIN_COL)

    # 原始数值列（原列也做 NaN 友好，避免 Excel 看到奇怪字符串）
    for col in NUM_COLS:
        if col in df.columns:
            candidate_text_cols.add(col)

    # 多分类原始列
    col_cnt   = "息肉个数（单发或个数）"
    col_base  = "基底情况（带蒂、宽基地）"
    col_echo  = "回声强度（强、中等、弱）"
    col_shape = "息肉形状（球状、桑葚状、乳头状、结节状）"
    if col_shape not in df.columns and "息肉形状" in df.columns:
        col_shape = "息肉形状"
    for c in [col_cnt, col_base, col_echo, col_shape]:
        if c in df.columns:
            candidate_text_cols.add(c)

    # 多选原始列
    col_site = "部位（底部、体部、颈部、胆囊管）"
    col_sym  = "合并消化系统症状（无、腹痛、腹胀、腹泻、消化不良）"
    for c in [col_site, col_sym]:
        if c in df.columns:
            candidate_text_cols.add(c)

    # 统一清洗 → NaN
    for c in candidate_text_cols:
        try:
            df[c] = df[c].apply(_to_na_or_norm_text)
        except Exception:
            pass


    return df[out_cols].copy()

from pathlib import Path
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="清洗临床变量 Excel -> CSV/Parquet/Excel")
    parser.add_argument("--input", required=True, help="输入 Excel 路径")
    parser.add_argument("--sheet", default=None, help="Excel 工作表名或索引（默认首个）")
    parser.add_argument("--output_csv", default=None, help="输出 CSV 路径（UTF-8 BOM，Excel可直接打开）")
    parser.add_argument("--output_parquet", default=None, help="输出 Parquet 路径（可选）")
    parser.add_argument("--output_xlsx", default=None, help="输出 Excel 路径（.xlsx，可选）")
    args = parser.parse_args()

    # --- 读取 Excel：只读一个表 + NA 映射 ---
    sheet_name = 0 if args.sheet is None else (int(args.sheet) if str(args.sheet).isdigit() else args.sheet)
    df = pd.read_excel(
        args.input,
        sheet_name=sheet_name,
        header=0,
        engine="openpyxl",
        na_values=list(NA_STRINGS),  # "无"、"—"、"——"、"" -> NaN
        keep_default_na=True
    )
    if isinstance(df, dict):
        first_key = next(iter(df.keys()))
        print(f"[WARN] read_excel 返回了 dict，使用第一个工作表: {first_key}")
        df = df[first_key]

    cleaned = clean_dataframe(df)

    # --- 输出：根据参数写出 ---
    wrote_any = False
    if args.output_csv is None and args.output_parquet is None and args.output_xlsx is None:
        # 默认导出 CSV（UTF-8 BOM），Excel 可直接打开
        out_path = str(Path(args.input).with_suffix("")) + "_cleaned.csv"
        cleaned.to_csv(out_path, index=False, encoding="utf-8-sig", lineterminator="\n",
                    quoting=csv.QUOTE_MINIMAL)
        print(f"[OK] Saved cleaned CSV to: {out_path}")
        wrote_any = True
    else:
        if args.output_csv:
            cleaned.to_csv(args.output_csv, index=False, encoding="utf-8-sig", lineterminator="\n",
                        quoting=csv.QUOTE_MINIMAL)
            print(f"[OK] Saved cleaned CSV to: {args.output_csv}")
            wrote_any = True
        if args.output_parquet:
            cleaned.to_parquet(args.output_parquet, index=False)
            print(f"[OK] Saved cleaned Parquet to: {args.output_parquet}")
            wrote_any = True
        if args.output_xlsx:
            cleaned.to_excel(args.output_xlsx, index=False, engine="openpyxl")
            print(f"[OK] Saved cleaned XLSX to: {args.output_xlsx}")
            wrote_any = True

    if not wrote_any:
        print("[WARN] 未写出任何文件（请提供 --output_csv 或 --output_parquet 或 --output_xlsx）")



if __name__ == "__main__":
    main()
