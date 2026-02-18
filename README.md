# GB-DualmodalNet

## Pipeline

1. **EMR API 拉取病历 JSON**
2. **结构化变量 → 临床语义文本（Prompt）**
3. **文本编码：PubMedBERT**
4. **图像分支：超声图像（可选 ROI/Mask 引导）→ VGG16 features**
5. **融合：Cross-Modal Attention**
6. **输出：三分类 logits / 概率**

---

## Repository Layout

```text
.
├── main.py
├── requirements.txt
├── train.py
├── infer.py
├── data/
│   └── dataset.py
├── models/
│   ├── gb_dualmodal.py
│   └── segmentation.py
└── utils/
    ├── api_client.py
    └── clinical_prompt.py
```

---

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> 首次运行会自动下载 `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract` 权重（HuggingFace Transformers）。

---

## EMR API Contract (必须满足)

### Endpoint

* `GET {BASE_URL}/records/{patient_id}`

### Response JSON Schema（字段名与类型必须一致）

```json
{
  "age": 52,
  "sex": 1,
  "long_diameter": 12.3,
  "short_diameter": 7.8,
  "number_of_polyps": 2,
  "base_morphology": 1,
  "wall_thickening": 0,
  "echogenicity": 0,
  "location": 2,
  "detection_time_months": 6
}
```

字段含义（与 `utils/clinical_prompt.py` 映射一致）：

* `sex`: 0 female, 1 male
* `base_morphology`: 0 pedunculated, 1 sessile
* `wall_thickening`: 0 no, 1 yes
* `echogenicity`: 0 hyperechoic, 1 hypoechoic
* `location`: 0 fundus, 1 body, 2 neck
* 直径单位：mm；检测时长单位：months

---

## Clinical Prompt Engineering (CPE)

临床文本由 `utils/clinical_prompt.py::build_prompt(record)` 生成，输出为一条自然语言描述，例如：

```text
52 year old male patient with 2 polyps, long diameter 12.3 mm, short diameter 7.8 mm, sessile broad base, no gallbladder wall thickening, hyperechoic lesion, lesion at neck, polyp detected for 6 months.
```

---

## Image Input

`data/dataset.py` 期望输入为：

* `images[idx]`: `H×W×3` 的 BGR/RGB numpy array（`uint8` 或可转 `float32`）
* 内部会 resize 到 `224×224` 并归一化到 `[0,1]`，再转 `C×H×W`

---

## Training

```bash
python main.py
```

训练完成后会保存权重：

```text
gb_dualmodal.pth
```

---

## Inference

`infer.py` 会加载 `gb_dualmodal.pth`，输出 `softmax` 概率（3 类）：

```python
prob = run_inference(device)
```

---

## Notes on ROI / Segmentation

`models/segmentation.py` 提供了一个冻结的分割模型包装器，用于加载外部训练好的分割 checkpoint 并输出 mask。若你在数据构建阶段已经有 ROI mask（例如来自 nnU-Net v2），建议在数据输入侧将 ROI 引导信息融合进图像（例如 mask crop/抑制背景）后再送入分类网络，以保证与论文路线一致。

---

## Reproducibility

* 建议固定随机种子、固定训练/测试划分并记录版本信息（Python/PyTorch/Transformers）。
* API 侧务必返回**去标识化**字段；不要在 prompt 中引入任何直接识别信息（姓名、ID、电话、地址等）。

---

## Citation

如果你使用了本仓库或其思路，请在论文中引用对应方法学描述与模型框架来源。
