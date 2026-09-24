# Qwen3-VL 图像分类（LoRA 微调）

基于 [Qwen3-VL-2B-Instruct](https://modelscope.cn/models/Qwen/Qwen3-VL-2B-Instruct) 大模型的多模态图像分类训练/推理框架。图像经 CLIP 视觉编码器提取特征后注入 Qwen 语言模型，配合轻量 LoRA 微调与「类别名查询读出头」（ClassQueryReadout）完成场景目标分类（示例任务为 `person` / `car` 二分类）。

## 特性

- **冻结视觉塔 + LoRA**：冻结 Qwen 原生视觉塔与 CLIP 编码器，仅微调注意力层 `q_proj` / `v_proj`，保留模型原生能力。
- **方案 B 读出头**：类别中文名+英文名经 Qwen 词嵌入通道注入预训练先验，作为读出头 query，替代「末位 token + 线性分类头」，避免注意力稀释。
- **混合精度**：FP32 主权重 + bf16 autocast（无 bf16 的卡自动退化 fp16 + GradScaler）。
- **自动续训**：每轮落盘中间状态（权重 / 优化器矩量 / 调度器 / RNG），中断后按参数名安全接续。
- **原子保存**：checkpoint 先写临时目录再原子替换，避免中断留下半损文件。
- **类别均衡加权**：按频率与图像尺寸自动计算损失权重，缓解类别不平衡。

## 环境依赖

```bash
pip install torch transformers peft modelscope pillow tqdm numpy
```

需要一个 CUDA 环境用于训练；HF 权重下载默认走国内镜像 `https://hf-mirror.com`。

## 目录结构

```
.
├── config.py           # 全局配置（模型路径、超参、LoRA 配置、prompt 单源）
├── dataset.py          # Dataset 实现（图像 + 文本指令 + 标签）
├── model.py            # Qwen3FactoryVLA 模型（CLIP 分支 + Qwen + 读出头）
├── model_down.py       # 基座模型下载脚本
├── train.py            # 训练主脚本（含评估、续训、早停）
├── infer_batch.py      # 批量推理脚本
└── dataset/test_data/
    ├── train.json      # 标注文件
    └── train/          # 图像目录
```

## 数据格式

`dataset/test_data/train.json` 为标注列表，每条包含：

```json
[
    {
        "image": "fram_00001_crop_0.jpg",
        "state": "person",
        "action_label": 0
    },
    {
        "image": "fram_00001_crop_2.jpg",
        "state": "car",
        "action_label": 1
    }
]
```

- `image`：相对于 `IMAGE_DIR` 的图像文件名
- `action_label`：类别索引，需与 `config.py` 中 `ACTION_CLASSES` 顺序对应

## 快速开始

### 1. 下载基座模型

```bash
python model_down.py
```

### 2. 准备数据

将标注写入 `dataset/test_data/train.json`，图像放入 `dataset/test_data/train/`，并按实际情况修改 `config.py`：

| 配置项 | 说明 |
|--------|------|
| `ACTION_CLASSES` | 类别名列表（顺序即标签索引） |
| `DATA_JSON` / `IMAGE_DIR` | 标注文件与图像目录路径 |
| `SAVE_DIR` | checkpoint 保存目录 |
| `EPOCHS` / `BATCH_SIZE` / `LR` | 训练轮数 / 批大小 / LoRA 学习率 |
| `RESUME` | 是否自动续训 |

### 3. 训练

```bash
python train.py
```

每轮在验证集评估，最优结果保存为 `SAVE_DIR/lora_adapter_best` + `SAVE_DIR/heads_best.pth`，训练结束另存 `final`。

### 4. 批量推理

```bash
python infer_batch.py --image-dir dataset/test_data/val --tag best --output dataset/test_data/val_valid.json
```

输出 JSON 每条含 `image` / `action_label` / `action` / `prob`，并打印预测类别分布。

## 说明

- 训练与推理 prompt 统一引自 `config.TRAIN_PROMPT`，推理侧不可另编，否则与训练分布不一致。
- `LOAD_IN_4BIT` 对 2B 小模型不建议开启（视觉特征会退化）。
- 改动模型结构时需同步递增 `train.py` 中的 `ARCH_TAG`，旧中间状态会被自动归档弃用。
