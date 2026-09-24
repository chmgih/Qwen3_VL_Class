import os
import torch

# 注意：部分 vGPU 环境不支持 CUDA VMM API，禁止设置
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True，
# 否则会在大分配时报 CUDA driver error: invalid argument
torch.cuda.empty_cache()

from peft import LoraConfig

QWEN3_PATH = "Qwen/Qwen3-VL-2B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_JSON = "dataset/test_data/train.json"
IMAGE_DIR = "dataset/test_data/train"
SAVE_DIR = "factory_lora_test"   # 保存目录（训练脚本会自动 makedirs）
RESUME = True   # 启动时若存在 last 中间状态则自动续训；想从头训删除 *last* 文件或改 False
VAL_SPLIT = 0.2                            # 验证集比例
SEED = 42
EARLY_STOP_PATIENCE = 10                    # 验证集准确率连续 N 轮不涨则停
MAX_GRAD_NORM = 1.0

ACTION_CLASSES = ["person", "car"]
NUM_CLASSES = len(ACTION_CLASSES)

# 训练/推理 prompt 单源（dataset.py / infer_batch.py 统一引用，禁止各写副本）。
# 实测短版收敛更快（长语义版会稀释单图像 token 信号），故默认短版；
# 若长训（>5 epoch）想再试语义版，换到下面注释行即可（需重训，旧 adapter 不兼容）
TRAIN_PROMPT = "观察图片中的场景，判断目标物的类别。"
# 方案B：类别名作为读出头 query（经 Qwen 词嵌入通道注入预训练先验），
# 中文名+英文名拼合，读出头用它替掉“末位token+control_head”，prompt 保持短版不变
ACTION_CLASSES_ZH = {"person": "人体", "car": "车"}
CLASS_QUERY_TEXTS = [f"{ACTION_CLASSES_ZH[c]} {c}" for c in ACTION_CLASSES]

# ====================== 核心训练参数（安全 + 稳定涨点）======================
BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 4
LR = 3e-4           # LoRA 专用学习率（比2e-5更适合小模型）3e-4
WARMUP_EPOCHS = 3   # LR 线性 warmup 轮数（按 step 计）
LR_MIN = 1e-5       # cosine 退火终点学习率
EPOCHS = 50         # 2B模型 10–20轮足够，500必崩
MAX_SEQ_LEN = 256
LOAD_IN_4BIT = False  # 2B模型千万别开4bit！视觉特征会炸！！！
FREEZE_VIT = True   # 冻结视觉编码器，只微调语言层对齐（VLA 最强涨点技巧）

# ====================== VLA 专用轻量 LoRA（不破坏原生能力）======================
# 全局唯一 LoRA 配置：训练脚本必须引用本对象，禁止在 train.py 内再局部定义覆盖
LORA_CONFIG = LoraConfig(
    r=8,                      # 越小越稳定，16足够2分类
    lora_alpha=16,
    target_modules=[
        "q_proj", "v_proj",    # 只微调注意力层，不碰FFN，保证视觉能力不变
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    use_dora=False,            # 小模型+小数据，Dora会学飞
)