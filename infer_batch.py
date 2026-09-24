import os
# 临时设置 Hugging Face 国内镜像源
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import argparse
import torch
import json
from pathlib import Path
from peft import PeftModel
from config import *
from model import Qwen3FactoryVLA
from PIL import Image

# 与训练同源：单自 config.TRAIN_PROMPT（含类别中文语义），推理不可另编
INFER_PROMPT = TRAIN_PROMPT

# 与训练一致的混合精度（FP32 主权重 + bf16 autocast）
if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    AMP_DTYPE = torch.bfloat16
elif torch.cuda.is_available():
    AMP_DTYPE = torch.float16
else:
    AMP_DTYPE = None

def load_model(tag="best"):
    """新保存格式：base 模型 + LoRA adapter + heads（不再是 full_model.pth 全量）"""
    adapter_dir = os.path.join(SAVE_DIR, f"lora_adapter_{tag}")
    heads_path = os.path.join(SAVE_DIR, f"heads_{tag}.pth")
    if not os.path.isdir(adapter_dir) or not os.path.exists(heads_path):
        raise FileNotFoundError(
            f"缺少 checkpoint：{adapter_dir} 或 {heads_path}"
            f"（--tag 可选 best/final，需先跑完 train.py）")

    model = Qwen3FactoryVLA(
        qwen_path=QWEN3_PATH,
        num_classes=NUM_CLASSES,
        load_in_4bit=LOAD_IN_4BIT,
        class_texts=CLASS_QUERY_TEXTS
    )
    # base 2B + 叠加载 LoRA adapter（约 6MB）
    model.qwen = PeftModel.from_pretrained(model.qwen, adapter_dir)

    heads = torch.load(heads_path, map_location="cpu")
    model.readout.load_state_dict(heads['readout'])
    model.vis_proj.load_state_dict(heads['vis_proj'])

    model.to(DEVICE)
    model.eval()
    print(f"已加载: {adapter_dir} + {heads_path}")
    return model

@torch.no_grad()
def predict_batch(model, image_paths, batch_size=8):
    """批量推理：图像逐张过 CLIP processor 后拼 batch，
    文本为固定指令（各样本同长，padding=longest 实际不产生 pad）"""
    results = []
    for i in range(0, len(image_paths), batch_size):
        chunk = image_paths[i:i + batch_size]
        pixels = []
        for p in chunk:
            img = Image.open(p).convert("RGB")
            pixels.append(model.vis_processor(img, return_tensors="pt").pixel_values)
        pixel_values = torch.cat(pixels, dim=0).to(DEVICE)

        tok = model.tokenizer(
            [INFER_PROMPT] * len(chunk),
            max_length=MAX_SEQ_LEN,
            padding="longest",
            truncation=True,
            return_tensors="pt"
        ).to(DEVICE)

        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE,
                            enabled=AMP_DTYPE is not None):
            logits = model(pixel_values, tok.input_ids, tok.attention_mask)

        probs = logits.float().softmax(dim=-1)
        preds = probs.argmax(dim=-1)
        for j, p in enumerate(chunk):
            idx = int(preds[j])
            results.append({
                "image": Path(p).name,
                "action_label": idx,
                "action": ACTION_CLASSES[idx],
                "prob": round(float(probs[j, idx]), 4),
            })
        print(f"  批次 {i // batch_size + 1}: {len(chunk)} 张完成")
    return results

def main():
    parser = argparse.ArgumentParser(description="Qwen3-CLASS 批量推理（对应新 LoRA adapter 保存格式）")
    parser.add_argument("--image-dir", default="dataset/test_data/val", help="待推理图片目录")
    parser.add_argument("--output", default="dataset/test_data/val_valid.json", help="结果 json 路径")
    parser.add_argument("--tag", default="best", choices=["best", "final"], help="checkpoint 后缀")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--limit", type=int, default=0, help="仅推理前 N 张（0=全部，调试用）")
    args = parser.parse_args()

    model = load_model(args.tag)

    image_dir = Path(args.image_dir)
    image_files = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))
    if args.limit > 0:
        image_files = image_files[:args.limit]
    print(f"共 {len(image_files)} 张图，batch_size={args.batch_size}")

    results = predict_batch(model, [str(p) for p in image_files], args.batch_size)

    output_path = args.output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"结果已写入: {output_path}")

    pred_counts = [0] * len(ACTION_CLASSES)
    for r in results:
        pred_counts[r["action_label"]] += 1
    pred_dist = ", ".join(f"{ACTION_CLASSES[i]}={pred_counts[i]}" for i in range(len(ACTION_CLASSES)))
    print(f"\n预测分布: {pred_dist}")

if __name__ == "__main__":
    main()
