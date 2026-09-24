import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from config import TRAIN_PROMPT

class FactoryVLA(Dataset):
    def __init__(self, json_file, img_dir, image_processor, tokenizer, max_len=256):
        with open(json_file, encoding='utf-8') as f:
            self.data = json.load(f)
        self.img_dir = img_dir
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        # 图像
        img = Image.open(os.path.join(self.img_dir, d['image'])).convert('RGB')
        pixel_values = self.image_processor(
            img, return_tensors="pt"
        ).pixel_values.squeeze(0)

        # 文本指令：含类别中文语义（方向1），单源自 config.TRAIN_PROMPT，
        # 推理侧必须引用同一常量（原来是空串/无语义短句，LLM 分支无从调用预训练知识）
        prompt = TRAIN_PROMPT
        tok = self.tokenizer(
            prompt,
            max_length=self.max_len,
            padding="longest",   # 原来 max_length=256 会把尾部填成 pad，取末尾 hidden 时拿到的是 pad 位
            truncation=True,
            return_tensors="pt"
        )

        # 标签：控制指令编号
        label = torch.tensor(d['action_label'], dtype=torch.long)

        return {
            "pixel_values": pixel_values,
            "input_ids": tok.input_ids.squeeze(0),
            "attention_mask": tok.attention_mask.squeeze(0),
            "label": label
        }