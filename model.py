import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoImageProcessor, AutoModel
from modelscope import Qwen3VLForConditionalGeneration

# ====================== 视觉投影层 ======================
class VisionProjector(nn.Module):
    def __init__(self, in_dim=1024, out_dim=2048):
        super().__init__()
        # FP32 主权重，autocast 下自动降精度
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x):
        return self.proj(x)

# ====================== 类别名查询读出头（方案B） ======================
class ClassQueryReadout(nn.Module):
    """替代“末位 token + 线性分类头”的匹配式读出：
    - queries：7 条类别名文本经 Qwen 词嵌入层均值池化，冻结（buffer）——
      类别先验从模型自己的词嵌入空间进入，而非拼进输入序列（避免注意力稀释）
    - keys/values：CLIP patch 特征投影 ⊕ Qwen 末端 hidden 全序列
    - 每类 query 对 token 集合做注意力匹配池化，共享打分器输出 7 logits
    泄漏安全：query 是与样本标签无关的类别先验，softmax 中 6 个不匹配类
    天然提供负梯度，不存在“从 prompt 抄答案”的捷径。"""
    def __init__(self, dim=2048, qk_dim=256, v_dim=512):
        super().__init__()
        self.w_q = nn.Linear(dim, qk_dim, bias=False)
        self.w_k = nn.Linear(dim, qk_dim, bias=False)
        self.w_v = nn.Linear(dim, v_dim, bias=False)
        self.out = nn.Linear(v_dim, 1)
        self.temp = nn.Parameter(torch.tensor(10.0))
        self.register_buffer("queries", torch.zeros(0, dim))

    def set_queries(self, queries):
        """[C, dim] fp32，冻结为 buffer（随 .to(device) 搬运，不进优化器）"""
        self.register_buffer("queries", queries.detach().clone().float())

    def forward(self, tokens):
        # 匹配计算统一在 FP32 做，避免 bf16 下点积抖动
        t = F.normalize(tokens.float(), dim=-1)        # [B, N, D]  末端 hidden norm 数千，先归一
        q = F.normalize(self.queries.float(), dim=-1)  # [C, D]
        k = self.w_k(t)                                # [B, N, QK]
        a = (self.w_q(q) @ k.transpose(1, 2)) * self.temp.clamp(-4.0, 13.0)
        attn = F.softmax(a, dim=-1)                    # [B, C, N]
        v = self.w_v(t)                                # [B, N, V]
        pooled = attn @ v                              # [B, C, V]
        return self.out(pooled).squeeze(-1)            # [B, C]

# ====================== VLA模型 ======================
class Qwen3FactoryVLA(nn.Module):
    def __init__(self, qwen_path, num_classes=3, load_in_4bit=False, class_texts=None):
        super().__init__()

        # ✅ 只加载一次 Qwen 模型（使用 ModelScope 的 Qwen3VL）
        # FP32 主权重；混合精度由训练脚本的 torch.autocast(bf16) 提供，不在这里降精度
        self.qwen = Qwen3VLForConditionalGeneration.from_pretrained(
            qwen_path,
            torch_dtype=torch.float32,
            device_map="auto" if load_in_4bit else None,
            trust_remote_code=True
        )

        # Qwen3-VL 自带视觉塔在本结构中不使用（图像走 CLIP 分支），直接冻结
        if hasattr(self.qwen.model, "visual"):
            self.qwen.model.visual.requires_grad_(False)

        # ✅ 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True)

        # ✅ 加载 CLIP 视觉编码器
        self.vis_processor = AutoImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.vis_encoder = AutoModel.from_pretrained(
            "openai/clip-vit-large-patch14"
        ).vision_model

        # 冻结视觉编码器
        for p in self.vis_encoder.parameters():
            p.requires_grad = False

        # ✅ 投影层（FP32 主权重）
        self.vis_proj = VisionProjector(in_dim=1024, out_dim=2048)

        # ✅ 方案B 读出头：类别名 query 必须与 num_classes 一一对应
        if not class_texts or len(class_texts) != num_classes:
            raise ValueError(f"class_texts 必须提供且长度等于 num_classes({num_classes})")
        self.readout = ClassQueryReadout(2048)
        self._build_class_queries(class_texts)

    def _build_class_queries(self, texts):
        """类别名 → Qwen 词嵌入 → 均值池化，作为冻结 query 先验"""
        emb = self.qwen.get_input_embeddings()
        queries = []
        for t in texts:
            ids = torch.tensor(self.tokenizer(t, add_special_tokens=False).input_ids)
            with torch.no_grad():
                queries.append(emb(ids).float().mean(dim=0))
        self.readout.set_queries(torch.stack(queries))  # [C, 2048]

    def forward(self, pixel_values, input_ids, attention_mask):
        # ✅ 图像特征处理（确保正确的数据类型）
        if pixel_values.dtype != torch.float32:
            pixel_values = pixel_values.to(torch.float32)

        # CLIP 编码（全冻结，无需建反向图；梯度经 vis_proj 自身权重回传不受影响）
        with torch.no_grad():
            vis_out = self.vis_encoder(pixel_values)
        img_pooled = vis_out.pooler_output        # [B, 1024] 给 LLM 注入用
        img_seq = vis_out.last_hidden_state       # [B, N, 1024] patch token，给读出头用

        # 投影到 LLM 维度，作为 1 个图像 token
        img_embed = self.vis_proj(img_pooled).unsqueeze(1)  # [B, 1, 2048]

        # ✅ 文本特征：词嵌入后在前面拼接图像 token（标准多模态输入）
        word_embeds = self.qwen.get_input_embeddings()(input_ids)  # [B, S, 2048]

        # 把图像 token 缩放到与词嵌入同量级（防劫持 LLM RMSNorm 路径）
        with torch.no_grad():
            target_norm = word_embeds.norm(dim=-1).mean()
        img_embed = img_embed * (target_norm / (img_embed.norm(dim=-1, keepdim=True) + 1e-6))
        img_embed = img_embed.to(dtype=word_embeds.dtype)
        inputs_embeds = torch.cat([img_embed, word_embeds], dim=1)  # [B, 1+S, 2048]

        img_mask = torch.ones(attention_mask.size(0), 1,
                              dtype=attention_mask.dtype, device=attention_mask.device)
        full_mask = torch.cat([img_mask, attention_mask], dim=1)

        # logits_to_keep=1 避免对全序列 × 15万词表算 lm_head 大张量
        outputs = self.qwen(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            output_hidden_states=True,
            logits_to_keep=1
        )

        # 方案B 读出：类别名 query attend 到 CLIP patch token ⊕ Qwen 序列 token
        img_kv = self.vis_proj(img_seq)             # [B, N, 2048]
        txt_kv = outputs.hidden_states[-1]          # [B, 1+S, 2048]
        logits = self.readout(torch.cat([img_kv, txt_kv], dim=1))
        return logits
