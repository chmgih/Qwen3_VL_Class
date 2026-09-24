import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 注意：部分 vGPU 环境不支持 CUDA VMM，禁止 expandable_segments:True
# （否则在 .to(device)/首次大分配时报 CUDA driver error: invalid argument）
os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)

import torch
import gc
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from peft import get_peft_model, PeftModel
from config import *          # LoRA 配置统一使用 config.py 中的 LORA_CONFIG，禁止在此局部覆盖
from model import Qwen3FactoryVLA
from dataset import FactoryVLA
import json
import math
import shutil
import time

# 架构标识：改动模型结构（如读出头替换）时必须同步递增，旧中间状态会被自动归档弃用
ARCH_TAG = "class-query-readout-v1"

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.8)
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()
gc.collect()

torch.manual_seed(SEED)

# ===== 混合精度：FP32 主权重 + bf16 autocast（bf16 无需 GradScaler）=====
if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    AMP_DTYPE, USE_SCALER = torch.bfloat16, False   # 本环境走这支
elif torch.cuda.is_available():
    AMP_DTYPE, USE_SCALER = torch.float16, True     # V100 等无 bf16 的卡退化为 fp16+Scaler
else:
    AMP_DTYPE, USE_SCALER = None, False             # CPU 不开 AMP

def autocast_ctx():
    return torch.autocast(device_type="cuda", dtype=AMP_DTYPE,
                          enabled=AMP_DTYPE is not None)

scaler = torch.amp.GradScaler("cuda", enabled=USE_SCALER)

# ======================== 容器资源监控 ========================
_GIB = 1.0 / (1 << 30)

def cgroup_mem_gb():
    """容器 cgroup 内存用量与限额（GiB）。cgroup v2 优先，失败退 v1。
    注意：/proc/meminfo 是宿主机数值（假象）；memory.current 含 page cache，
    读图多了会偏高属正常，配合 RSS 一起看。"""
    try:
        with open("/sys/fs/cgroup/memory.current") as f:
            cur = int(f.read())
        with open("/sys/fs/cgroup/memory.max") as f:
            s = f.read().strip()
        lim = -1 if s == "max" else int(s)
    except Exception:
        try:
            with open("/sys/fs/cgroup/memory/memory.usage_in_bytes") as f:
                cur = int(f.read())
            with open("/sys/fs/cgroup/memory/memory.limit_in_bytes") as f:
                lim = int(f.read())
            if lim >= (1 << 60):
                lim = -1
        except Exception:
            return None, None
    return cur * _GIB, (lim * _GIB if lim > 0 else None)

def proc_rss_gb():
    """当前主进程 RSS（GiB）；DataLoader worker 的 RSS 不含在内。"""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / (1 << 20)
    except Exception:
        pass
    return None

def gpu_mem_gb():
    """本进程显存（GiB）：vGPU 配额内 torch 视角的已分配/保留值。"""
    if not torch.cuda.is_available():
        return None, None
    return (torch.cuda.memory_allocated() * _GIB,
            torch.cuda.memory_reserved() * _GIB)

def resource_kv():
    """tqdm postfix 用的资源字典：gpu=已分配/保留，mem=cgroup 用量/限额"""
    kv = {}
    _ga, _gr = gpu_mem_gb()
    if _ga is not None:
        kv["gpu"] = f"{_ga:.1f}/{_gr:.1f}G"
    _c, _l = cgroup_mem_gb()
    if _c is not None:
        kv["mem"] = f"{_c:.1f}/{_l:.1f}G" if _l else f"{_c:.1f}G"
    return kv

def resource_suffix(sep=", "):
    """拼合一行的 gpu/mem/rss 资源字符串，取不到的项自动省略"""
    parts = []
    _ga, _gr = gpu_mem_gb()
    if _ga is not None:
        parts.append(f"gpu={_ga:.1f}/{_gr:.1f}G")
    _c, _l = cgroup_mem_gb()
    if _c is not None:
        parts.append(f"mem={_c:.1f}/{_l:.1f}G" if _l else f"mem={_c:.1f}G")
    _rss = proc_rss_gb()
    if _rss is not None:
        parts.append(f"rss={_rss:.1f}G")
    return (sep + " ".join(parts)) if parts else ""

def compute_class_weights(json_file, img_dir, min_weight=0.7):
    """类别权重分配
    - 频率权重：1/sqrt(count)，样本越少权重越高（平方根倒数，比纯反比平滑）
    - 尺寸权重：1/sqrt(sqrt(平均像素数))，图像越小权重越高（四次方根平滑）
    - 归一化使均值为 1 后，以 1 为中心做偏差等比缩放设下限：
      硬截断 max(w, 0.7) 会把所有低于下限的类截成同一值丢掉排序信息，
      缩放保证最小类恰好落在下限、均值仍为 1，不扰动损失量级
    """
    from PIL import Image
    import numpy as np

    with open(json_file, encoding='utf-8') as f:
        data = json.load(f)

    # 统计每个类别的样本数和图像尺寸
    class_samples = {i: [] for i in range(NUM_CLASSES)}
    for d in data:
        label = d['action_label']
        img_path = os.path.join(img_dir, d['image'])
        if os.path.exists(img_path):
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                    class_samples[label].append(w * h)
            except Exception:
                pass

    # 频率权重（空类按 1 个样本处理，避免除零）
    class_counts = np.array([max(len(class_samples[i]), 1) for i in range(NUM_CLASSES)],
                            dtype=np.float64)
    freq_weights = 1.0 / np.sqrt(class_counts)

    # 尺寸权重：空类回退到已有类的均值（中性值，不人为放大权重）
    observed = [np.mean(class_samples[i]) for i in range(NUM_CLASSES) if class_samples[i]]
    fallback = float(np.mean(observed)) if observed else 1.0
    avg_sizes = np.array([np.mean(class_samples[i]) if class_samples[i] else fallback
                          for i in range(NUM_CLASSES)], dtype=np.float64)
    size_weights = 1.0 / np.sqrt(np.sqrt(avg_sizes))

    # 组合权重，归一化使平均权重为 1.0
    raw_weights = freq_weights * size_weights
    if raw_weights.sum() > 0:
        normalized_weights = raw_weights / raw_weights.mean()
    else:
        normalized_weights = np.ones(NUM_CLASSES)

    # 权重下限：以 1 为中心的偏差等比缩放（不做硬截断）
    raw_min = normalized_weights.min()
    if raw_min < min_weight and raw_min < 1.0:
        # min_weight < 1 时分子分母同为负，scale > 0
        scale = (min_weight - 1.0) / (raw_min - 1.0)
        final_weights = 1.0 + (normalized_weights - 1.0) * scale
    elif raw_min < min_weight:
        # min_weight >= 1 的非常规情况退回硬截断
        final_weights = np.maximum(normalized_weights, min_weight)
    else:
        # 无类低于下限，无需处理
        final_weights = normalized_weights.copy()

    # 打印统计信息
    for i in range(NUM_CLASSES):
        print(f"类别 {ACTION_CLASSES[i]}: 样本数={int(class_counts[i])}, "
              f"平均尺寸={avg_sizes[i]:.0f}, 最终权重={final_weights[i]:.4f}")
    print(f"权重均值={final_weights.mean():.4f}（恒为1），最小值={final_weights.min():.4f}")

    return torch.tensor(final_weights, dtype=torch.float32)

def _atomic_torch_save(obj, path):
    """先写 .tmp 再原子替换，避免中断留下半损 checkpoint"""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)

def save_checkpoint(model, tag):
    """保存 LoRA adapter（几 MB）+ 自训练模块，全部原子替换，不再全量保存 2B base 权重"""
    os.makedirs(SAVE_DIR, exist_ok=True)
    final_dir = os.path.join(SAVE_DIR, f"lora_adapter_{tag}")
    tmp_dir = final_dir + ".tmp"
    old_dir = final_dir + ".old"
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)
    model.qwen.save_pretrained(tmp_dir)          # 先写临时目录
    if os.path.isdir(old_dir):
        shutil.rmtree(old_dir)
    if os.path.isdir(final_dir):
        os.rename(final_dir, old_dir)            # 旧版挪走而非删，窗口期内仍有可读备份
    os.rename(tmp_dir, final_dir)
    if os.path.isdir(old_dir):
        shutil.rmtree(old_dir)
    _atomic_torch_save({
        'readout': model.readout.state_dict(),
        'vis_proj': model.vis_proj.state_dict(),
    }, os.path.join(SAVE_DIR, f"heads_{tag}.pth"))

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch in loader:
        pixel_values = batch["pixel_values"].to(DEVICE, non_blocking=True)
        input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
        attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
        label = batch["label"].to(DEVICE, non_blocking=True)

        with autocast_ctx():
            logits = model(pixel_values, input_ids, attention_mask)
        loss = criterion(logits.float(), label)

        total_loss += loss.item() * label.size(0)
        correct += (logits.float().argmax(dim=1) == label).sum().item()
        total += label.size(0)
    return correct / max(total, 1), total_loss / max(total, 1)

def restore_optimizer_state(optimizer, saved_optimizer_state, named_params):
    """按参数名恢复优化器状态。
    背景：get_peft_model 新建与 PeftModel.from_pretrained 加载的 LoRA 参数
    注册顺序可能不同，按索引匹配的 load_state_dict 会报 param group 尺寸
    不符或直接错位；这里用保存时的名字->本次参数重建 state。失败则退回全新优化器。"""
    try:
        name2param = dict(named_params)
        name_by_index = saved_optimizer_state.get('param_names', [])
        restored = {}
        for idx, s in saved_optimizer_state['state'].items():
            if idx < len(name_by_index):
                p = name2param.get(name_by_index[idx])
                if p is not None:
                    s = dict(s)
                    if isinstance(s.get('step'), int):
                        s['step'] = torch.tensor(s['step'])
                    restored[p] = s
        if len(restored) != len(saved_optimizer_state['state']):
            raise RuntimeError("参数名不匹配，无法安全恢复优化器状态")
        for g in optimizer.param_groups:
            g['lr'] = saved_optimizer_state['param_groups'][0]['lr']
            g['weight_decay'] = saved_optimizer_state['param_groups'][0]['weight_decay']
        for p, s in restored.items():
            s = {k: (v.to(p.device) if torch.is_tensor(v) else v) for k, v in s.items()}
            optimizer.state[p] = s
        print(f"[续训] Adam 矩量等优化器状态已按参数名恢复（{len(restored)} 个张量）")
        return True
    except Exception as e:
        print(f"[警告] 优化器状态恢复失败（{e}），将以全新优化器继续训练")
        return False

def _resume_compatible(adapter_last, heads_last, state_last):
    """续训前预检：heads 键名与架构标签必须和当前代码一致。
    不兼容（如旧 control_head 架构的落盘）则整套归档，本次从头训练，
    避免加载到一半 KeyError 崩溃。"""
    try:
        heads = torch.load(heads_last, map_location="cpu")
        st = torch.load(state_last, map_location="cpu")
    except Exception as e:
        print(f"[续训检查] 中间状态读取失败（{e}），本次从头训练")
        return False
    if 'readout' in heads and 'vis_proj' in heads and st.get('arch') == ARCH_TAG:
        return True
    archive = os.path.join(SAVE_DIR,
                           f"incompatible_last_{time.strftime('%Y%m%d_%H%M')}")
    try:
        os.makedirs(archive, exist_ok=True)
        for p in (adapter_last, heads_last, state_last):
            if os.path.exists(p):
                shutil.move(p, os.path.join(archive, os.path.basename(p)))
        print(f"[续训检查] last 为旧架构状态（heads 键: {sorted(heads.keys())}，"
              f"arch={st.get('arch')!r} ≠ {ARCH_TAG!r}），已归档至 {archive}/，本次从头训练")
    except Exception as e:
        print(f"[续训检查] 旧状态不兼容且归档失败（{e}），本次从头训练")
    return False

def main():
    # ===== 续训检测：存在上一轮保存的 last 中间状态则接着练（config.RESUME=False 可关）=====
    adapter_last = os.path.join(SAVE_DIR, "lora_adapter_last")
    heads_last = os.path.join(SAVE_DIR, "heads_last.pth")
    state_last = os.path.join(SAVE_DIR, "train_state_last.pth")
    resume = (bool(RESUME) and os.path.isdir(adapter_last)
              and os.path.exists(heads_last) and os.path.exists(state_last))
    if resume:
        resume = _resume_compatible(adapter_last, heads_last, state_last)
    if RESUME and not resume:
        print("未发现 last 中间状态，从头开始训练")

    # 加载模型（FP32 主权重，前向计算走 bf16 autocast）
    model = Qwen3FactoryVLA(
        qwen_path=QWEN3_PATH,
        num_classes=NUM_CLASSES,
        load_in_4bit=LOAD_IN_4BIT,
        class_texts=CLASS_QUERY_TEXTS
    )
    print(f"模型主权重类型: {next(model.parameters()).dtype}，"
          f"训练精度: {AMP_DTYPE}（GradScaler: {USE_SCALER}）")

    # 冻结视觉塔（CLIP；Qwen 自带视觉塔已在 model.py 中冻结）
    for param in model.vis_encoder.parameters():
        param.requires_grad = False

    # 应用 LoRA（统一使用 config.py 的 LORA_CONFIG）；续训时加载上次 adapter 而非随机初始化
    if resume:
        # is_trainable=True 必须显式传：默认会按推理模式加载，LoRA 全部被冻结
        model.qwen = PeftModel.from_pretrained(model.qwen, adapter_last, is_trainable=True)
    else:
        model.qwen = get_peft_model(model.qwen, LORA_CONFIG)
    model.train().to(DEVICE)
    model.qwen.print_trainable_parameters()

    # 恢复 heads / 训练进度 / RNG（在创建 DataLoader 之前恢复 RNG，使 shuffle 顺序接续）
    start_epoch, best_val, bad_epochs = 0, 0.0, 0
    train_state = None
    if resume:
        heads = torch.load(heads_last, map_location=DEVICE)
        model.readout.load_state_dict(heads['readout'])
        model.vis_proj.load_state_dict(heads['vis_proj'])
        train_state = torch.load(state_last, map_location="cpu")
        start_epoch = train_state['epoch']
        best_val = train_state['best_val']
        bad_epochs = train_state['bad_epochs']
        torch.set_rng_state(train_state['rng'])
        print(f"[续训] 从第 {start_epoch+1} 轮继续（best val={best_val:.4f}，"
              f"已停滞 {bad_epochs} 轮），heads/RNG 已恢复")

    # 数据集：切分训练/验证
    dataset = FactoryVLA(
        json_file=DATA_JSON,
        img_dir=IMAGE_DIR,
        image_processor=model.vis_processor,
        tokenizer=model.tokenizer,
        max_len=MAX_SEQ_LEN
    )
    n_val = max(1, int(len(dataset) * VAL_SPLIT))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED)
    )
    print(f"训练集 {n_train} / 验证集 {n_val}（seed={SEED}）")

    loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=8, pin_memory=True, drop_last=True,
                        persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True,
                            persistent_workers=True)

    # 损失和优化器（只收集可训练参数：LoRA + vis_proj + control_head）
    class_weights = compute_class_weights(DATA_JSON, IMAGE_DIR).to(DEVICE)
    print(f"类别权重: {class_weights} (dtype: {class_weights.dtype})")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    named_trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    trainable_params = [p for _, p in named_trainable]
    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=0.01)
    if resume and 'optimizer' in train_state:
        restore_optimizer_state(optimizer, train_state['optimizer'], named_trainable)

    # LR 调度：按 step 线性 warmup + cosine 退火至 LR_MIN（后期恒 3e-4 会在收敛区震荡）
    steps_per_epoch = max(1, len(loader))
    total_steps = steps_per_epoch * max(1, EPOCHS)
    warmup_steps = max(1, steps_per_epoch * max(0, WARMUP_EPOCHS))
    lr_floor = LR_MIN / LR

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return lr_floor + (1.0 - lr_floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    if resume and 'scheduler' in train_state:
        scheduler.load_state_dict(train_state['scheduler'])
        print("[续训] LR 调度器状态已恢复（warmup/退火进度接续）")

    # 训练循环
    accum = max(1, GRADIENT_ACCUMULATION_STEPS)
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        true_label_counts = [0] * NUM_CLASSES
        true_positive_counts = [0] * NUM_CLASSES
        pred_counts = [0] * NUM_CLASSES
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(pbar):
            pixel_values = batch["pixel_values"].to(DEVICE, non_blocking=True)
            input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
            attention_mask = batch["attention_mask"].to(DEVICE, non_blocking=True)
            label = batch["label"].to(DEVICE, non_blocking=True)

            # 前向：autocast 只包模型，损失在 autocast 外用 FP32 计算
            with autocast_ctx():
                logits = model(pixel_values, input_ids, attention_mask)
            loss = criterion(logits.float(), label)

            # 反向（梯度累积：反传缩放后的 loss，统计用原始 loss）
            scaler.scale(loss / accum).backward()

            if (step + 1) % accum == 0 or (step + 1) == len(loader):
                if USE_SCALER:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            # 统计信息
            loss_val = loss.item()
            total_loss += loss_val
            pred = logits.float().argmax(dim=1)
            correct += (pred == label).sum().item()
            total += label.size(0)

            label_np = label.cpu().numpy()
            pred_np = pred.cpu().numpy()
            for lbl, p in zip(label_np, pred_np):
                true_label_counts[lbl] += 1
                if p == lbl:
                    true_positive_counts[lbl] += 1
                if p < len(pred_counts):  # 防止索引错误
                    pred_counts[p] += 1

            # 容器级实时资源占用：gpu=已分配/保留（vGPU 配额内 torch 视角），
            # mem=cgroup 用量/限额（含 page cache）
            postfix = dict(loss=f"{loss_val:.4f}", acc=f"{correct/total:.4f}",
                           lr=f"{optimizer.param_groups[0]['lr']:.6f}",
                           **resource_kv())
            pbar.set_postfix(**postfix)

        train_acc = correct / total
        print(f"Epoch {epoch+1} 训练 avg loss: {total_loss/len(loader):.4f}, "
              f"acc: {train_acc:.4f}{resource_suffix()}")
        pred_dist = ", ".join([f"{ACTION_CLASSES[i]}={pred_counts[i]}" for i in range(NUM_CLASSES)])
        print(f"预测分布: {pred_dist}")

        # 计算并打印各类别的召回率（训练集）
        recall_list = []
        for i in range(NUM_CLASSES):
            if true_label_counts[i] > 0:
                recall = true_positive_counts[i] / true_label_counts[i]
            else:
                recall = 0.0
            recall_list.append(recall)
        recall_dist = ", ".join([f"{ACTION_CLASSES[i]}={recall_list[i]:.4f}" for i in range(NUM_CLASSES)])
        print(f"召回率分布: {recall_dist}")

        # ===== 验证集评估：best 保存与早停都以 val acc 为准 =====
        val_acc, val_loss = evaluate(model, val_loader, criterion)
        print(f"Epoch {epoch+1} 验证 loss: {val_loss:.4f}, acc: {val_acc:.4f} (best: {best_val:.4f})")

        if val_acc > best_val:
            best_val = val_acc
            bad_epochs = 0
            save_checkpoint(model, "best")
            print(f"最佳模型已保存（val acc: {best_val:.4f}）-> {SAVE_DIR}/")
        else:
            bad_epochs += 1

        # 每轮落盘 last 中间状态（权重+优化器+调度器+进度+RNG），供中断后自动续训
        save_checkpoint(model, "last")
        # 记录参数名->索引映射，恢复时按名字对齐（参数注册顺序不保证跨方式一致）
        opt_state = optimizer.state_dict()
        opt_state['param_names'] = [n for n, _ in named_trainable]
        _atomic_torch_save({'arch': ARCH_TAG,
                            'epoch': epoch + 1, 'best_val': best_val,
                            'bad_epochs': bad_epochs,
                            'optimizer': opt_state,
                            'scheduler': scheduler.state_dict(),
                            'rng': torch.get_rng_state()}, state_last)

        if bad_epochs >= EARLY_STOP_PATIENCE:
            print(f"验证集准确率连续 {EARLY_STOP_PATIENCE} 轮未提升，提前终止训练")
            break

    # 保存最终模型
    save_checkpoint(model, "final")
    print(f"训练完成，模型已保存！best val acc: {best_val:.4f}，目录: {SAVE_DIR}/")

if __name__ == "__main__":
    main()
