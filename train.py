"""
train.py
Türkçe GPT modelini sıfırdan eğitir.

Kullanım:
    python train.py --preset small --data data/clean.txt --out checkpoints/
    python train.py --preset medium --epochs 10 --batch 16 --grad-accum 4
"""

import argparse
import math
import time
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast

from model import TINAConfig, TINAModel, PRESETS


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TextDataset(Dataset):
    """
    Ham metin dosyasını yükler, tokenize eder ve sabit uzunluklu
    bloklara böler (sliding window).
    """
    def __init__(
        self,
        path: str,
        tokenizer: PreTrainedTokenizerFast,
        block_size: int = 512,
        stride: int | None = None,       # None → stride = block_size (örtüşme yok)
    ):
        self.block_size = block_size
        self.stride     = stride or block_size

        print(f"[Dataset] Yükleniyor: {path}")
        text = Path(path).read_text(encoding="utf-8")
        ids  = tokenizer.encode(text)
        print(f"[Dataset] Token sayısı: {len(ids):,}")

        self.samples: list[list[int]] = []
        for start in range(0, len(ids) - block_size, self.stride):
            self.samples.append(ids[start : start + block_size + 1])  # +1 label shift için

        print(f"[Dataset] Örnek sayısı: {len(self.samples):,}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        chunk = torch.tensor(self.samples[idx], dtype=torch.long)
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


def collate_fn(batch, pad_id: int):
    xs, ys = zip(*batch)
    xs = torch.stack(xs)
    ys = torch.stack(ys)
    return xs, ys


# ---------------------------------------------------------------------------
# Learning rate scheduler — cosine + warmup
# ---------------------------------------------------------------------------

def get_lr(step: int, warmup_steps: int, max_steps: int, lr_max: float, lr_min: float) -> float:
    if step < warmup_steps:
        return lr_max * step / max(warmup_steps, 1)
    if step > max_steps:
        return lr_min
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Eğitim
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Cihaz: {device}")

    # --- Tokenizer ---
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer)
    vocab_size = len(tokenizer)
    pad_id     = tokenizer.pad_token_id or 1
    bos_id     = tokenizer.bos_token_id or 2
    eos_id     = tokenizer.eos_token_id or 3

    # --- Model config ---
    preset_kw = PRESETS[args.preset]
    cfg = TINAConfig(
        vocab_size=vocab_size,
        pad_token_id=pad_id,
        bos_token_id=bos_id,
        eos_token_id=eos_id,
        **preset_kw,
    )
    model = TINAModel(cfg).to(device)
    total_params = model.get_num_params()
    print(f"[Model] Preset: {args.preset}  |  Parametre: {total_params:,}")

    # --- Dataset & DataLoader ---
    dataset = TextDataset(args.data, tokenizer, block_size=cfg.n_ctx)
    loader  = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=lambda b: collate_fn(b, pad_id),
        drop_last=True,
    )

    # --- Optimizer ---
    # Ağırlık çürümesini (weight decay) yalnızca 2D parametrelere uygula
    decay_params  = [p for n, p in model.named_parameters() if p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params,   "weight_decay": args.wd},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=True,   # CUDA fused AdamW (PyTorch ≥2.0, daha hızlı)
    )

    # AMP (otomatik karışık hassasiyet)
    scaler   = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    autocast = torch.cuda.amp.autocast

    # Adım hesabı
    steps_per_epoch = len(loader) // args.grad_accum
    total_steps     = steps_per_epoch * args.epochs
    warmup_steps    = max(100, total_steps // 20)   # %5 warmup
    print(f"[Train] Epoch: {args.epochs}  |  Toplam adım: {total_steps:,}  |  Warmup: {warmup_steps}")

    # --- Çıktı klasörü ---
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Log ---
    log_path = out_dir / "train_log.jsonl"
    log_file = open(log_path, "w", encoding="utf-8")

    global_step = 0
    best_loss   = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        epoch_loss = 0.0
        t0 = time.time()

        for batch_idx, (x, y) in enumerate(loader, 1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Forward
            with autocast(dtype=torch.bfloat16):
                out  = model(x, labels=y)
                loss = out.loss / args.grad_accum

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * args.grad_accum

            # Gradient accumulation
            if batch_idx % args.grad_accum == 0:
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # LR güncelle
                lr = get_lr(global_step, warmup_steps, total_steps, args.lr, args.lr / 10)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1

                # Log
                if global_step % args.log_every == 0:
                    ppl = math.exp(min(epoch_loss / batch_idx, 20))
                    elapsed = time.time() - t0
                    tok_per_sec = (batch_idx * args.batch * cfg.n_ctx) / elapsed
                    msg = {
                        "step": global_step, "epoch": epoch,
                        "loss": round(epoch_loss / batch_idx, 4),
                        "ppl": round(ppl, 2), "lr": round(lr, 8),
                        "tok_s": int(tok_per_sec),
                    }
                    print(f"  step {global_step:5d} | loss {msg['loss']:.4f} | "
                          f"ppl {msg['ppl']:.1f} | lr {lr:.2e} | {tok_per_sec:.0f} tok/s")
                    log_file.write(json.dumps(msg, ensure_ascii=False) + "\n")
                    log_file.flush()

        # Epoch özeti
        avg_loss = epoch_loss / len(loader)
        print(f"\n[Epoch {epoch}/{args.epochs}] loss={avg_loss:.4f}  "
              f"ppl={math.exp(min(avg_loss, 20)):.1f}  "
              f"süre={time.time() - t0:.0f}s\n")

        # Her epoch sonunda kaydet
        epoch_dir = out_dir / f"epoch_{epoch}"
        save_model(model, cfg, epoch_dir)

        # En iyi modeli ayrıca sakla
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_model(model, cfg, out_dir / "best")
            print(f"  ✓ En iyi model güncellendi (loss={best_loss:.4f})")

    log_file.close()
    print(f"\n[Train] Tamamlandı. Çıktılar → {out_dir}")


def save_model(model: TINAModel, cfg: TINAConfig, directory: Path):
    """HuggingFace formatında (safetensors dahil) kaydet."""
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(directory), safe_serialization=True)  # .safetensors
    cfg.save_pretrained(str(directory))
    print(f"  → Kaydedildi: {directory}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TurkishGPT Eğitim")
    parser.add_argument("--preset",     default="small",          choices=PRESETS.keys(),
                        help="Model boyutu: micro(~1M) | small(~3M) | medium(~5M)")
    parser.add_argument("--data",       default="data/clean.txt", help="Eğitim verisi")
    parser.add_argument("--tokenizer",  default="tokenizer/",     help="Tokenizer dizini")
    parser.add_argument("--out",        default="checkpoints/",   help="Çıktı dizini")
    parser.add_argument("--epochs",     type=int,   default=5)
    parser.add_argument("--batch",      type=int,   default=16,   help="Batch boyutu")
    parser.add_argument("--grad-accum", type=int,   default=4,    help="Gradient accumulation adımları")
    parser.add_argument("--lr",         type=float, default=3e-4, help="Maksimum öğrenme hızı")
    parser.add_argument("--wd",         type=float, default=0.1,  help="Weight decay")
    parser.add_argument("--log-every",  type=int,   default=50,   help="Kaç adımda bir log")
    args = parser.parse_args()
    train(args)
