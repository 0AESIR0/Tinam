"""
chat.py
Eğitilmiş Türkçe GPT modeliyle terminal üzerinden sohbet.

Kullanım:
    python chat.py --model checkpoints/best/
    python chat.py --model checkpoints/best/ --temp 0.7 --top-p 0.85
"""

import argparse
import sys
import torch
from transformers import PreTrainedTokenizerFast

from model import TINAModel, TINAConfig


# ---------------------------------------------------------------------------
# Prompt şablonu
# ---------------------------------------------------------------------------
# Model bu format üzerinde fine-tune edilmezse genel metin üretir.
# Fine-tune edilirse <user>/<bot> tokenleri ile dialog öğrenir.

SYSTEM_PROMPT = "Sen yardımcı bir Türkçe yapay zeka asistanısın."

def build_prompt(history: list[tuple[str, str]], user_msg: str, tokenizer) -> str:
    """Konuşma geçmişini tek string'e çevirir."""
    parts = [f"<bos>{SYSTEM_PROMPT}<sep>"]
    for u, b in history:
        parts.append(f"<user>{u}<sep><bot>{b}<sep>")
    parts.append(f"<user>{user_msg}<sep><bot>")
    return "".join(parts)


def trim_history(history: list, max_turns: int = 8) -> list:
    """Bağlam penceresinin taşmasını önlemek için eski turları sil."""
    return history[-max_turns:]


# ---------------------------------------------------------------------------
# Ana döngü
# ---------------------------------------------------------------------------

def chat(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[Chat] Model yükleniyor: {args.model}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model)
    model     = TINAModel.from_pretrained(args.model).to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Chat] Parametre: {total_params:,}  |  Cihaz: {device}")
    print("─" * 50)
    print("Türkçe GPT — sohbet başlıyor. Çıkmak için 'quit' veya Ctrl+C.")
    print("─" * 50 + "\n")

    history: list[tuple[str, str]] = []
    eos_id  = tokenizer.eos_token_id or 3
    sep_id  = tokenizer.convert_tokens_to_ids("<sep>")

    while True:
        try:
            user_input = input("Sen: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Chat] Görüşürüz!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "çık"):
            print("[Chat] Görüşürüz!")
            break
        if user_input.lower() == "/temizle":
            history = []
            print("[Geçmiş temizlendi]\n")
            continue

        # Prompt oluştur
        history = trim_history(history, args.max_turns)
        prompt  = build_prompt(history, user_input, tokenizer)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # Bağlam penceresi kontrolü
        max_ctx = model.cfg.n_ctx
        if input_ids.shape[1] >= max_ctx - 50:
            history = history[len(history) // 2:]   # yarıya düşür
            prompt  = build_prompt(history, user_input, tokenizer)
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # Üretim
        with torch.inference_mode():
            out_ids = model.generate_text(
                input_ids,
                max_new_tokens=args.max_tokens,
                temperature=args.temp,
                top_k=args.top_k,
                top_p=args.top_p,
                eos_token_id=eos_id,
            )

        # Sadece yeni üretilen kısım
        new_ids  = out_ids[0, input_ids.shape[1]:]
        response = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        # Eğer <sep> varsa yalnızca bot yanıtını al
        if "<sep>" in response:
            response = response.split("<sep>")[0].strip()

        print(f"\nBot: {response}\n")
        history.append((user_input, response))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TurkishGPT Sohbet")
    parser.add_argument("--model",      default="checkpoints/best/", help="Model dizini")
    parser.add_argument("--temp",       type=float, default=0.8,  help="Sıcaklık (0.1–1.5)")
    parser.add_argument("--top-k",      type=int,   default=50,   help="Top-k örnekleme")
    parser.add_argument("--top-p",      type=float, default=0.90, help="Nucleus örnekleme")
    parser.add_argument("--max-tokens", type=int,   default=200,  help="Maksimum yeni token")
    parser.add_argument("--max-turns",  type=int,   default=8,    help="Hafızada tutulan konuşma sayısı")
    args = parser.parse_args()
    chat(args)
