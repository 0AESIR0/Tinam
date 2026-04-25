import re
import argparse
import html
from pathlib import Path
from langdetect import detect

# --- Sabitler ---
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
DATA_DIR       = PROJECT_ROOT / "data"
DEFAULT_INPUT  = DATA_DIR / "raw_big.txt"
DEFAULT_OUTPUT = DATA_DIR / "clean.txt"

MOJIBAKE_MARKERS = ["Ã", "Â", "â€", "Ð", "Ñ"]

TURKISH_SPECIALS = ["ş", "ğ", "ı", "ç", "ö", "ü", "Ş", "Ğ", "İ", "Ç", "Ö", "Ü"]

COMMON_TR_WORDS = ["ve", "bir", "de", "da", "mi", "mı", "sen", "ben", "için", "çok",
                   "bu", "ile", "var", "ne", "ki", "ama", "ya", "daha", "o", "biz"]


def resolve_cli_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def clean(text: str) -> str:
    text = html.unescape(text)                          # &amp; gibi HTML entity'leri çöz
    text = re.sub(r'<.*?>', '', text)                   # HTML etiketlerini kaldır
    text = re.sub(r'http\S+', '', text)                 # URL'leri kaldır
    text = re.sub(r'\s+', ' ', text)                    # Fazla boşlukları temizle
    text = text.replace('\u201c', '"').replace('\u201d', '"').replace('\u2019', "'")
    text = re.sub(r'-{2,}', '-', text)                  # Çoklu tire
    text = re.sub(r'(\w)\1{4,}', r'\1\1\1', text)      # aaaaa → aaa
    # Mojibake kontrolü: bozuk encoding işaretleri içeriyorsa boşalt
    if any(marker in text for marker in MOJIBAKE_MARKERS):
        return ""
    return text.strip()


def is_tr(text: str) -> bool:
    text = text.strip()
    if len(text) < 20:
        cnt = sum(1 for w in COMMON_TR_WORDS if w in text.lower().split())
        return cnt >= 1
    # Türkçe özel karakter varlığı güçlü bir sinyal
    if any(ch in text for ch in TURKISH_SPECIALS):
        return True
    try:
        return detect(text) == "tr"
    except Exception:
        cnt = sum(1 for w in COMMON_TR_WORDS if w in text.lower().split())
        return cnt >= 2


def main():
    parser = argparse.ArgumentParser(description="Temizleme scripti: dil tespiti ve normalizasyon")
    parser.add_argument("--input",   default=str(DEFAULT_INPUT),  help="Girdi dosyası")
    parser.add_argument("--output",  default=str(DEFAULT_OUTPUT), help="Çıktı dosyası")
    parser.add_argument("--min-len", type=int, default=50,        help="Minimum karakter uzunluğu")
    args = parser.parse_args()

    cleaned: list[str] = []

    input_path = resolve_cli_path(args.input)

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = clean(line)
            if len(line) >= args.min_len and is_tr(line):
                cleaned.append(line)

    out = resolve_cli_path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for line in cleaned:
            f.write(line + "\n")

    print(f"Temizlendi: {len(cleaned)} satır → {out}")


if __name__ == "__main__":
    main()