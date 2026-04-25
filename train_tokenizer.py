"""
train_tokenizer.py
BPE tokenizer'ı ham veri üzerinde eğitir ve HuggingFace formatında kaydeder.
Kullanım: python train_tokenizer.py --input data/clean.txt --output tokenizer/
"""

import argparse
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from transformers import PreTrainedTokenizerFast


def train_tokenizer(input_path: str, output_dir: str, vocab_size: int = 8000):
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Tokenizer] Eğitim verisi: {input_path}")
    print(f"[Tokenizer] Vocab boyutu: {vocab_size}")

    # BPE modeli
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    special_tokens = ["<unk>", "<pad>", "<bos>", "<eos>", "<user>", "<bot>", "<sep>"]

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=2,
        show_progress=True,
    )

    tokenizer.train(files=[str(input_path)], trainer=trainer)

    # Post-processor: her sequence başına <bos>, sonuna <eos>
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<bos>:0 $A:0 <eos>:0",
        pair="<bos>:0 $A:0 <sep>:0 $B:0 <eos>:0",
        special_tokens=[("<bos>", bos_id), ("<eos>", eos_id), ("<sep>", tokenizer.token_to_id("<sep>"))],
    )

    # HuggingFace wrapper
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        additional_special_tokens=["<user>", "<bot>", "<sep>"],
    )
    hf_tokenizer.save_pretrained(str(output_dir))
    print(f"[Tokenizer] Kaydedildi → {output_dir}")
    return hf_tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      default="data/clean.txt")
    parser.add_argument("--output",     default="tokenizer/")
    parser.add_argument("--vocab-size", type=int, default=8000)
    args = parser.parse_args()
    train_tokenizer(args.input, args.output, args.vocab_size)
