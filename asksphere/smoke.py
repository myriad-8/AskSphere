"""Build tiny, randomly initialised checkpoints in the exact formats the server expects.

These exist to verify the whole stack (loading, tokenisation, inference, API, UI) without any
training or downloads. The weights are random, so the answers are meaningless. For real answers,
train the models or drop real checkpoints into ``checkpoints/`` (see the README).

    python -m asksphere.smoke --out checkpoints_smoke
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors.torch import save_file
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizerFast

from asksphere.extractive.extractive_finetuning import BertForQuestionAnswering
from asksphere.generative.main_hybrid_decoder import DecoderConfig, GenerativeQAModelHybrid
from asksphere.pretraining.mlm_pretraining import ModelConfig

ROOT = Path(__file__).resolve().parents[1]
SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def _train_tokenizer(out_dir: Path, corpus_file: Path) -> BertTokenizerFast:
    out_dir.mkdir(parents=True, exist_ok=True)
    wp = BertWordPieceTokenizer(lowercase=True)
    wp.train([str(corpus_file)], vocab_size=600, min_frequency=1, special_tokens=SPECIALS)
    wp.save_model(str(out_dir))
    tok = BertTokenizerFast(vocab_file=str(out_dir / "vocab.txt"), do_lower_case=True)
    tok.save_pretrained(str(out_dir))
    return tok


def build_smoke_checkpoints(out_dir: str | Path, seed: int = 0) -> tuple[Path, Path]:
    """Create ``<out>/extractive`` and ``<out>/generative``; return both paths."""
    out = Path(out_dir)
    ext_dir, gen_dir = out / "extractive", out / "generative"
    torch.manual_seed(seed)

    # Tokenizer corpus: the bundled passages (column 2 of the TSV).
    with tempfile.TemporaryDirectory() as tmp:
        corpus = Path(tmp) / "corpus.txt"
        lines = (ROOT / "data" / "passages.tsv").read_text(encoding="utf-8").splitlines()[1:]
        corpus.write_text("\n".join(l.split("\t")[1] for l in lines if "\t" in l), encoding="utf-8")
        tok = _train_tokenizer(ext_dir, corpus)
        _train_tokenizer(gen_dir, corpus)

    enc_cfg = ModelConfig(
        vocab_size=len(tok), max_position_embeddings=256, hidden_size=64,
        num_hidden_layers=2, num_attention_heads=4, intermediate_size=128,
    )
    (ext_dir / "model_config.json").write_text(json.dumps(asdict(enc_cfg)), encoding="utf-8")
    qa = BertForQuestionAnswering(enc_cfg)
    save_file({k: v.contiguous() for k, v in qa.state_dict().items()}, str(ext_dir / "model.safetensors"))

    dec_cfg = DecoderConfig(
        vocab_size=len(tok), hidden_size=64, num_layers=2, num_attention_heads=4,
        intermediate_size=128, max_position_embeddings=64,
    )
    gen = GenerativeQAModelHybrid(enc_cfg, dec_cfg)
    torch.save(
        {"model": gen.state_dict(), "encoder_config": enc_cfg.__dict__, "decoder_config": dec_cfg.to_dict(), "step": 0},
        gen_dir / "best.pt",
    )
    return ext_dir, gen_dir


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="checkpoints_smoke")
    args = p.parse_args()
    ext, gen = build_smoke_checkpoints(args.out)
    print(f"Wrote RANDOM-WEIGHT smoke checkpoints:\n  {ext}\n  {gen}")
    print(f"Run:  ASKSPHERE_EXTRACTIVE_DIR={ext} ASKSPHERE_GENERATIVE_DIR={gen} uvicorn asksphere.api:app")


if __name__ == "__main__":
    main()
