"""AskSphere FastAPI server.

Start it from the repository root:

    uvicorn asksphere.api:app --host 0.0.0.0 --port 7860

Models are looked up in ``checkpoints/extractive`` and ``checkpoints/generative`` by default.
Override with the ASKSPHERE_EXTRACTIVE_DIR / ASKSPHERE_GENERATIVE_DIR environment variables.
A missing model never stops the server: that mode is reported as unavailable in /api/health.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from safetensors.torch import load_file
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from asksphere import __version__
from asksphere.extractive.extractive_finetuning import BertForQuestionAnswering
from asksphere.generative.generative_inference import build_target_ids, decode_generated_ids
from asksphere.generative.main_hybrid_decoder import (
    DecoderConfig as HybridDecoderConfig,
    GenerativeQAModelHybrid,
)
from asksphere.generative.standard_generative_decoder import (
    DecoderConfig as StandardDecoderConfig,
    GenerativeQAModel as StandardGenerativeQAModel,
)
from asksphere.pretraining.mlm_pretraining import ModelConfig

log = logging.getLogger("asksphere")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s: %(message)s")

ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT / "static"
NO_ANSWER_TEXT = "The context does not contain the answer."

torch.set_num_threads(int(os.environ.get("ASKSPHERE_THREADS", min(4, os.cpu_count() or 1))))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser()


class Engine:
    """Holds the loaded models and why a model could not be loaded."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.ext_model: Optional[BertForQuestionAnswering] = None
        self.ext_tok = None
        self.ext_cfg: Optional[ModelConfig] = None
        self.ext_error: Optional[str] = None
        self.gen_model = None
        self.gen_tok = None
        self.gen_enc_cfg: Optional[ModelConfig] = None
        self.gen_variant: Optional[str] = None
        self.gen_error: Optional[str] = None
        self.pre_model = None
        self.pre_tok = None
        self.pre_name: Optional[str] = None
        self.pre_error: Optional[str] = None

    # ----- loading -------------------------------------------------------------------------
    @staticmethod
    def _load_tokenizer(path: Path):
        tok = AutoTokenizer.from_pretrained(str(path), use_fast=True)
        if tok.pad_token is None:
            tok.pad_token = tok.sep_token
        return tok

    def load_extractive(self, ext_dir: Path) -> None:
        self.ext_model = self.ext_tok = self.ext_cfg = None
        cfg_path, weights = ext_dir / "model_config.json", ext_dir / "model.safetensors"
        if not (cfg_path.is_file() and weights.is_file()):
            self.ext_error = f"Expected model_config.json and model.safetensors in '{ext_dir}'."
            return
        try:
            cfg = ModelConfig(**json.loads(cfg_path.read_text(encoding="utf-8")))
            model = BertForQuestionAnswering(cfg)
            model.load_state_dict(load_file(str(weights)), strict=True)
            self.ext_tok = self._load_tokenizer(ext_dir)
            self.ext_model = model.to(DEVICE).eval()
            self.ext_cfg, self.ext_error = cfg, None
        except Exception as exc:  # report, never crash the server
            self.ext_error = f"Could not load extractive model: {exc}"

    def load_generative(self, gen_dir: Path, ckpt_name: str, variant: str) -> None:
        self.gen_model = self.gen_tok = self.gen_enc_cfg = self.gen_variant = None
        ckpt = gen_dir / ckpt_name
        if not ckpt.is_file():
            self.gen_error = f"Expected '{ckpt_name}' (plus tokenizer files) in '{gen_dir}'."
            return
        try:
            payload = torch.load(str(ckpt), map_location="cpu", weights_only=False)
            enc_cfg = ModelConfig(**payload["encoder_config"])
            variants = {
                "hybrid": (GenerativeQAModelHybrid, HybridDecoderConfig),
                "standard": (StandardGenerativeQAModel, StandardDecoderConfig),
            }
            order = ["hybrid", "standard"] if variant == "auto" else [variant]
            last_exc: Optional[Exception] = None
            for name in order:
                model_cls, dec_cls = variants[name]
                try:
                    model = model_cls(enc_cfg, dec_cls(**payload["decoder_config"]))
                    model.load_state_dict(payload["model"], strict=True)
                except Exception as exc:
                    last_exc = exc
                    continue
                self.gen_model = model.to(DEVICE).eval()
                self.gen_variant = name
                break
            if self.gen_model is None:
                raise RuntimeError(f"checkpoint matches neither decoder variant ({last_exc})")
            self.gen_tok = self._load_tokenizer(gen_dir)
            self.gen_enc_cfg, self.gen_error = enc_cfg, None
        except Exception as exc:
            self.gen_model = None
            self.gen_error = f"Could not load generative model: {exc}"

    def load_pretrained(self, model_name: str) -> None:
        """Optional accurate fallback: a public pretrained QA model (not the custom Transformer)."""
        self.pre_model = self.pre_tok = self.pre_name = None
        if not model_name:
            self.pre_error = "Disabled (ASKSPHERE_PRETRAINED_MODEL is empty)."
            return
        try:
            tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(DEVICE).eval()
            self.pre_tok, self.pre_model, self.pre_name, self.pre_error = tok, model, model_name, None
        except Exception as exc:
            self.pre_error = (
                f"Could not load '{model_name}': {exc}. The first run needs an internet connection "
                "to download it (about 500 MB)."
            )

    def status(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "device": DEVICE,
            "models": {
                "extractive": {"loaded": self.ext_model is not None, "error": self.ext_error},
                "generative": {
                    "loaded": self.gen_model is not None,
                    "variant": self.gen_variant,
                    "error": self.gen_error,
                },
                "pretrained": {"loaded": self.pre_model is not None, "name": self.pre_name, "error": self.pre_error},
            },
        }


engine = Engine()


@asynccontextmanager
async def lifespan(_: FastAPI):
    ext_dir = _env_path("ASKSPHERE_EXTRACTIVE_DIR", ROOT / "checkpoints" / "extractive")
    gen_dir = _env_path("ASKSPHERE_GENERATIVE_DIR", ROOT / "checkpoints" / "generative")
    log.info("Loading extractive model from %s", ext_dir)
    engine.load_extractive(ext_dir)
    log.info("Loading generative model from %s", gen_dir)
    engine.load_generative(
        gen_dir,
        os.environ.get("ASKSPHERE_GENERATIVE_CKPT", "best.pt"),
        os.environ.get("ASKSPHERE_DECODER_VARIANT", "auto"),
    )
    default_pre = "deepset/roberta-base-squad2"
    pre_name = os.environ.get("ASKSPHERE_PRETRAINED_MODEL", default_pre)
    log.info("Loading pretrained model %s", pre_name or "(disabled)")
    engine.load_pretrained(pre_name)
    for name, err in (("extractive", engine.ext_error), ("generative", engine.gen_error), ("pretrained", engine.pre_error)):
        log.warning("%s model unavailable: %s", name, err) if err else log.info("%s model ready", name)
    yield


app = FastAPI(title="AskSphere", version=__version__, lifespan=lifespan)


# ----- request / response ------------------------------------------------------------------
class PredictRequest(BaseModel):
    model_type: str = Field(..., pattern="^(extractive|generative|pretrained)$")
    question: str = Field(..., min_length=1, max_length=2000)
    context: str = Field(..., min_length=1, max_length=50000)
    max_length: int = Field(256, ge=32, le=512)
    # extractive
    doc_stride: int = Field(64, ge=8, le=256)
    n_best: int = Field(20, ge=1, le=50)
    max_answer_length: int = Field(30, ge=1, le=200)
    # generative
    beam_size: int = Field(4, ge=1, le=10)
    max_new_tokens: int = Field(32, ge=2, le=128)
    length_penalty: float = Field(1.0, ge=0.0, le=5.0)
    enable_no_answer_gate: bool = True
    # shared: each model type interprets it on its own score scale
    no_answer_threshold: float = 0.0


@app.get("/api/health")
def health():
    return engine.status()


@app.post("/predict")
@app.post("/api/predict")
def predict(req: PredictRequest):
    if not req.question.strip() or not req.context.strip():
        raise HTTPException(status_code=422, detail="Question and context must not be blank.")
    if req.model_type == "extractive":
        if engine.ext_model is None:
            raise HTTPException(503, f"Extractive model is not loaded. {engine.ext_error or ''}".strip())
        with engine.lock:
            return run_extractive(req, engine.ext_tok, engine.ext_model, engine.ext_cfg.max_position_embeddings)
    if req.model_type == "pretrained":
        if engine.pre_model is None:
            raise HTTPException(503, f"Pretrained model is not loaded. {engine.pre_error or ''}".strip())
        with engine.lock:
            return run_extractive(req, engine.pre_tok, engine.pre_model, 512)
    if engine.gen_model is None:
        raise HTTPException(503, f"Generative model is not loaded. {engine.gen_error or ''}".strip())
    with engine.lock:
        return run_generative(req)


# ----- inference ---------------------------------------------------------------------------
def run_extractive(req: PredictRequest, tok, model, max_positions: int) -> dict[str, Any]:
    """Span extraction. Works for the custom model and for any Hugging Face QA model."""
    max_length = min(req.max_length, max_positions)
    doc_stride = min(req.doc_stride, max(8, max_length // 4))

    enc = tok(
        [req.question],
        [req.context],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=True,  # long passages split into several windows; they must share a length
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(DEVICE)
    model_inputs = {k: enc[k].to(DEVICE) for k in ("input_ids", "attention_mask", "token_type_ids") if k in enc}
    if isinstance(model, BertForQuestionAnswering) and "token_type_ids" not in model_inputs:
        model_inputs["token_type_ids"] = torch.zeros_like(input_ids)

    with torch.no_grad():
        out = model(**model_inputs)
    start_logits = out["start_logits"].cpu().numpy()
    end_logits = out["end_logits"].cpu().numpy()

    best_score: Optional[float] = None
    best_text = ""
    best_null = float("inf")

    for i in range(input_ids.shape[0]):
        offsets = enc["offset_mapping"][i].tolist()
        seq_ids = enc.sequence_ids(i)
        cls_idx = input_ids[i].tolist().index(tok.cls_token_id)
        best_null = min(best_null, float(start_logits[i][cls_idx] + end_logits[i][cls_idx]))

        n_best = min(req.n_best, start_logits.shape[1])
        starts = start_logits[i].argsort()[-1 : -n_best - 1 : -1].tolist()
        ends = end_logits[i].argsort()[-1 : -n_best - 1 : -1].tolist()
        for s in starts:
            for e in ends:
                if seq_ids[s] != 1 or seq_ids[e] != 1:
                    continue
                if e < s or (e - s + 1) > req.max_answer_length:
                    continue
                score = float(start_logits[i][s] + end_logits[i][e])
                if best_score is None or score > best_score:
                    best_score = score
                    best_text = req.context[offsets[s][0] : offsets[e][1]]

    if best_score is None:  # no valid span anywhere: the model can only say "no answer"
        return {
            "question": req.question,
            "answer": "",
            "span_score": None,
            "null_score": best_null,
            "score_diff_null_minus_span": None,
            "predicted_no_answer": True,
        }

    diff = best_null - best_score
    no_answer = diff > req.no_answer_threshold
    return {
        "question": req.question,
        "answer": "" if no_answer else best_text,
        "span_score": best_score,
        "null_score": best_null,
        "score_diff_null_minus_span": diff,
        "predicted_no_answer": no_answer,
    }


def run_generative(req: PredictRequest) -> dict[str, Any]:
    tok, model, enc_cfg = engine.gen_tok, engine.gen_model, engine.gen_enc_cfg
    enc = tok(
        [f"question: {req.question} context: {req.context}"],
        truncation=True,
        max_length=min(req.max_length, enc_cfg.max_position_embeddings),
        return_tensors="pt",
    )
    enc_ids = enc["input_ids"].to(DEVICE)
    enc_mask = enc["attention_mask"].to(DEVICE)
    enc_ttype = enc.get("token_type_ids", torch.zeros_like(enc_ids)).to(DEVICE)

    bos = tok.cls_token_id if tok.cls_token_id is not None else tok.pad_token_id
    eos = tok.sep_token_id if tok.sep_token_id is not None else tok.pad_token_id
    pad = tok.pad_token_id
    gen_kwargs = dict(
        encoder_input_ids=enc_ids,
        encoder_token_type_ids=enc_ttype,
        encoder_attention_mask=enc_mask,
        bos_token_id=bos,
        eos_token_id=eos,
        pad_token_id=pad,
        max_new_tokens=req.max_new_tokens,
        beam_size=req.beam_size,
        length_penalty=req.length_penalty,
    )

    with torch.no_grad():
        if req.enable_no_answer_gate:
            out, pred_logprob, _ = model.generate(**gen_kwargs, return_logprob=True)
        else:
            out, pred_logprob = model.generate(**gen_kwargs), None
    raw_answer = decode_generated_ids(tok, out[0].tolist(), bos=bos, eos=eos, pad=pad)
    result: dict[str, Any] = {"question": req.question, "answer": raw_answer, "raw_answer": raw_answer}

    if req.enable_no_answer_gate:
        no_ans_ids = build_target_ids(
            tokenizer=tok, text=NO_ANSWER_TEXT, bos=bos, eos=eos,
            max_new_tokens=req.max_new_tokens, device=DEVICE,
        )
        with torch.no_grad():
            no_ans_logprob = model.sequence_logprob(
                encoder_input_ids=enc_ids,
                encoder_token_type_ids=enc_ttype,
                encoder_attention_mask=enc_mask,
                target_ids=no_ans_ids,
                pad_token_id=pad,
                normalize_by_length=True,
            )[0].item()
        score_diff = no_ans_logprob - pred_logprob
        gated = score_diff > req.no_answer_threshold
        result["answer"] = NO_ANSWER_TEXT if gated else raw_answer
        result["gate"] = {
            "score_diff": score_diff,
            "pred_avg_logprob": pred_logprob,
            "no_answer_avg_logprob": no_ans_logprob,
            "selected_no_answer": gated,
        }
    return result


# The static UI must be mounted last so it does not shadow the API routes.
if STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
