# AskSphere

Ask questions about any passage of text. AskSphere answers with a Transformer built from scratch in PyTorch, in two ways:

- **Extractive QA** picks the answer span out of your passage, and says "no answer" when the passage doesn't contain one (SQuAD v2 style).
- **Generative QA** writes a free-form answer with a custom hybrid decoder, and can also decline to answer.

Both modes share one custom Transformer encoder, pre-trained with masked language modeling (MLM).

A third option, **Pretrained QA**, uses a public Hugging Face model (`deepset/roberta-base-squad2` by default). It is *not* the custom Transformer. It is included so the app gives accurate answers out of the box, and as a baseline to compare the custom models against.

```
Phase 1  MLM pre-training of the custom encoder
            ├── Phase 2  Extractive QA fine-tuning (SQuAD v2)
            └── Phase 3  Generative QA (hybrid decoder, multi-stage training)
                              └── FastAPI server + web UI
```

> **Based on** [Question-Answer-System-with-a-Custom-Transformer-Encoder-and-Generative-Decoder](https://github.com/Harsha081459/Question-Answer-System-with-a-Custom-Transformer-Encoder-and-Generative-Decoder)
> by Harsha081459 (MIT). AskSphere restructures it into an installable package, fixes the import and
> serving problems, and adds tests. Full credit for the original models and research goes to the original author.

## Quick start

```bash
git clone https://github.com/myriad-8/AskSphere.git
cd AskSphere
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn asksphere.api:app --port 7860
```

Open <http://localhost:7860>. The page tells you which models are loaded. API docs are at `/docs`.

## Models

The weights are too large for git, so they are not in this repo. The server looks for:

```
checkpoints/
├── extractive/    model.safetensors, model_config.json, tokenizer files
└── generative/    best.pt, tokenizer files
```

Point it somewhere else with `ASKSPHERE_EXTRACTIVE_DIR`, `ASKSPHERE_GENERATIVE_DIR`, `ASKSPHERE_GENERATIVE_CKPT`
(default `best.pt`) and `ASKSPHERE_DECODER_VARIANT` (`auto`, `hybrid` or `standard`).
If a model is missing, the server still starts and that mode is shown as "not loaded".

**Check your install without any weights.** This builds tiny random-weight checkpoints (the answers are meaningless):

```bash
python -m asksphere.smoke --out checkpoints_smoke
ASKSPHERE_EXTRACTIVE_DIR=checkpoints_smoke/extractive \
ASKSPHERE_GENERATIVE_DIR=checkpoints_smoke/generative \
uvicorn asksphere.api:app --port 7860
```

**Pretrained QA (works out of the box).** On first start the server downloads `deepset/roberta-base-squad2` (about 500 MB, internet needed once). Change it with `ASKSPHERE_PRETRAINED_MODEL`, or set it to an empty string to disable.

**Get real custom models** by training them (below) and copying the output folders into `checkpoints/`.

## Training pipeline

Install the extras first: `pip install -r requirements-train.txt`. Run everything from the repo root.

```bash
# 1. Pre-train the encoder (MLM on Wikipedia + C4)
python -m asksphere.pretraining.mlm_pretraining --out_dir checkpoints_pretrain

# 2. Fine-tune for extractive QA on SQuAD v2
python -m asksphere.extractive.extractive_finetuning \
  --checkpoint_dir checkpoints_pretrain/<step_dir> --dataset squad_v2 --output_dir checkpoints_qa

# 3. Train the generative model
python -m asksphere.generative.generative_finetuning \
  --decoder_variant hybrid \
  --tokenizer_path checkpoints_pretrain/<step_dir> \
  --pretrain_ckpt checkpoints_pretrain/<step_dir>/checkpoint.pt \
  --output_dir checkpoints_generative_qa
```

Copy the results into `checkpoints/extractive` and `checkpoints/generative`.
Detailed recipes: [`docs/GENERATIVE_DECODER.md`](docs/GENERATIVE_DECODER.md), [`docs/MLM_PRETRAINING.md`](docs/MLM_PRETRAINING.md).
Other tools: `extractive_inference`, `extractive_threshold_tuning`, `extractive_error_analysis`,
`compare_extractive_models`, `generative_evaluation` (EM / F1 / ROUGE-L / BLEU), `compare_generative_models`.

## API

`POST /predict` (also `/api/predict`)

```json
{
  "model_type": "extractive",
  "question": "When did Einstein receive the Nobel Prize?",
  "context": "Albert Einstein ... received the Nobel Prize in Physics in 1921 ..."
}
```

`GET /api/health` returns which models are loaded, the device, and the reason for any model that failed to load.

## Project layout

```
asksphere/
  api.py            FastAPI server (model loading, inference, health)
  smoke.py          tiny random-weight checkpoints for testing
  pretraining/      custom encoder + MLM pre-training
  extractive/       span-extraction model, inference, tuning, analysis
  generative/       decoders, data, training, evaluation, inference
static/             web UI (HTML / CSS / vanilla JS)
data/passages.tsv   sample passages
tests/              end-to-end tests (no downloads needed)
reports/            project report and figure scripts
docs/               detailed training docs
```

## Tests and Docker

```bash
pip install -r requirements-dev.txt && python -m pytest -q
docker build -t asksphere . && docker run -p 7860:7860 -v "$PWD/checkpoints:/app/checkpoints" asksphere
```
## Acknowledgements

Built on the open-source project [Question-Answer-System-with-a-Custom-Transformer-Encoder-and-Generative-Decoder](https://github.com/Harsha081459/Question-Answer-System-with-a-Custom-Transformer-Encoder-and-Generative-Decoder) by Harsha081459.
Restructured, fixed and tested by Sahithi (myriad-8).
