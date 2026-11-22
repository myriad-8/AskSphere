FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# CPU-only PyTorch keeps the image small; swap the index URL if you need CUDA.
COPY requirements.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

RUN useradd -m -u 1000 user
COPY --chown=user . /app
USER user

# Mount or copy your model files into /app/checkpoints/{extractive,generative}
EXPOSE 7860
CMD ["uvicorn", "asksphere.api:app", "--host", "0.0.0.0", "--port", "7860"]
