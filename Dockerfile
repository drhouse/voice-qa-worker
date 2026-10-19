# CUDA 12 + cuDNN 9 (CTranslate2 4.5+); torch >= 2.6 (transformers refuses torch.load below it).
FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

ENV HF_HOME=/models \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake both models into the image so a cold worker never downloads them.
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('large-v3-turbo', device='cpu', compute_type='int8')" && \
    python -c "from transformers import AutoModelForCTC, Wav2Vec2FeatureExtractor; from huggingface_hub import hf_hub_download; n='facebook/wav2vec2-lv-60-espeak-cv-ft'; Wav2Vec2FeatureExtractor.from_pretrained(n); AutoModelForCTC.from_pretrained(n); hf_hub_download(n, 'vocab.json')"

# Everything is local from here on; fail loudly rather than re-download.
ENV HF_HUB_OFFLINE=1

COPY handler.py .
CMD ["python", "-u", "handler.py"]
