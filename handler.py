"""Runpod Serverless worker: voice clips in, raw speech observations out.

Game-agnostic by design. The worker holds no script, dictionary or thresholds -
it answers one question per clip, "what does the recogniser hear", and every
judgement (expected text, pronunciation table, pass/fail limits) stays on the
caller's machine where it can be re-tuned without paying for the GPU again.

Both models are baked into the image (see Dockerfile), so a cold worker loads
from local disk instead of pip-installing and downloading ~3 GB first.

Input:  {"input": {"clips": [{"id": str, "audio_b64": str}, ...],
                   "phonemes": true}}
Output: {"results": [...], "seconds": float, "device": "cuda" | "CPU-FALLBACK"}
"""
import base64
import json
import os
import tempfile
import time

import librosa
import runpod
import torch
from faster_whisper import WhisperModel
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCTC, Wav2Vec2FeatureExtractor

PHONEME_MODEL = "facebook/wav2vec2-lv-60-espeak-cv-ft"
DEVICE = "cuda" if torch.cuda.is_available() else "CPU-FALLBACK"

# Loaded once per worker at import; the handler reuses them for every job.
WHISPER = WhisperModel(os.environ.get("WHISPER_MODEL", "large-v3-turbo"),
                       device="cuda" if DEVICE == "cuda" else "cpu",
                       compute_type="float16" if DEVICE == "cuda" else "int8")
EXTRACTOR = Wav2Vec2FeatureExtractor.from_pretrained(PHONEME_MODEL)
CTC = AutoModelForCTC.from_pretrained(PHONEME_MODEL).eval()
if DEVICE == "cuda":
    CTC = CTC.to("cuda").half()
with open(hf_hub_download(PHONEME_MODEL, "vocab.json"), encoding="utf-8") as fh:
    LOOKUP = {i: s for s, i in json.load(fh).items()}


def transcribe(path):
    # Greedy, no conditioning on previous text: clips are independent, and a
    # hallucination must not seed the next one.
    segments, info = WHISPER.transcribe(
        path, language=None, beam_size=1, temperature=0,
        condition_on_previous_text=False, word_timestamps=True, vad_filter=False)
    words, said, logprobs = [], [], []
    for seg in segments:
        said.append(seg.text)
        logprobs.append(seg.avg_logprob)
        for w in seg.words or []:
            words.append({"word": w.word.strip(), "prob": float(round(w.probability, 4))})
    return {
        "language": str(info.language),
        "language_prob": float(round(info.language_probability, 4)),
        "text": "".join(said).strip(),
        "avg_logprob": float(round(sum(logprobs) / len(logprobs), 4)) if logprobs else None,
        "words": words,
        "duration": float(round(info.duration, 3)),
    }


def phonemise(path):
    audio, _ = librosa.load(path, sr=16000, mono=True)
    values = EXTRACTOR(audio, sampling_rate=16000, return_tensors="pt").input_values
    if DEVICE == "cuda":
        values = values.to("cuda").half()
    with torch.no_grad():
        frames = torch.argmax(CTC(values).logits, dim=-1)[0].tolist()
    # Greedy CTC: collapse repeats, drop blank and special tokens.
    symbols, previous = [], None
    for index in frames:
        if index != previous:
            s = LOOKUP.get(index, "")
            if s and not (s.startswith("<") and s.endswith(">")):
                symbols.append(s)
        previous = index
    return symbols


def handler(job):
    inp = job["input"]
    want_phonemes = inp.get("phonemes", True)
    started, out = time.time(), []
    for clip in inp["clips"]:
        handle, path = tempfile.mkstemp()
        os.close(handle)
        try:
            with open(path, "wb") as fh:
                fh.write(base64.b64decode(clip["audio_b64"]))
            row = {"id": clip.get("id"), **transcribe(path)}
            if want_phonemes:
                row["phonemes"] = phonemise(path)
            out.append(row)
        except Exception as exc:  # one bad clip must not cost the batch
            out.append({"id": clip.get("id"), "error": f"{type(exc).__name__}: {exc}"})
        finally:
            os.unlink(path)
    return {"results": out, "seconds": round(time.time() - started, 2), "device": DEVICE}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
