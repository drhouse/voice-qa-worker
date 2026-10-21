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
import numpy as np
import runpod
import torch
from faster_whisper import BatchedInferencePipeline, WhisperModel, decode_audio
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCTC, Wav2Vec2FeatureExtractor

PHONEME_MODEL = "facebook/wav2vec2-lv-60-espeak-cv-ft"
DEVICE = "cuda" if torch.cuda.is_available() else "CPU-FALLBACK"

# Loaded once per worker at import; the handler reuses them for every job.
WHISPER = WhisperModel(os.environ.get("WHISPER_MODEL", "large-v3-turbo"),
                       device="cuda" if DEVICE == "cuda" else "cpu",
                       compute_type="float16" if DEVICE == "cuda" else "int8")
BATCHED = BatchedInferencePipeline(model=WHISPER)
# Clips per GPU decode step. Batched mode is the default; BATCH_SIZE=0 restores
# the one-clip-at-a-time path for parity checks.
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
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


GAP = 0.5        # seconds of silence between clips in the joined audio
LONG = 28.0      # batched mode reads at most 30 s per clip; longer ones go one by one


def language_of(audio):
    lang, prob, _ = WHISPER.detect_language(audio)
    return str(lang), float(round(prob, 4))


def transcribe_batch(audios):
    """Whisper over many short clips in one batched pass.

    The clips are joined with short gaps and handed over as clip_timestamps, so
    each clip is its own chunk and the pipeline decodes BATCH_SIZE chunks per
    step. Settings match transcribe(): greedy, temperature 0, no conditioning on
    previous text, word timestamps. Language is still detected per clip.
    """
    rate, gap = 16000, np.zeros(int(GAP * 16000), dtype=np.float32)
    joined, spans, t = [], [], 0.0
    for a in audios:
        d = len(a) / rate
        spans.append((t, t + d))
        joined += [a, gap]
        t += d + GAP
    segments, _ = BATCHED.transcribe(
        np.concatenate(joined), language="en", beam_size=1, temperature=0,
        condition_on_previous_text=False, word_timestamps=True, batch_size=BATCH_SIZE,
        clip_timestamps=[{"start": s, "end": e} for s, e in spans])
    per = [{"said": [], "logprobs": [], "words": []} for _ in audios]
    for seg in segments:
        k = next((i for i, (s, e) in enumerate(spans) if s - 0.05 <= seg.start < e + 0.05), None)
        if k is None:
            continue
        per[k]["said"].append(seg.text)
        per[k]["logprobs"].append(seg.avg_logprob)
        for w in seg.words or []:
            per[k]["words"].append({"word": w.word.strip(), "prob": float(round(w.probability, 4))})
    out = []
    for a, p in zip(audios, per):
        lang, lprob = language_of(a)
        lp = p["logprobs"]
        out.append({"language": lang, "language_prob": lprob, "text": "".join(p["said"]).strip(),
                    "avg_logprob": float(round(sum(lp) / len(lp), 4)) if lp else None,
                    "words": p["words"], "duration": float(round(len(a) / 16000, 3))})
    return out


def phonemise_batch(audios):
    feats = EXTRACTOR(audios, sampling_rate=16000, return_tensors="pt", padding=True,
                      return_attention_mask=True)
    values, mask = feats.input_values, feats.attention_mask
    if DEVICE == "cuda":
        values, mask = values.to("cuda").half(), mask.to("cuda")
    with torch.no_grad():
        frames = torch.argmax(CTC(values, attention_mask=mask).logits, dim=-1).tolist()
    out = []
    for a, row in zip(audios, frames):
        n = int(len(row) * len(a) / values.shape[1]) + 1   # drop frames that are padding
        symbols, previous = [], None
        for index in row[:n]:
            if index != previous:
                s = LOOKUP.get(index, "")
                if s and not (s.startswith("<") and s.endswith(">")):
                    symbols.append(s)
            previous = index
        out.append(symbols)
    return out


def handler_batched(inp, want_phonemes, started):
    out, audios, keep = [], [], []
    for clip in inp["clips"]:
        handle, path = tempfile.mkstemp()
        os.close(handle)
        try:
            with open(path, "wb") as fh:
                fh.write(base64.b64decode(clip["audio_b64"]))
            audios.append(decode_audio(path, sampling_rate=16000))
            keep.append(clip.get("id"))
        except Exception as exc:
            out.append({"id": clip.get("id"), "error": f"{type(exc).__name__}: {exc}"})
        finally:
            os.unlink(path)
    short = [i for i, a in enumerate(audios) if len(a) / 16000 <= LONG]
    long_ = [i for i, a in enumerate(audios) if len(a) / 16000 > LONG]
    rows = {}
    if short:
        for i, r in zip(short, transcribe_batch([audios[i] for i in short])):
            rows[i] = r
    for i in long_:  # rare runaway clips: the old path, no 30 s cap
        segs, info = WHISPER.transcribe(audios[i], language=None, beam_size=1, temperature=0,
                                        condition_on_previous_text=False, word_timestamps=True,
                                        vad_filter=False)
        segs = list(segs)
        lp = [s.avg_logprob for s in segs]
        rows[i] = {"language": str(info.language), "language_prob": float(round(info.language_probability, 4)),
                   "text": "".join(s.text for s in segs).strip(),
                   "avg_logprob": float(round(sum(lp) / len(lp), 4)) if lp else None,
                   "words": [{"word": w.word.strip(), "prob": float(round(w.probability, 4))}
                             for s in segs for w in (s.words or [])],
                   "duration": float(round(info.duration, 3))}
    if want_phonemes and audios:
        for i, ph in enumerate(phonemise_batch(audios)):
            rows[i]["phonemes"] = ph
    out += [{"id": keep[i], **rows[i]} for i in range(len(audios))]
    return {"results": out, "seconds": round(time.time() - started, 2), "device": DEVICE,
            "batched": True}


def handler(job):
    inp = job["input"]
    want_phonemes = inp.get("phonemes", True)
    started, out = time.time(), []
    if BATCH_SIZE > 0:
        return handler_batched(inp, want_phonemes, started)
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
