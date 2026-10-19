# voice-qa-worker

A Runpod Serverless worker that listens to voice clips and reports what it hears:
Whisper large-v3-turbo transcript, detected language, per-word confidence, and
a phoneme string from `wav2vec2-lv-60-espeak-cv-ft`. Both models are baked into
the image, so a cold worker starts in seconds instead of pip-installing and
downloading ~3 GB.

It is game-agnostic: it returns raw observations only. Expected text,
pronunciation tables and pass/fail thresholds live with the caller (for the
WitchSpring R mod, `pipeline/qa_serverless.py` + `wsrvoice/qa_asr.judge`).

The image is built by GitHub Actions on every push to `main` and published to
`ghcr.io/drhouse/voice-qa-worker`.

## Request

```json
{"input": {"clips": [{"id": "Home_0", "audio_b64": "<base64 ogg/wav>"}], "phonemes": true}}
```

Send ~15 clips per job to `https://api.runpod.ai/v2/<endpoint>/runsync`. Use a
**queue** endpoint (load-balanced endpoints drop requests when busy).
