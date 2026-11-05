"""Chatterbox worker: runs in its own venv (pinned torch), called by audio.py.

    $CHATTERBOX_PY sixdeg/tts_chatterbox.py '{"text": "...", "out": "o.wav", "variant": "orig",
                                             "ref": null, "exaggeration": 0.7, "cfg": 0.4}'

Setup (once per container, ~3 GB):
    python3 -m venv /home/user/cbenv
    /home/user/cbenv/bin/pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
    /home/user/cbenv/bin/pip install chatterbox-tts soundfile
"""
import json
import sys

import librosa
import numpy as np
import soundfile as sf
import torch

# chatterbox on CPU mixes float64 (librosa) into float32 models; keep everything float32
_ll, _lr = librosa.load, librosa.resample
librosa.load = lambda *a, **k: (lambda r: (r[0].astype(np.float32), r[1]))(_ll(*a, **k))
librosa.resample = lambda *a, **k: _lr(*a, **k).astype(np.float32)
import chatterbox.models.s3tokenizer.s3tokenizer as _s3  # noqa: E402

_orig = _s3.S3Tokenizer.log_mel_spectrogram


def _patched(self, *a, **k):
    a = tuple(x.float() if torch.is_tensor(x) else x for x in a)
    return _orig(self, *a, **k)


_s3.S3Tokenizer.log_mel_spectrogram = _patched


def main():
    job = json.loads(sys.argv[1])
    torch.set_num_threads(4)
    ref = job.get("ref")
    if ref and ref.startswith("hf://"):
        from huggingface_hub import hf_hub_download
        parts = ref[5:].split("/")
        ref = hf_hub_download("/".join(parts[:2]), "/".join(parts[2:]))
    if job.get("variant", "orig") == "turbo":
        from chatterbox.tts_turbo import ChatterboxTurboTTS
        m = ChatterboxTurboTTS.from_pretrained(device="cpu")
        w = m.generate(job["text"], audio_prompt_path=ref) if ref else m.generate(job["text"])
    else:
        from chatterbox.tts import ChatterboxTTS
        m = ChatterboxTTS.from_pretrained(device="cpu")
        kw = dict(exaggeration=job.get("exaggeration", 0.5), cfg_weight=job.get("cfg", 0.5))
        w = m.generate(job["text"], audio_prompt_path=ref, **kw) if ref else m.generate(job["text"], **kw)
    sf.write(job["out"], w.squeeze().numpy(), m.sr)


if __name__ == "__main__":
    main()
