"""Gemini TTS (AI Studio API). One request voices a whole script; callers split it by Whisper word timings.

Key: env GEMINI_API_KEY (GitHub secret), or a network secret that the session proxy injects as x-goog-api-key.
Output: mono float32 at 24 kHz.
"""
from __future__ import annotations

import base64
import json
import os
import time

import numpy as np
import requests

API = "https://generativelanguage.googleapis.com/v1beta"
SR = 24000
# Google's recommended TTS prompt layout: notes for the "director", then a clearly marked transcript.
# Only the transcript is spoken. (A plain instruction line in front of the text gets read aloud.)
NOTES = """# AUDIO PROFILE: Buh, narrator of a YouTube Shorts channel about surprising connections between famous people.

## DIRECTOR'S NOTES
Style: upbeat, curious, confident; a quick "no way!" energy on each reveal. American accent.
Pace: fast, like a Shorts narrator, with only a short beat between lines.
Do not read these notes aloud. Speak only the transcript below, exactly as written, in order.

## TRANSCRIPT
"""


def _headers():
    k = os.environ.get("GEMINI_API_KEY")
    return {"x-goog-api-key": k} if k else {}


def list_tts_models() -> list[str]:
    r = requests.get(f"{API}/models", params={"pageSize": 200}, headers=_headers(), timeout=30)
    r.raise_for_status()
    return [m["name"].split("/")[-1] for m in r.json().get("models", []) if "tts" in m["name"].lower()]


def clean_tail(a: np.ndarray, sr: int = SR) -> np.ndarray:
    """Gemini sometimes ends with a short loud noise burst after the speech has finished (a 'scratch').
    Cut at the last real silence if what follows it is under 0.6 s, then fade out."""
    hop = int(sr * 0.01)
    n = len(a) // hop
    db = np.array([20 * np.log10(np.sqrt(np.mean(a[i * hop:(i + 1) * hop] ** 2)) + 1e-9) for i in range(n)])
    quiet = db < -50
    i = n - 1
    while i > 0 and not quiet[i]:          # walk back over the trailing burst
        i -= 1
    if (n - 1 - i) * 0.01 < 0.6 and i > 0:
        j = i
        while j > 0 and quiet[j]:          # and back over the silence before it
            j -= 1
        a = a[:min(len(a), (j + 15) * hop)]   # keep 150 ms after the last sound
    f = min(len(a), int(sr * 0.03))
    a = a.copy()
    a[-f:] *= np.linspace(1, 0, f)
    return a


def synth(lines: list[str], voice: str = "Puck", model: str = "gemini-3.8-flash-tts",
          notes: str = NOTES, tries: int = 4) -> np.ndarray:
    prompt = notes + "\n".join(lines)
    body = {"contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseModalities": ["AUDIO"],
                                 "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}}}}
    for i in range(tries):
        r = requests.post(f"{API}/models/{model}:generateContent", headers=_headers(), json=body, timeout=300)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(20 * (i + 1))
            continue
        if r.status_code != 200:
            raise RuntimeError(f"Gemini TTS {r.status_code}: {r.text[:400]}")
        part = r.json()["candidates"][0]["content"]["parts"][0]["inlineData"]
        pcm = np.frombuffer(base64.b64decode(part["data"]), dtype="<i2")
        return clean_tail(pcm.astype(np.float32) / 32768.0)
    raise RuntimeError("Gemini TTS: rate-limited, gave up")


if __name__ == "__main__":   # samples for the voice comparison: python -m sixdeg.tts_gemini <slug> <outdir>
    import subprocess
    import sys

    import soundfile as sf

    from .audio import speakable
    slug, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    print("TTS models:", list_tts_models())
    spec = json.load(open(f"episodes/{slug}/episode.json"))
    lines = [speakable(spec["hook"])] + [speakable(l["line"]) for l in spec["links"]]
    for model in sys.argv[3:] or ["gemini-3.8-flash-tts"]:
        try:
            a = synth(lines, "Puck", model, tries=2)
        except RuntimeError as e:
            print("skipped", model, e)
            continue
        wav = f"{outdir}/puck-{model}.wav"
        sf.write(wav, a, SR)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav, "-b:a", "160k", wav[:-4] + ".mp3"], check=True)
        os.remove(wav)
        print("wrote", wav[:-4] + ".mp3", f"{len(a) / SR:.1f}s")
