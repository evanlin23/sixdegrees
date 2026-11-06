"""Voice (local TTS: Kokoro, Pocket TTS or Chatterbox), procedurally generated music bed + SFX, and the mix.

The narrator is configured in data/voice.json, e.g.
  {"engine": "kokoro", "voice": "am_michael", "speed": 1.12}
  {"engine": "pocket", "voice": "peter_yearsley"}          (any Pocket catalog name, wav path or hf:// url)
  {"engine": "chatterbox", "variant": "orig", "ref": null, "exaggeration": 0.7, "cfg": 0.4}
An episode can override it with a "voice" object in episode.json.

Everything here is generated locally, so there is nothing to license: no
stock music, no Content ID claims on the soundtrack.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import soundfile as sf

from .common import CACHE, DATA, MODELS, ROOT

SR = 24000
_kokoro = None

# Names TTS tends to mangle -> phonetic respelling. Extend as episodes need.
RESPELL = {
    "Wernher": "Vairner",
    "Göring": "Gerring",
    "Goebbels": "Gerbels",
    "Hatsune Miku": "Hot-sue-neh Mee-koo",
    "IShowSpeed": "I Show Speed",
    "MrBeast": "Mister Beast",
    "Jae-in": "Jay In",
    "Mr. T": "Mister T",
    "Jung-jae": "Joongjay",
}

# How names get pronounced (Kokoro's espeak front end guesses names by English spelling rules, so it says
# MON-roe, PYOO-tin, PEN-tag-un):
#   1. PHONEMES: exact IPA we set by hand (wins over everything). Use Wikipedia's pronunciation as the source.
#   1b. data/pronunciations.json: English IPA pulled automatically from each person's Wikipedia article
#       (sixdeg/pronounce.py, run before every render).
#   2. The CMU Pronouncing Dictionary (134k words, many names) for every capitalized word it knows.
#   3. espeak's guess, only for ordinary lowercase words.
# QA fails when a capitalized word is in neither PHONEMES nor CMU, so every unusual name gets a human-set IPA.
# Keys are matched as whole words, case-sensitive; multi-word keys are fine.
PHONEMES = {
    "Kanye": "kˈɑːnjeɪ",
    "Durk": "dˈɜːɹk",
    "Mandalorian": "mˌændəlˈɔːɹiən",
    "Grogu": "ɡɹˈoʊɡuː",
    "Kim Jong Un": "kˌɪm dʒʌŋˈʊn",  # "Jong Un" as one word, or Kokoro pauses before "Un"
    "Kim Jong-un": "kˌɪm dʒʌŋˈʊn",
    "Bongbong": "bˈɑːŋbɑːŋ",
    "Svetek": "svˈɛtɛk",
    "Khrushchev": "kɹuːʃtʃˈɔf",
    "Rivaldo": "ɹᵻvˈɑːldoʊ",
    "Ronaldinho": "ɹˌɑːnəldˈiːnjoʊ",
    "Neymar": "neɪmˈɑːɹ",
    "Bolsonaro": "bˌoʊlsənˈɑːɹoʊ",
    "Jair": "ʒɑːˈiːɹ",
    "Tereshkova": "tˌɛɹɛʃkˈoʊvə",
    "Valentina": "vˌæləntˈiːnə",
    "Grug": "ɡɹˈʌɡ",
    "Fazbear": "fˈæzbɛɹ",  # Wikipedia pull gives the "FNAF" acronym for this one
    "Emmanuel Macron": "ɪmˈænjuəl mækɹˈɔːn",
    "Lamine Yamal": "ləmˈiːn jəmˈɑːl",
    "Yamal": "jəmˈɑːl",
    "Pedro Sánchez": "pˈɛdɹoʊ sˈɑːntʃɛz",
    "Sánchez": "sˈɑːntʃɛz",
    "Comic-Con": "kˈɑːmɪk kˌɑːn",
    "MatPat": "mˈæt pˈæt",
    "Comicon": "kˈɑːmɪk kˌɑːn",
    "Imahara": "ˌiːməhˈɑːɹə",
    "Luffy": "lˈʌfi",
    "MythBusters": "mˈɪθbʌstɚz",
    "MythBuster": "mˈɪθbʌstɚ",
    "Haberkorn": "hˈeɪbɚkɔːɹn",
    "Clinkenbeard": "klˈɪŋkənbˌɪɹd",
    "Hendaye": "ɑːndˈaɪ",
    "Yosuke": "jˈoʊskeɪ",
    "Matsuoka": "mˌɑːtsuːˈoʊkə",
    "Zedong": "zˌɛdˈɑːŋ",
    "Mao Zedong": "mˈaʊ zˌɛdˈɑːŋ",
    "Medvedev": "mɛdvˈɛdɛf",
    "Adolf": "ˈædɑːlf",
    "Barack": "bəɹˈɑːk",
    "JD": "dʒˌeɪdˈiː",
    "Croods": "kɹˈuːdz",
    "WrestleMania": "ɹˌɛsəlmˈeɪniə",
    "Dmitry": "dəmˈiːtɹi",
    "Sadat": "sɑːdˈɑːt",
    "FUBAR": "fˈuːbɑːɹ",
    "Qin Shi Huang": "tʃˈɪn ʃˈɜː hwˈɑːŋ",
    "LaBeouf": "ləbˈʌf",
    "Shia": "ʃˈaɪə",
    "Filippino": "fˌɪlɪpˈiːnoʊ",
    # regnal numbers and Renaissance names (Adam/Zeus to The Rock)
    "Sixtus IV": "sˈɪkstəs ðə fˈɔːɹθ",
    "John Paul II": "dʒˈɑːn pˈɔːl ðə sˈɛkənd",
    "Sixtus": "sˈɪkstəs",
    "Henry IV": "hˈɛnɹi ðə fˈɔːɹθ",
    "Louis XIV": "lˈuːi ðə fˈɔːɹtiːnθ",
    "Raphael": "ɹˈæfiəl",
    "Disputa": "dˌɪspuːtˈɑː",
    "Macron": "mækɹˈɔːn",
    "Filippino": "fˌɪlɪpˈiːnoʊ",
    # CMU entries that read worse than espeak
    "de Gaulle": "də ɡˈɔːl",
    "Hogan": "hˈoʊɡən",
    "Sistine": "sˈɪstiːn",
    "Marie": "mɚɹˈiː",
    "Korea": "kəɹˈiːə",
    "Khalifa": "kəlˈiːfə",
    "Ovechkin": "ˈoʊvətʃkɪn",
    "Perugino": "pˌɛɹʊdʒˈiːnoʊ",
}

_ARPA = {"AA": "ɑː", "AE": "æ", "AH": "ʌ", "AO": "ɔː", "AW": "aʊ", "AY": "aɪ", "EH": "ɛ", "ER": "ɜː", "EY": "eɪ",
         "IH": "ɪ", "IY": "iː", "OW": "oʊ", "OY": "ɔɪ", "UH": "ʊ", "UW": "uː", "B": "b", "CH": "tʃ", "D": "d",
         "DH": "ð", "F": "f", "G": "ɡ", "HH": "h", "JH": "dʒ", "K": "k", "L": "l", "M": "m", "N": "n", "NG": "ŋ",
         "P": "p", "R": "ɹ", "S": "s", "SH": "ʃ", "T": "t", "TH": "θ", "V": "v", "W": "w", "Y": "j", "Z": "z",
         "ZH": "ʒ"}
_cmu = None


def cmu_ipa(word: str) -> str | None:
    """CMU dictionary pronunciation of `word` in Kokoro's IPA (stress mark before the vowel), or None."""
    global _cmu
    if _cmu is None:
        import cmudict
        _cmu = cmudict.dict()
    prons = _cmu.get(word.lower())
    if not prons:
        return None
    out = []
    for ph in prons[0]:
        base, st = ph.rstrip("012"), ph[len(ph.rstrip("012")):]
        if base == "AH" and st == "0":
            out.append("ə")
        elif base == "ER" and st == "0":
            out.append("ɚ")
        elif base == "IY" and st == "0":
            out.append("i")
        else:
            out.append(("ˈ" if st == "1" else "") + _ARPA[base])  # CMU's secondary stresses sound robotic
    return "".join(out)


_NAME = re.compile(r"\b[A-Z][a-zA-Z]+(?:-[A-Za-z]+)*\b")
_COMMON = {"I", "A", "In", "And", "The", "On", "At", "By", "As", "Then", "But", "So", "That", "This", "His", "Her"}


def name_ipa(text: str) -> dict[str, str]:
    """Every capitalized word/phrase in `text` we have a pronunciation for -> IPA."""
    from .pronounce import load
    found = {}
    for k, ipa in {**load(), **PHONEMES}.items():
        if re.search(rf"\b{re.escape(k)}\b", text):
            found[k] = ipa
    covered = " ".join(found)
    for w in set(_NAME.findall(text)):
        if w in _COMMON or w in covered.split():
            continue
        ipa = cmu_ipa(w)
        if ipa:
            found[w] = ipa
    return found


def unknown_names(text: str) -> list[str]:
    """Capitalized words with no PHONEMES or CMU entry: their pronunciation would be espeak's guess.
    (Only the Kokoro voice reads our phonemes; QA skips this gate for the Gemini voice.)"""
    from .pronounce import load
    known = " ".join(list(PHONEMES) + list(load()) + list(RESPELL.values())).split()
    return sorted({w for w in _NAME.findall(speakable(text)) if w not in _COMMON and w not in known
                   and not cmu_ipa(w)})


def phonemes(text: str) -> str:
    """Phonemize `text` for Kokoro with names taken from PHONEMES, then CMU, then espeak."""
    t = _tts().tokenizer
    todo = dict(sorted(name_ipa(text).items(), key=lambda kv: -len(kv[0])))
    for _ in range(3):
        sub, marks = text, {}
        for i, (k, ipa) in enumerate(todo.items()):
            tok = "Zqx" + "".join("bcdfghjklmnprstvw"[int(c)] for c in f"{i:02d}")
            new = re.sub(rf"\b{re.escape(k)}\b", tok, sub)
            if new != sub:
                sub = new
                marks[k] = (t.phonemize(tok, "en-us").strip(" .,?!"), ipa)
        ph = t.phonemize(sub, "en-us")
        missing = [k for k, (m, _) in marks.items() if m not in ph]
        if not missing:
            for m, ipa in marks.values():
                ph = ph.replace(m, ipa)
            return ph
        for k in missing:
            todo.pop(k)
    return t.phonemize(text, "en-us")


def _tts():
    global _kokoro
    if _kokoro is None:
        from kokoro_onnx import Kokoro
        _kokoro = Kokoro(str(MODELS / "kokoro-v1.0.onnx"), str(MODELS / "voices-v1.0.bin"))
    return _kokoro


_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def _two(n: int) -> str:
    if n < 20:
        return _ONES[n]
    t, o = divmod(n, 10)
    return _TENS[t] + ("-" + _ONES[o] if o else "")


def say_year(y: int) -> str:
    """1973 -> nineteen seventy-three, 1905 -> nineteen oh-five, 1900 -> nineteen hundred,
    2005 -> two thousand five, 2013 -> twenty thirteen (how a narrator says years)."""
    hi, lo = divmod(y, 100)
    if 2000 <= y <= 2009:
        return "two thousand" + ("" if y == 2000 else " " + _ONES[lo])
    if lo == 0:
        return _two(hi) + " hundred"
    if lo < 10:
        return _two(hi) + " oh-" + _ONES[lo]
    return _two(hi) + " " + _two(lo)


_ROMAN = {"II": "the Second", "III": "the Third", "IV": "the Fourth", "V": "the Fifth", "VI": "the Sixth",
          "VII": "the Seventh", "VIII": "the Eighth", "IX": "the Ninth", "X": "the Tenth", "XI": "the Eleventh",
          "XII": "the Twelfth", "XIII": "the Thirteenth", "XIV": "the Fourteenth", "XV": "the Fifteenth",
          "XVI": "the Sixteenth"}


def speakable(text: str) -> str:
    for k, v in RESPELL.items():
        text = text.replace(k, v)
    # A period after an initial or title reads as a full stop, so the voice pauses mid-name
    # ("George W... Bush", "Franklin D... Roosevelt"): drop it when the name continues.
    text = re.sub(r"\b(Mr|Mrs|Dr|St|Jr|Sr)\.(?=\s)", lambda m: {"Mr": "Mister", "Mrs": "Missus", "Dr": "Doctor",
                  "St": "Saint", "Jr": "Junior", "Sr": "Senior"}[m.group(1)], text)
    text = re.sub(r"\b([A-Z])\.(?=\s*[A-Z])(?!\s*(?:In|The|And|A|An|But|He|She|They|It|That|This|Then|After|"
                  r"Before|Later|When|Years|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten)\b)", r"\1", text)
    # Regnal numbers after a name: "Henry VIII" -> "Henry the Eighth" (I alone is left alone: it's a pronoun)
    text = re.sub(r"\b([A-Z][a-z]+)\s+(XVI|XV|XIV|XIII|XII|XI|IX|VIII|VII|VI|IV|III|II)\b",
                  lambda m: f"{m.group(1)} {_ROMAN[m.group(2)]}", text)
    text = re.sub(r"\b(Henry|Charles|Louis|George|Edward|William|Pius|Leo|Gregory|Philip|Ferdinand|Alfonso|Frederick|"
                  r"Clement|Boniface|Innocent|Urban|Paul|John|Peter|Ivan)\s+(V|X)\b",
                  lambda m: f"{m.group(1)} {_ROMAN[m.group(2)]}", text)
    # years: Kokoro reads "1973" as "nineteen hundred seventy-three"
    text = re.sub(r"\b(1[0-9]{3}|20[0-9]{2})\b(?![,.]?\d)", lambda m: say_year(int(m.group(1))), text)
    return text


DEFAULT_VOICE = {"engine": "kokoro", "voice": "am_michael", "speed": 1.12}
_pocket = None
_pocket_states: dict = {}


def voice_config(override: dict | None = None) -> dict:
    import json
    cfg = dict(DEFAULT_VOICE)
    vf = DATA / "voice.json"
    if vf.exists():
        cfg = json.loads(vf.read_text())
    if override:
        cfg = {**cfg, **override}
    return cfg


def _resample(a: np.ndarray, sr: int) -> np.ndarray:
    if sr == SR:
        return a.astype(np.float32)
    n = int(len(a) * SR / sr)
    return np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype(np.float32)


GEMINI_CACHE = CACHE / "gemini"     # one .flac per spoken line; synced with the tts-cache branch in CI


def _gemini_key(spoken: str, cfg: dict) -> str:
    import json
    from .tts_gemini import NOTES
    sig = {k: cfg.get(k) for k in ("voice", "model")}
    return hashlib.sha1(f"{json.dumps(sig, sort_keys=True)}|{NOTES}|{spoken}".encode()).hexdigest()


def _norm_words(text: str) -> list[str]:
    return [w for w in (re.sub(r"[^a-z0-9]", "", t.lower()) for t in text.replace("-", " ").split()) if w]


def _gemini_batch(spoken_lines: list[str], cfg: dict) -> list[np.ndarray]:
    """Voice all lines in ONE Gemini request, then cut the result back into lines with Whisper word timings."""
    import difflib
    import tempfile
    from .tts_gemini import SR as GSR, synth
    a = synth(spoken_lines, cfg.get("voice", "Achird"), cfg.get("model", "gemini-3.8-flash-tts"))
    a = _resample(a, GSR)
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "batch.wav"
        sf.write(wav, a, SR)
        heard = transcribe(wav)
    hw = [re.sub(r"[^a-z0-9]", "", w.lower()) for w, _, _ in heard]
    script, owner = [], []
    for li, line in enumerate(spoken_lines):
        for w in _norm_words(line):
            script.append(w)
            owner.append(li)
    match = {}
    for blk in difflib.SequenceMatcher(None, script, hw, autojunk=False).get_matching_blocks():
        for k in range(blk.size):
            match[blk.a + k] = blk.b + k
    n = len(spoken_lines)
    first, last = [None] * n, [None] * n
    for si, li in enumerate(owner):
        if si in match:
            if first[li] is None:
                first[li] = match[si]
            last[li] = match[si]
    for li in range(n):
        words = sum(1 for o in owner if o == li)
        got = sum(1 for si, o in enumerate(owner) if o == li and si in match)
        if first[li] is None or got < 0.6 * words:
            raise RuntimeError(f"Gemini skipped or garbled line {li + 1}: {spoken_lines[li][:60]}")
    cuts = [0.0]
    for li in range(n - 1):
        e, s2 = heard[last[li]][2], heard[first[li + 1]][1]
        cuts.append((e + s2) / 2 if s2 > e else s2)
    cuts.append(len(a) / SR)
    return [_trim(a[int(cuts[i] * SR):int(cuts[i + 1] * SR)]) for i in range(n)]


def prefetch(lines: list[str], voice: dict | None = None, log=print) -> None:
    """Gemini engine: voice every not-yet-cached line of an episode in one request (free tier: ~15 a day)."""
    cfg = voice_config(voice)
    if cfg.get("engine") != "gemini":
        return
    import os
    GEMINI_CACHE.mkdir(parents=True, exist_ok=True)
    spoken = [speakable(t) for t in lines]
    todo = [s for s in dict.fromkeys(spoken) if not (GEMINI_CACHE / f"{_gemini_key(s, cfg)}.flac").exists()]
    if not todo:
        return
    if not os.environ.get("GEMINI_API_KEY"):
        log("voice: no GEMINI_API_KEY here (local preview): narration is silent; GitHub renders Achird")
        return
    try:
        clips = _gemini_batch(todo, cfg)
    except Exception as e:  # noqa: BLE001 - reported here; the render then stops on the first unvoiced line
        log(f"voice: Gemini batch failed ({e})")
        return
    for s, c in zip(todo, clips):
        sf.write(GEMINI_CACHE / f"{_gemini_key(s, cfg)}.flac", c, SR)
    log(f"voice: Gemini {cfg.get('voice')} voiced {len(todo)} lines in one request")


def narrator(lines: list[str], voice: dict | None = None) -> str:
    """'gemini' if every line is voiced by the configured Gemini voice, else the engine actually used."""
    cfg = voice_config(voice)
    if cfg.get("engine") != "gemini":
        return cfg.get("engine", "kokoro")
    ok = all((GEMINI_CACHE / f"{_gemini_key(speakable(t), cfg)}.flac").exists() for t in lines)
    return "gemini" if ok else "none"


def _synth(text: str, cfg: dict) -> np.ndarray:
    eng = cfg.get("engine", "kokoro")
    if eng == "gemini":
        p = GEMINI_CACHE / f"{_gemini_key(text, cfg)}.flac"
        if p.exists():
            a, _ = sf.read(p, dtype="float32")
            return a
        import os
        if os.environ.get("GEMINI_API_KEY") or os.environ.get("GITHUB_ACTIONS"):
            raise RuntimeError("line not voiced by Gemini (daily quota?); the render waits, there is no backup voice: "
                               + text[:60])
        # local session without a key: silent placeholder so faces/layout can still be previewed.
        # QA marks this render as not publishable; GitHub renders the real voice in the PR check.
        return np.zeros(int(SR * max(1.0, 0.065 * len(text))), np.float32)
    if eng == "kokoro":
        k = _tts()
        v = cfg.get("voice", "am_michael")
        if isinstance(v, dict):  # blend, e.g. {"am_michael": 0.7, "am_onyx": 0.3}
            v = sum(k.get_voice_style(name) * w for name, w in v.items())
        a, sr = k.create(phonemes(text), voice=v, speed=cfg.get("speed", 1.12), lang="en-us", is_phonemes=True)
        return _resample(np.asarray(a), sr)
    if eng == "pocket":
        global _pocket
        if _pocket is None:
            from pocket_tts import TTSModel
            _pocket = TTSModel.load_model()
        v = cfg.get("voice", "peter_yearsley")
        if v not in _pocket_states:
            _pocket_states[v] = _pocket.get_state_for_audio_prompt(v)
        a = _pocket.generate_audio(_pocket_states[v], text)
        a = a.numpy() if hasattr(a, "numpy") else np.asarray(a)
        return _resample(a.squeeze(), _pocket.sample_rate)
    if eng == "chatterbox":
        import json
        import os
        import subprocess
        import tempfile
        py = os.environ.get("CHATTERBOX_PY", "/home/user/cbenv/bin/python")
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "o.wav"
            job = {**cfg, "text": text, "out": str(out)}
            subprocess.run([py, str(ROOT / "sixdeg" / "tts_chatterbox.py"), json.dumps(job)], check=True,
                           stdout=subprocess.DEVNULL)
            a, sr = sf.read(out, dtype="float32")
        return _resample(a if a.ndim == 1 else a.mean(1), sr)
    raise ValueError(f"unknown TTS engine {eng}")


# "Broadcast" narrator chain: warmth, less mud, presence, de-ess, gentle compression.
BROADCAST_EQ = ("highpass=f=70,lowshelf=f=140:g=3,equalizer=f=350:t=q:w=1:g=-2,"
                "equalizer=f=3200:t=q:w=1.2:g=3,deesser=i=0.4,"
                "acompressor=threshold=-20dB:ratio=3:attack=5:release=80:makeup=3")


def post_chain(a: np.ndarray, tempo: float = 1.0, broadcast: bool = False) -> np.ndarray:
    """Optional speed-up (pitch-preserving atempo) and broadcast EQ/compression."""
    import subprocess
    import tempfile
    filters = []
    if tempo != 1.0:
        filters.append(f"atempo={tempo}")
    if broadcast:
        filters.append(BROADCAST_EQ)
    with tempfile.TemporaryDirectory() as td:
        i, o = Path(td) / "i.wav", Path(td) / "o.wav"
        sf.write(i, a, SR)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(i), "-af", ",".join(filters), "-ar", str(SR),
                        str(o)], check=True)
        b, _ = sf.read(o, dtype="float32")
    return b


def say(text: str, voice: dict | None = None) -> tuple[np.ndarray, list[tuple[str, float, float]]]:
    """Synthesize `text`. Returns (mono float32 @ SR, word timings [(word, t0, t1)])."""
    import json
    cfg = voice_config(voice)
    src = cfg
    if cfg.get("engine") == "gemini" and not (GEMINI_CACHE / f"{_gemini_key(speakable(text), cfg)}.flac").exists():
        src = {"engine": "silent-preview"}                   # local preview without a key: never cached as real
    key = hashlib.sha1(f"{json.dumps(src, sort_keys=True)}|{json.dumps(PHONEMES, sort_keys=True)}|trim2|{speakable(text)}".encode()).hexdigest()  # respellings change the audio
    p = CACHE / "tts" / f"{key}.wav"
    if p.exists():
        a, _ = sf.read(p, dtype="float32")
    else:
        a = _synth(speakable(text), cfg)
        if src.get("broadcast") or src.get("tempo", 1.0) != 1.0:
            a = post_chain(a, tempo=src.get("tempo", 1.0), broadcast=bool(src.get("broadcast")))
        a = _trim(a / (np.abs(a).max() + 1e-9) * 0.9)
        p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(p, a, SR)
    return a, aligned_word_times(text, p, len(a) / SR)


_whisper = None


def transcribe(wav: Path) -> list[tuple[str, float, float]]:
    """Whisper word timestamps (cached next to the wav)."""
    import json
    jp = Path(str(wav) + ".words.json")
    if jp.exists():
        return [tuple(x) for x in json.loads(jp.read_text())]
    global _whisper
    if _whisper is None:
        from faster_whisper import WhisperModel
        _whisper = WhisperModel("base.en", device="cpu", compute_type="int8",
                                download_root=str(MODELS / "whisper"))
    a, sr = sf.read(str(wav), dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    if sr != 16000:  # whisper wants 16 kHz; linear resample is fine for speech recognition
        n = int(len(a) * 16000 / sr)
        a = np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype(np.float32)
    segs, _ = _whisper.transcribe(a, word_timestamps=True, language="en", beam_size=5)
    words = [(w.word.strip(), float(w.start), float(w.end)) for sg in segs for w in (sg.words or [])]
    jp.write_text(json.dumps(words))
    return words


def aligned_word_times(text: str, wav: Path, dur: float) -> list[tuple[str, float, float]]:
    """Script words with real timings from Whisper when the word counts line up, else estimates."""
    est = _word_times(text, dur)
    try:
        heard = transcribe(wav)
    except Exception:  # noqa: BLE001 - whisper is a nicety for captions
        return est
    if len(heard) == len(est):
        return [(w, h[1], h[2]) for (w, _, _), h in zip(est, heard)]
    # counts differ (numbers, hyphens...): map each script word to the heard word at the same relative position
    out = []
    for i, (w, a, b) in enumerate(est):
        j = min(len(heard) - 1, int(round(i * (len(heard) - 1) / max(1, len(est) - 1)))) if heard else -1
        out.append((w, heard[j][1], heard[j][2]) if j >= 0 else (w, a, b))
    return out



def _trim(a: np.ndarray, thr: float = 0.01) -> np.ndarray:
    """Cut leading/trailing silence. Speech starts where the level stays up for 60 ms, so a lone click
    before the first word (Gemini adds one) doesn't count as speech."""
    hop = int(0.01 * SR)
    n = len(a) // hop
    if n < 8:
        return a
    peak = np.abs(a).max() + 1e-9
    db = np.array([20 * np.log10(np.sqrt(np.mean(a[i * hop:(i + 1) * hop] ** 2)) / peak + 1e-9) for i in range(n)])
    loud = db > -35
    run = np.convolve(loud.astype(int), np.ones(6, int), "valid")      # 6 x 10 ms frames in a row
    on = np.where(run == 6)[0]
    if len(on) == 0:
        return a
    s = max(0, (on[0] - 2) * hop)                                     # 20 ms before the onset
    e = min(len(a), (on[-1] + 6 + 8) * hop)                           # 80 ms after the last sustained sound
    out = a[s:e].copy()
    f = min(len(out), int(0.005 * SR))
    out[:f] *= np.linspace(0, 1, f)                                   # no click at the cut
    return out


def _word_times(text: str, dur: float) -> list[tuple[str, float, float]]:
    """Estimate word timings: proportional to syllable-ish weight, plus pauses at punctuation."""
    words = text.split()
    if not words:
        return []
    weights = []
    for w in words:
        core = re.sub(r"[^A-Za-z0-9]", "", w)
        wt = 0.6 + 0.28 * max(1, len(re.findall(r"[aeiouy]+", core.lower())))
        if core.isdigit():
            wt += 0.4 * len(core)
        pause = 0.0
        if w.endswith((",", ";", ":")):
            pause = 0.45
        elif w.endswith((".", "!", "?")):
            pause = 0.75
        weights.append((wt, pause))
    total = sum(a + b for a, b in weights)
    t, out = 0.0, []
    for w, (wt, pause) in zip(words, weights):
        d = dur * wt / total
        out.append((w, t, t + d))
        t += d + dur * pause / total
    return out


# ------------------------------------------------------------------ music + sfx

def _env(n, a=0.01, r=0.3):
    t = np.arange(n) / SR
    e = np.minimum(1, t / a) * np.exp(-t / r)
    return e.astype(np.float32)


def music_bed(seconds: float, seed: int = 0, bpm: float = 104) -> np.ndarray:
    """Dark, driving 'mystery' bed: pad chords + pulsing bass + soft kick/hat. Mono."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR) + SR
    out = np.zeros(n, np.float32)
    beat = 60 / bpm
    # minor progressions (semitones from root), pick by seed for variety
    progs = [[0, -4, -7, -2], [0, -2, -4, -5], [0, 3, -4, -2], [0, -5, -4, -2]]
    prog = progs[seed % len(progs)]
    root = 110 * 2 ** (rng.integers(-3, 3) / 12)  # around A2
    bar = beat * 4
    t_all = np.arange(n) / SR
    for bi in range(int(seconds / bar) + 2):
        semi = prog[bi % len(prog)]
        f0 = root * 2 ** (semi / 12)
        s, e = int(bi * bar * SR), min(n, int((bi + 1) * bar * SR))
        if s >= n:
            break
        t = t_all[s:e] - t_all[s]
        # pad: minor triad, detuned saws softened
        pad = np.zeros(e - s, np.float32)
        for iv in (0, 3, 7, 12):
            f = f0 * 2 * 2 ** (iv / 12)
            for det in (-0.15, 0.15):
                ph = 2 * np.pi * (f + det) * t
                pad += (np.sin(ph) + 0.3 * np.sin(2 * ph) + 0.12 * np.sin(3 * ph)).astype(np.float32)
        fade = np.minimum(1, np.minimum(t / 0.25, (bar - t) / 0.25)).astype(np.float32)
        out[s:e] += 0.018 * pad * fade
        # bass: eighth-note pulse
        for k in range(8):
            bs = s + int(k * beat / 2 * SR)
            bl = int(beat / 2 * SR * 0.9)
            if bs + bl > n:
                break
            tt = np.arange(bl) / SR
            out[bs:bs + bl] += 0.11 * np.sin(2 * np.pi * f0 * tt) * _env(bl, 0.005, 0.12)
    # drums
    for k in range(int(seconds / beat) + 2):
        ks = int(k * beat * SR)
        if ks >= n:
            break
        if k % 2 == 0:
            kl = min(int(0.25 * SR), n - ks)
            tt = np.arange(kl) / SR
            f = 120 * np.exp(-tt * 30) + 45
            out[ks:ks + kl] += 0.28 * np.sin(2 * np.pi * np.cumsum(f) / SR) * _env(kl, 0.002, 0.09)
        hs = ks + int(beat / 2 * SR)
        hl = min(int(0.05 * SR), n - hs)
        if hl > 0:
            out[hs:hs + hl] += 0.03 * rng.standard_normal(hl).astype(np.float32) * _env(hl, 0.001, 0.015)
    return out[: int(seconds * SR)]


def whoosh(dur=0.45, seed=1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(dur * SR)
    noise = rng.standard_normal(n).astype(np.float32)
    # moving one-pole lowpass sweep
    out = np.zeros(n, np.float32)
    y = 0.0
    for i in range(n):
        a = 0.02 + 0.5 * np.sin(np.pi * i / n) ** 2
        y += a * (noise[i] - y)
        out[i] = y
    env = np.sin(np.pi * np.arange(n) / n) ** 1.5
    return 0.5 * out * env


def ding(f=1320.0, dur=0.5) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    s = np.sin(2 * np.pi * f * t) + 0.4 * np.sin(2 * np.pi * f * 2.01 * t)
    return (0.16 * s * _env(n, 0.002, 0.18)).astype(np.float32)


def boom(dur=1.2) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = 70 * np.exp(-t * 3) + 30
    return (0.5 * np.sin(2 * np.pi * np.cumsum(f) / SR) * _env(n, 0.003, 0.4)).astype(np.float32)


def mix(total: float, voice: list[tuple[float, np.ndarray]], sfx: list[tuple[float, np.ndarray]],
        seed: int, out: Path, music: np.ndarray | None = None) -> Path:
    n = int(total * SR)
    v = np.zeros(n, np.float32)
    for t, a in voice:
        s = int(t * SR)
        e = min(n, s + len(a))
        v[s:e] += a[: e - s]
    m = music[:n] if music is not None else music_bed(total, seed=seed)
    if len(m) < n:
        m = np.pad(m, (0, n - len(m)))
    # sidechain-ish ducking under the voice
    env = np.abs(v)
    k = int(0.15 * SR)
    env = np.convolve(env, np.ones(k) / k, mode="same")
    speaking = np.clip(env / (env.max() + 1e-9) * 4, 0, 1)
    duck = 1 - 0.7 * speaking                 # music drops ~10 dB under the voice
    fx = np.zeros(n, np.float32)
    for t, a in sfx:
        s = int(t * SR)
        e = min(n, s + len(a))
        if s < n:
            fx[s:e] += a[: e - s]
    # fade music out at the end
    fo = int(0.03 * SR)   # no fade-out: the video cuts hard and loops
    m[-fo:] *= np.linspace(1, 0, fo)
    fx_duck = 1 - 0.5 * speaking               # effects also step back while someone is talking
    y = v * 1.0 + m * duck * 0.55 + fx * fx_duck * 0.5
    y = y / max(1e-6, np.abs(y).max())
    # soft limiter: lowers crest factor so loudness normalisation can reach -14 LUFS
    drive = 2.2
    y = np.tanh(y * drive) / np.tanh(drive)
    raw = out.with_name(out.stem + "_raw.wav")
    sf.write(raw, y * 0.95, SR)
    loudnorm(raw, out)
    return out


def loudnorm(src: Path, dst: Path, target=-14.0, tp=-1.5):
    """Two-pass EBU R128 normalisation (single pass undershoots on peaky voice)."""
    import json
    import subprocess
    f = f"loudnorm=I={target}:TP={tp}:LRA=11"
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(src), "-af", f + ":print_format=json",
                        "-f", "null", "-"], capture_output=True, text=True)
    m = json.loads(r.stderr[r.stderr.rfind("{"):r.stderr.rfind("}") + 1])
    f2 = (f + f":measured_I={m['input_i']}:measured_TP={m['input_tp']}:measured_LRA={m['input_lra']}"
          f":measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-af", f2, "-ar", "48000", str(dst)],
                   check=True)
