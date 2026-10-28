import sys; from pathlib import Path
sys.path.insert(0, "src"); import soundfile as sf
from voice_input import DEFAULT_MODEL, load_asr
m, *_ = load_asr(DEFAULT_MODEL)
ctx = "GitHub Obsidian Claude Code launchd plist bootout bootstrap caffeinate PyTorch 回测 纳斯达克"
for n in ["s30", "tech", "mix", "s15", "zh", "en2"]:
    w, sr = sf.read(f"data/samples/{n}.wav", dtype="float32")
    print(n, "| 无词表:", m.transcribe((w, sr))[0].text); print(n, "| 有词表:", m.transcribe((w, sr), context=ctx)[0].text)
