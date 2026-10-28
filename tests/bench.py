"""无界面基准：与常驻工具同一加载路径，测延迟、内存、重复稳定性、静音。用法：bench.py [repeat]"""
import os, subprocess, sys, time
from pathlib import Path
import numpy as np, soundfile as sf
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from voice_input import DEFAULT_MODEL, SR, load_asr

def fp():
    out = subprocess.run(["footprint", "-p", str(os.getpid())], capture_output=True, text=True).stdout
    line = next(l for l in out.splitlines() if "Footprint:" in l)
    return line.split("Footprint:")[1].split("(")[0].strip()

S = Path(__file__).resolve().parent.parent / "data/samples"
print("imports done, footprint", fp())
model, load_s, warm_s = load_asr(DEFAULT_MODEL)
print(f"load {load_s:.1f}s warmup {warm_s:.1f}s, idle footprint {fp()}")
for name in ["s5", "s15", "s30", "zh", "en", "en2", "mix", "tech"]:
    wav, sr = sf.read(S / f"{name}.wav", dtype="float32")
    t = time.time(); r = model.transcribe((wav, sr))[0]
    print(f"{name:4s} audio {len(wav)/sr:4.1f}s  asr {time.time()-t:.2f}s  [{r.language}] {r.text}")
print("after samples footprint", fp())
for label, wav in [("silence", np.zeros(SR * 3, np.float32)), ("noise", np.random.default_rng(1).normal(0, 0.01, SR * 3).astype(np.float32))]:
    r = model.transcribe((wav, SR))[0]; print(f"{label}: [{r.language}] {r.text!r}")
n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
wav, sr = sf.read(S / "s5.wav", dtype="float32"); ts = []
for i in range(n):
    t = time.time(); txt = model.transcribe((wav, sr))[0].text; ts.append(time.time() - t)
    if i in (0, n - 1): print(f"repeat {i+1}: {ts[-1]:.2f}s footprint {fp()} text_same={txt == model.transcribe((wav, sr))[0].text}")
print(f"repeat {n}x median {np.median(ts):.2f}s max {max(ts):.2f}s")
