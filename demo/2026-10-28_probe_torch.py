import sys, time, resource, torch
from qwen_asr import Qwen3ASRModel
dev, dt = sys.argv[1], getattr(torch, sys.argv[2])
t = time.time()
m = Qwen3ASRModel.from_pretrained("data/models/Qwen3-ASR-0.6B", dtype=dt, device_map=dev, max_new_tokens=512)
print(f"load {time.time()-t:.1f}s device={m.device} dtype={m.dtype}")
for f in ["zh", "en", "mix", "zh"]:
    t = time.time()
    r = m.transcribe(f"data/samples/{f}.wav")[0]
    if dev == "mps": torch.mps.synchronize()
    print(f"{f} {time.time()-t:.2f}s [{r.language}] {r.text}")
print("maxrss MB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20)
if dev == "mps": print("mps driver MB", torch.mps.driver_allocated_memory()/2**20)
