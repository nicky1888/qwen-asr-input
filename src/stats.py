"""从日志的 `deliver ok` 行统计用量。只读次数、音频秒数、字数，口述原文从不进日志，这里也拿不到。"""
import re
import time

LINE = re.compile(r"^(\d{4}-\d{2}-\d{2}) \S+ \w+ deliver ok .*?audio=([\d.]+)s .*?chars=(\d+)")
TYPING_CPM = 40  # 打字速度估算：每分钟 40 字


def empty():
    return {"count": 0, "chars": 0, "audio_s": 0.0, "saved_min": 0.0}


def parse_log_stats(path, today=None):
    """返回 {"today": {...}, "total": {...}}。日志无轮转，逐行流式读，只做一次正则匹配。"""
    today = today or time.strftime("%Y-%m-%d")
    out = {"today": empty(), "total": empty()}
    try:
        f = open(path, encoding="utf-8", errors="replace")
    except OSError:
        return out
    with f:
        for line in f:
            m = LINE.match(line)
            if not m:
                continue
            day, audio, chars = m.group(1), float(m.group(2)), int(m.group(3))
            for key in ("total", "today") if day == today else ("total",):
                s = out[key]
                s["count"] += 1
                s["chars"] += chars
                s["audio_s"] += audio
    for s in out.values():
        s["audio_s"] = round(s["audio_s"], 1)
        s["saved_min"] = round(max(0.0, s["chars"] / TYPING_CPM - s["audio_s"] / 60), 1)  # 估算：打字要的时间减去说话花的时间
    return out
