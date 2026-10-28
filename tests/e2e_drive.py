"""端到端驱动：发快捷键 → 外放样本让麦克风收音 → 再发快捷键。用法：e2e_drive.py sample.wav [hotkey]"""
import subprocess, sys, time
from Quartz import CGEventCreateKeyboardEvent, CGEventPost, CGEventSetFlags, kCGHIDEventTap, kCGEventFlagMaskAlternate, kCGEventFlagMaskShift

def press(vk, flags):
    for down in (True, False):
        e = CGEventCreateKeyboardEvent(None, vk, down); CGEventSetFlags(e, flags); CGEventPost(kCGHIDEventTap, e)
    time.sleep(0.05)

HOT = (49, kCGEventFlagMaskAlternate | kCGEventFlagMaskShift)
press(*HOT); time.sleep(0.4)
if len(sys.argv) > 2 and sys.argv[2] == "cancel":
    subprocess.run(["afplay", sys.argv[1]]); press(53, 0); sys.exit()
subprocess.run(["afplay", sys.argv[1]]); time.sleep(0.3)
press(*HOT)
