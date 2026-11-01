"""Qwen3-ASR 本地语音输入：全局快捷键切换录音，识别后粘贴到原输入框。

主线程跑 AppKit 事件循环（菜单栏、AX、剪贴板都在这里），只做不会卡住的事；
键盘监听和录音在原生线程（keytap.c、audiorec.c），开关麦、模型加载和识别在后台线程，
结果用 callAfter 回主线程插入。后台的每一步都有超时，卡住了也只影响这一段，主线程照常响应。
"""
import argparse
import ctypes
import functools
import json
import logging
import os
import queue
import struct
import subprocess
import threading
import time
from concurrent.futures import Future
from pathlib import Path

import numpy as np
from AppKit import (
    NSAlert, NSApplication, NSApplicationActivationPolicyAccessory, NSColor, NSImage, NSMenu, NSMenuItem,
    NSPasteboard, NSPasteboardItem, NSPasteboardTypeString, NSSound,
    NSStatusBar, NSVariableStatusItemLength, NSWorkspace,
)
from ApplicationServices import (
    AXIsProcessTrustedWithOptions, AXUIElementCopyAttributeValue,
    AXUIElementCreateApplication, AXUIElementIsAttributeSettable,
    AXUIElementCreateSystemWide, AXUIElementSetAttributeValue, AXUIElementSetMessagingTimeout, kAXTrustedCheckOptionPrompt,
)
from CoreFoundation import CFEqual
from Foundation import (
    NSActivityUserInitiatedAllowingIdleSystemSleep, NSBundle, NSDistributedNotificationCenter, NSObject, NSProcessInfo,
)
from PyObjCTools import AppHelper
from Quartz import (
    CGEventCreateKeyboardEvent, CGEventPost, CGEventSetFlags, kCGEventFlagMaskAlternate, kCGEventFlagMaskCommand,
    kCGEventFlagMaskControl, kCGEventFlagMaskSecondaryFn, kCGEventFlagMaskShift, kCGEventTapDisabledByTimeout,
    kCGHIDEventTap,
)

import i18n
from cleanup import clean
from cn_numbers import to_digits
from i18n import tr
from overlay import Overlay
from panel import Panel
from stats import parse_log_stats

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "data/models/Qwen3-ASR-0.6B"
VOCAB_PATH = ROOT / "config/vocab.txt"
SETTINGS_PATH = ROOT / "config/settings.json"
DEFAULT_HOTKEY = "alt+shift+space"
LOG_PATH = Path.home() / "Library/Logs/qwen-asr-input.log"
PID_PATH = Path.home() / "Library/Caches/qwen-asr-input.pid"
REOPEN_NOTE = "com.nickyz.qwen-asr-input.reopen"  # Dock 启动器在你再次点图标时发出
SR = 16000
SILENCE_PEAK = 0.01        # 峰值低于此视为没录到声音（含麦克风未授权时的全零）
PASTE_RESTORE_DELAY = 0.6  # 发出 ⌘V 后等目标 App 读完剪贴板再恢复
DEFAULT_MAX_SECONDS = 120     # 单次录音上限；优先级：命令行显式传 > settings.json > 这个默认值
DEFAULT_IDLE_UNLOAD_MIN = 10  # 闲置这么久卸下模型（约省 2GB），下次按快捷键边录边重载；0 表示常驻
TEXT_ROLES = {"AXTextField", "AXTextArea", "AXComboBox", "AXSearchField"}
AX_TIMEOUT = 1.0         # 单次 AX 查询最多等这么久；系统默认 6 秒，目标 App 忙时会把主线程整个卡住
MIC_TIMEOUT = 5.0        # 开麦、关麦最多等这么久，超了按麦克风卡住处理
MODEL_WAIT_TIMEOUT = 60  # 识别前等模型重载的上限
ASR_CHUNK_SEC = 60       # 长录音切成不超过这么长的段逐段识别：单段生成上限 1024 token，一整段 3 分钟以上会被截断

# 菜单栏图标用 SF Symbols；录音时染红，其余跟随系统明暗
ICONS = {"loading": "hourglass", "ready": "mic", "recording": "mic.fill", "transcribing": "waveform",
         "error": "exclamationmark.triangle"}
LABELS = {"loading": ("加载模型中", "Loading model"), "ready": ("就绪", "Ready"),
          "recording": ("录音中（再按快捷键结束，Esc 取消）", "Recording (press hotkey to stop, Esc to cancel)"),
          "transcribing": ("识别中", "Transcribing"), "error": ("失败", "Error")}

log = logging.getLogger("voice")
_carbon = ctypes.CDLL("/System/Library/Frameworks/Carbon.framework/Carbon")
_carbon.IsSecureEventInputEnabled.restype = ctypes.c_bool


def load_asr(path):
    """加载到 MPS 并预热；返回 (model, 加载秒数, 预热秒数)。不回退 CPU。"""
    import torch
    from qwen_asr import Qwen3ASRModel
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()  # 每次识别都打一行无关的生成参数警告，刷满日志
    t0 = time.time()
    model = Qwen3ASRModel.from_pretrained(str(path), dtype=torch.bfloat16, device_map="mps", max_new_tokens=1024)
    if str(model.device) != "mps:0":
        raise RuntimeError(f"模型落在 {model.device}，不是 MPS")
    t1 = time.time()
    model.transcribe((np.random.default_rng(0).normal(0, 0.05, SR * 2).astype(np.float32), SR))
    return model, t1 - t0, time.time() - t1


def transcribe_audio(model, audio, context):
    """长录音先在低音量处切段，再逐段识别拼起来。"""
    from qwen_asr.inference.utils import split_audio_into_chunks
    parts = split_audio_into_chunks(wav=audio, sr=SR, max_chunk_sec=ASR_CHUNK_SEC)
    texts = [model.transcribe((wav, SR), context=context)[0].text.strip() for wav, _ in parts]
    return join_parts(texts)


def join_parts(texts):
    """分段结果拼接：两边都是英文字母或数字时补一个空格，中文直接接。"""
    out = ""
    for t in filter(None, texts):
        if out and out[-1].isascii() and out[-1].isalnum() and t[0].isascii() and t[0].isalnum():
            out += " "
        out += t
    return out


def release_mps():
    """PyTorch 用完不把 GPU 缓存还给系统，长录音后会一阶一阶往上攒；识别完和卸载后各清一次。"""
    import gc
    import torch
    gc.collect()
    torch.mps.empty_cache()


def read_vocab():
    """专有名词表作为识别上下文，每次现读，改完不用重启。"""
    if not VOCAB_PATH.exists():
        return ""
    lines = (l.strip() for l in VOCAB_PATH.read_text().splitlines())
    return " ".join(l for l in lines if l and not l.startswith("#"))


# ---------- 热键 ----------
# 用键盘事件监听（CGEventTap）而不用 Carbon 热键：Carbon 不认 fn。
# 监听靠辅助功能权限；命中快捷键的按键会被吞掉，不再传给前台 App。

KEYCODES = {  # macOS 虚拟键码（ANSI 布局）
    "a": 0, "s": 1, "d": 2, "f": 3, "h": 4, "g": 5, "z": 6, "x": 7, "c": 8, "v": 9, "b": 11, "q": 12,
    "w": 13, "e": 14, "r": 15, "y": 16, "t": 17, "1": 18, "2": 19, "3": 20, "4": 21, "6": 22, "5": 23,
    "9": 25, "7": 26, "8": 28, "0": 29, "o": 31, "u": 32, "i": 34, "p": 35, "l": 37, "j": 38, "k": 40,
    "n": 45, "m": 46, "space": 49,
    "f1": 122, "f2": 120, "f3": 99, "f4": 118, "f5": 96, "f6": 97, "f7": 98, "f8": 100, "f9": 101,
    "f10": 109, "f11": 103, "f12": 111, "f13": 105, "f14": 107, "f15": 113, "f16": 106, "f17": 64,
    "f18": 79, "f19": 80, "f20": 90,
}
KEY_NAMES = {code: name for name, code in KEYCODES.items()}
FKEY_CODES = {code for name, code in KEYCODES.items() if name[0] == "f" and name[1:].isdigit()}
ESC = 53
KEYTAP_SRC = Path(__file__).resolve().parent / "keytap.c"
KEYTAP_LIB = ROOT / "out/libkeytap.dylib"
TAP_RECORD = struct.Struct("<IIQ")  # 和 keytap.c 的 Record 一致：事件类型、键码、修饰键标记
TAP_TOGGLE, TAP_CANCEL, TAP_CAPTURE, TAP_REENABLED = 1, 2, 3, 4
MOD_ORDER = [("fn", "fn ", kCGEventFlagMaskSecondaryFn), ("ctrl", "⌃", kCGEventFlagMaskControl),
             ("alt", "⌥", kCGEventFlagMaskAlternate), ("shift", "⇧", kCGEventFlagMaskShift),
             ("cmd", "⌘", kCGEventFlagMaskCommand)]
MOD_NAMES = {"fn": "fn", "ctrl": "ctrl", "alt": "alt", "option": "alt", "shift": "shift", "cmd": "cmd"}


def parse_hotkey(spec):
    """配置串 → (键码, 修饰键集合)。"""
    *mods, key = spec.lower().split("+")
    return KEYCODES[key], frozenset(MOD_NAMES[m] for m in mods)


def combo_from_event(keycode, flags):
    mods = {name for name, _, mask in MOD_ORDER if flags & mask}
    if keycode in FKEY_CODES:
        mods.discard("fn")  # 系统给 F 键自动带 fn 标记，分不清是不是真按了 fn
    return keycode, frozenset(mods)


def spec_from_combo(keycode, mods):
    """录到的组合转成配置串；不支持的键、或光按字母数字空格（没带修饰键）返回 None。"""
    key = KEY_NAMES.get(keycode)
    if key is None or (not mods and keycode not in FKEY_CODES):
        return None
    return "+".join([name for name, _, _ in MOD_ORDER if name in mods] + [key])


def hotkey_label(spec):
    _, mods = parse_hotkey(spec)
    key = spec.lower().split("+")[-1]
    return "".join(sym for name, sym, _ in MOD_ORDER if name in mods) + key.capitalize()


def load_setting(key, default):
    if SETTINGS_PATH.exists():
        return json.loads(SETTINGS_PATH.read_text()).get(key, default)
    return default


def load_hotkey():
    return load_setting("hotkey", DEFAULT_HOTKEY)


def save_setting(key, value):
    data = json.loads(SETTINGS_PATH.read_text()) if SETTINGS_PATH.exists() else {}
    data[key] = value
    tmp = SETTINGS_PATH.with_suffix(".tmp")  # 先写临时文件再替换：写到一半退出不会留下坏文件，下次照常启动
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(SETTINGS_PATH)


def save_hotkey(spec):
    save_setting("hotkey", spec)


def resolve_max_seconds(cli_value):
    return cli_value if cli_value is not None else float(load_setting("max_seconds", DEFAULT_MAX_SECONDS))


def mods_mask(mods):
    return sum(mask for name, _, mask in MOD_ORDER if name in mods)


def build_native(src, out, framework):
    """按需编译原生模块（源码比产物新才编），返回加载好的库。"""
    if not out.exists() or out.stat().st_mtime < src.stat().st_mtime:
        out.parent.mkdir(exist_ok=True)
        subprocess.run(["clang", "-O2", "-Wall", "-dynamiclib", "-framework", framework,
                        "-o", str(out), str(src)], check=True, capture_output=True)
    return ctypes.CDLL(str(out))


def load_keytap_lib():
    lib = build_native(KEYTAP_SRC, KEYTAP_LIB, "ApplicationServices")
    lib.kt_handle.argtypes = [ctypes.c_int, ctypes.c_int64, ctypes.c_uint64, ctypes.c_int]
    lib.kt_set_hotkey.argtypes = [ctypes.c_int64, ctypes.c_uint64]
    return lib


class KeyTap:
    """全局键盘监听。按键判断和吞键在 keytap.c 的原生线程里做，按键路径上不碰 Python；
    命中的事件写进管道，这里的读线程收到后调 on_event(kind, keycode, flags)。"""

    def __init__(self, on_event):
        self.on_event = on_event
        self.lib = None

    def start(self, hotkey):
        self.lib = load_keytap_lib()
        r, w = os.pipe()
        os.set_blocking(w, False)
        self.lib.kt_set_output(w)
        self.set_hotkey(hotkey)
        if self.lib.kt_start() != 0:
            raise RuntimeError(tr("开不了键盘监听，快捷键不可用：缺辅助功能权限",
                                  "Can't listen to the keyboard, hotkey disabled: Accessibility permission missing"))
        threading.Thread(target=self.read_events, args=(r,), daemon=True, name="keytap").start()

    def read_events(self, fd):
        buf = b""
        while True:
            try:
                buf += os.read(fd, TAP_RECORD.size * 64)
            except OSError:
                log.exception("键盘事件管道读失败，快捷键失效，需要重启")
                return
            whole = len(buf) - len(buf) % TAP_RECORD.size  # 万一读到半条，留到下次拼上，别让这个线程崩掉
            for kind, code, flags in TAP_RECORD.iter_unpack(buf[:whole]):
                try:
                    self.on_event(kind, code, flags)
                except Exception:
                    log.exception("处理键盘事件出错")
            buf = buf[whole:]

    def set_hotkey(self, hotkey):
        code, mods = hotkey
        self.lib.kt_set_hotkey(code, mods_mask(mods))

    def set_recording(self, on):
        if self.lib is not None:
            self.lib.kt_set_recording(int(on))

    def set_capturing(self, on):
        if self.lib is not None:
            self.lib.kt_set_capturing(int(on))


# ---------- 录音 ----------

AUDIOREC_SRC = Path(__file__).resolve().parent / "audiorec.c"
AUDIOREC_LIB = ROOT / "out/libaudiorec.dylib"


def load_audiorec_lib():
    lib = build_native(AUDIOREC_SRC, AUDIOREC_LIB, "AudioToolbox")
    lib.ar_start.argtypes = [ctypes.c_double]
    lib.ar_stop.restype = ctypes.c_size_t
    lib.ar_copy.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.ar_copy.restype = ctypes.c_size_t
    lib.ar_append.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.ar_level.restype = ctypes.c_float
    lib.ar_blocks.restype = ctypes.c_uint64
    return lib


class Recorder:
    """麦克风。采集在 audiorec.c 里；开麦关麦排进一个专用后台线程按顺序做，调用方拿 Future，主线程从不等。
    线程是 daemon：真卡死在系统调用里也不拦着进程退出。"""

    def __init__(self, lib=None):
        self.lib = lib or load_audiorec_lib()
        self.jobs = queue.Queue()
        threading.Thread(target=self.run_jobs, daemon=True, name="mic").start()

    def run_jobs(self):
        while True:
            fn, fut = self.jobs.get()
            try:
                fut.set_result(fn())
            except Exception as e:
                fut.set_exception(e)

    def submit(self, fn):
        fut = Future()
        self.jobs.put((fn, fut))
        return fut

    def start(self):
        def go():
            err = self.lib.ar_start(float(SR))
            if err:
                raise RuntimeError(tr(f"系统录音接口报错 {err}（检查麦克风权限或输入设备）",
                                      f"audio input error {err} (check microphone permission or input device)"))
        return self.submit(go)

    def stop(self):
        """关麦，Future 的结果是录到的音频（float32 一维）。"""
        def go():
            n = self.lib.ar_stop()
            audio = np.empty(n, np.float32)
            n = self.lib.ar_copy(audio.ctypes.data, n)
            self.lib.ar_reset()
            return audio[:n]
        return self.submit(go)

    def level(self):
        return self.lib.ar_level()


# ---------- 目标输入框 ----------

def ax_get(el, attr):
    err, val = AXUIElementCopyAttributeValue(el, attr, None)
    return val if err == 0 else None


def capture_target():
    app = NSWorkspace.sharedWorkspace().frontmostApplication()
    pid = app.processIdentifier()
    if pid == os.getpid():  # 前台是本进程（控制面板）：主线程上 AX 查自己会卡到超时，deliver 会直接拦下
        return {"pid": pid, "name": app.localizedName(), "ax_app": None, "focused": None, "window": None}
    t0 = time.time()
    ax_app = AXUIElementCreateApplication(pid)
    # Electron（Obsidian、Slack 等）默认不暴露 AX 树，打开它才能拿到焦点控件
    AXUIElementSetAttributeValue(ax_app, "AXManualAccessibility", True)
    focused = ax_get(ax_app, "AXFocusedUIElement")
    for _ in range(3):  # 刚打开 AX 树时 Electron 要几十毫秒才给出焦点；目标 App 卡着就别再等
        if focused is not None or time.time() - t0 > 0.3:
            break
        time.sleep(0.05)
        focused = ax_get(ax_app, "AXFocusedUIElement")
    window = ax_get(ax_app, "AXFocusedWindow") if focused is None else None
    return {"pid": pid, "name": app.localizedName(), "ax_app": ax_app, "focused": focused, "window": window}


def check_target(t):
    """返回 None 表示可以粘贴，否则返回不粘贴的原因。"""
    now = NSWorkspace.sharedWorkspace().frontmostApplication()
    if now.processIdentifier() != t["pid"]:
        return tr(f"前台已从「{t['name']}」切到「{now.localizedName()}」",
                  f"switched from {t['name']} to {now.localizedName()}")
    cur = ax_get(t["ax_app"], "AXFocusedUIElement")
    if cur is None:
        return tr(f"拿不到「{t['name']}」的输入焦点", f"can't read the focused field in {t['name']}")
    if t["focused"] is not None:
        if not CFEqual(cur, t["focused"]):
            return tr("输入焦点已变化", "focused field changed")
    else:  # 开录时没拿到焦点（Electron 刚打开 AX 树）：改用现在的焦点，但窗口必须还是开录时那个
        win = ax_get(t["ax_app"], "AXFocusedWindow")
        if t.get("window") is None or win is None or not CFEqual(win, t["window"]):
            return tr(f"拿不到「{t['name']}」的输入焦点", f"can't read the focused field in {t['name']}")
    if _carbon.IsSecureEventInputEnabled() or ax_get(cur, "AXSubrole") == "AXSecureTextField":
        return tr("目标是密码框", "target is a password field")
    role = ax_get(cur, "AXRole")
    err, settable = AXUIElementIsAttributeSettable(cur, "AXValue", None)
    if role not in TEXT_ROLES and not (err == 0 and settable):
        return tr(f"焦点不在可编辑区域（{role}）", f"focus is not an editable field ({role})")
    return None


# ---------- 剪贴板粘贴 ----------

def snapshot_pasteboard(pb):
    return [{str(t): it.dataForType_(t) for t in it.types()} for it in (pb.pasteboardItems() or [])]


def restore_pasteboard(pb, saved):
    pb.clearContents()
    items = []
    for d in saved:
        it = NSPasteboardItem.new()
        for t, data in d.items():
            if data is not None:
                it.setData_forType_(data, t)
        items.append(it)
    if items:
        pb.writeObjects_(items)


def send_cmd_v():
    for down in (True, False):
        e = CGEventCreateKeyboardEvent(None, 9, down)  # 9 = V
        CGEventSetFlags(e, kCGEventFlagMaskCommand)
        CGEventPost(kCGHIDEventTap, e)


def paste_text(text):
    pb = NSPasteboard.generalPasteboard()
    saved = snapshot_pasteboard(pb)
    pb.clearContents()
    pb.setString_forType_(text, NSPasteboardTypeString)
    pb.setString_forType_("", "org.nspasteboard.TransientType")  # 让剪贴板管理器忽略这次写入
    ours = pb.changeCount()
    send_cmd_v()

    def restore():
        if pb.changeCount() == ours:  # 期间用户又复制了东西就不覆盖
            restore_pasteboard(pb, saved)
        else:
            log.info("剪贴板在粘贴后被改动，跳过恢复")
    AppHelper.callLater(PASTE_RESTORE_DELAY, restore)


def copy_text(text):
    pb = NSPasteboard.generalPasteboard()
    pb.clearContents()
    pb.setString_forType_(text, NSPasteboardTypeString)


# ---------- 主控 ----------

class NoSound(Exception):
    """录到的音频太短或全是静音。"""


def guarded(fn):
    """主线程入口兜底：出了异常记日志、关麦、回到就绪，不让状态卡在录音或识别中。"""
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        try:
            return fn(self, *args, **kwargs)
        except Exception as e:
            log.exception("%s 出错", fn.__name__)
            self.recover(tr(f"出错已复位：{e}", f"Reset after an error: {e}"))
    return wrapper


class MenuTarget(NSObject):
    def copyLast_(self, _):
        if self.app.last_text:
            copy_text(self.app.last_text)

    def insertLast_(self, _):
        if self.app.last_text:
            paste_text(self.app.last_text)

    def cancel_(self, _):
        self.app.cancel()

    def changeHotkey_(self, _):
        self.app.change_hotkey()

    def editVocab_(self, _):
        NSWorkspace.sharedWorkspace().openFile_withApplication_(str(VOCAB_PATH), "TextEdit")

    def openLog_(self, _):
        NSWorkspace.sharedWorkspace().openFile_withApplication_(str(LOG_PATH), "Console")

    def openPanel_(self, _):
        self.app.panel.open()

    def reopen_(self, _):
        log.info("dock reopen")
        self.app.panel.open()  # 点 Dock 图标开控制面板；菜单栏下拉菜单仍从状态栏图标进

    def quit_(self, _):
        self.app.quit()


class VoiceInput:
    panel = None
    overlay = None
    stats = None  # 最近一次解析的日志统计，面板推送时复用

    def __init__(self, args):
        self.args = args
        self.panel = Panel(self)
        self.recorder = None  # start() 里建，原生模块编不出来时保持 None
        self.overlay = Overlay(lambda: self.recorder.level() if self.recorder is not None else 0.0)
        self.model = None
        self.model_ready = threading.Event()  # 置位＝没有加载在进行；加载或重载期间清掉
        self.idle_unload = float(load_setting("idle_unload_minutes", DEFAULT_IDLE_UNLOAD_MIN))
        self.cleanup = bool(load_setting("cleanup", True))  # 识别后删语气词和明显重复，见 cleanup.py
        self.digits = bool(load_setting("digits", True))    # 中文数字转阿拉伯数字，见 cn_numbers.py
        self.activity = 0  # 每回到就绪递增，让旧的闲置计时器作废
        self.state = None
        self.target = None
        self.session = 0  # 每段录音递增，复位时也递增：过期的计时器和晚到的识别结果靠它认出来
        self.mic_on = False  # 本段开麦成功
        self.rec_t0 = 0.0
        self.asr_lock = threading.Lock()  # 复位后旧识别可能还在跑，新一段排在它后面，不并发用模型
        self.last_text = ""
        self.hotkey_spec = load_hotkey()
        self.hotkey = parse_hotkey(self.hotkey_spec)
        self.capturing = None  # 录制新快捷键时是收集结果的列表
        self.keytap = KeyTap(self.on_tap_event)

    # 界面
    def build_menu(self):
        self.item = NSStatusBar.systemStatusBar().statusItemWithLength_(NSVariableStatusItemLength)
        self.menu_target = MenuTarget.new()
        self.menu_target.app = self
        menu = NSMenu.alloc().init()
        self.status_line = menu.addItemWithTitle_action_keyEquivalent_("", None, "")
        self.detail_line = menu.addItemWithTitle_action_keyEquivalent_("", None, "")
        menu.addItem_(NSMenuItem.separatorItem())
        self.menu_titles = []  # (菜单项, 中文, 英文)，切换语言时重设标题
        for entry in [("打开控制面板", "Open Control Panel", "openPanel:"), None,
                      ("复制上次结果", "Copy Last Result", "copyLast:"),
                      ("把上次结果插入当前输入框", "Insert Last Result Here", "insertLast:"),
                      ("取消本次录音", "Cancel Recording", "cancel:"), None,
                      ("更改快捷键…", "Change Hotkey…", "changeHotkey:"),
                      ("编辑专有名词表", "Edit Vocabulary", "editVocab:"), ("查看日志", "View Log", "openLog:"), None,
                      ("退出", "Quit", "quit:")]:
            if entry is None:
                menu.addItem_(NSMenuItem.separatorItem())
                continue
            zh, en, sel = entry
            item = menu.addItemWithTitle_action_keyEquivalent_(tr(zh, en), sel, "")
            item.setTarget_(self.menu_target)
            self.menu_titles.append((item, zh, en))
        self.item.setMenu_(menu)
        NSDistributedNotificationCenter.defaultCenter().addObserver_selector_name_object_(
            self.menu_target, "reopen:", REOPEN_NOTE, None)

    def set_state(self, state, detail=None):
        self.state = state
        self.keytap.set_recording(state == "recording")  # 放最前：后面界面更新出错，Esc 也不会被一直吞
        button = self.item.button()
        button.setImage_(NSImage.imageWithSystemSymbolName_accessibilityDescription_(ICONS[state], tr(*LABELS[state])))
        button.setContentTintColor_(NSColor.systemRedColor() if state == "recording" else None)
        self.status_line.setTitle_(f"{tr(*LABELS[state])} · {tr('快捷键', 'Hotkey')} {hotkey_label(self.hotkey_spec)}")
        if detail is not None:
            self.detail_line.setTitle_(detail)
        log.info("state=%s %s", state, detail or "")
        if self.overlay is not None and state in ("recording", "transcribing"):
            self.overlay.show(state)
        elif self.overlay is not None:
            self.overlay.hide()
        if state == "ready":
            self.schedule_unload()
        self.push_panel(fresh_stats=state == "ready")

    def push_panel(self, fresh_stats=False):
        if self.panel is not None and self.panel.is_open():
            self.panel.push(fresh_stats)

    def notify(self, detail, sound="Basso"):
        self.detail_line.setTitle_(detail)
        log.info("notice: %s", detail)
        if sound:
            NSSound.soundNamed_(sound).play()
        self.push_panel()

    # 控制面板用的查询/设置接口
    def panel_state(self, fresh_stats=False):
        if fresh_stats or self.stats is None:
            self.stats = parse_log_stats(LOG_PATH)
        if self.model is not None:
            model = "loaded"
        elif self.state == "loading" or not self.model_ready.is_set():
            model = "loading"
        else:
            model = "unloaded"
        return {"state": self.state, "detail": self.detail_line.title(), "hotkey": hotkey_label(self.hotkey_spec),
                "model": model, "stats": self.stats, "has_last": bool(self.last_text),
                "idle_unload_minutes": self.idle_unload, "max_seconds": self.args.max_seconds, "lang": i18n.LANG,
                "cleanup": self.cleanup, "digits": self.digits}

    def copy_last(self):
        if not self.last_text:
            return False
        copy_text(self.last_text)
        return True

    def save_panel_settings(self, idle, max_s):
        """返回 (是否成功, 提示)。当场生效：idle_unload 和 max_seconds 都是用到时才读。"""
        try:
            idle, max_s = float(idle), float(max_s)
        except (TypeError, ValueError):
            return False, tr("请填数字", "Please enter numbers")
        if not (0 <= idle <= 1440 and 10 <= max_s <= 600):
            return False, tr("闲置卸载 0–1440 分钟（0 为常驻），录音上限 10–600 秒",
                             "Idle unload 0–1440 min (0 = always loaded), max recording 10–600 s")
        idle, max_s = (int(x) if x == int(x) else x for x in (idle, max_s))  # 存 5 而不是 5.0
        save_setting("idle_unload_minutes", idle)
        save_setting("max_seconds", max_s)
        self.idle_unload, self.args.max_seconds = idle, max_s
        if self.state == "ready":  # 作废按旧分钟数排的计时器，按新值重排（0 则不再排）
            self.activity += 1
            self.schedule_unload()
        return True, tr("已保存，立即生效", "Saved, effective now")

    def read_vocab_text(self):
        return VOCAB_PATH.read_text(encoding="utf-8") if VOCAB_PATH.exists() else ""

    def save_vocab_text(self, text):
        if not isinstance(text, str) or len(text) > 100_000:
            return False, tr("词表内容无效或过长", "Vocabulary is invalid or too long")
        tmp = VOCAB_PATH.with_suffix(".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(VOCAB_PATH)
        return True, tr("词表已保存，下一次识别生效", "Vocabulary saved, used from the next transcription")

    def set_cleanup(self, on):
        self.cleanup = bool(on)
        save_setting("cleanup", self.cleanup)

    def set_digits(self, on):
        self.digits = bool(on)
        save_setting("digits", self.digits)

    def set_language(self, lang):
        if lang not in i18n.LANGS:
            return False
        save_setting("language", lang)
        i18n.set_lang(lang)
        for item, zh, en in self.menu_titles:
            item.setTitle_(tr(zh, en))
        self.panel.retitle()
        self.set_state(self.state, "")  # 刷新状态行；旧语言的详情行清掉
        return True

    # 生命周期
    def start(self):
        self.build_menu()
        self.set_state("loading", "")
        # AX 查询的全局超时：系统默认 6 秒，目标 App 忙时会把主线程整个卡住
        AXUIElementSetMessagingTimeout(AXUIElementCreateSystemWide(), AX_TIMEOUT)
        if not AXIsProcessTrustedWithOptions({kAXTrustedCheckOptionPrompt: True}):
            self.notify(tr("缺辅助功能权限：系统设置 → 隐私与安全性 → 辅助功能，勾上「语音输入」后从 Dock 重开（前台 run 时勾终端）",
                           "Accessibility permission missing: System Settings → Privacy & Security → Accessibility, "
                           "enable 语音输入, then reopen it from the Dock"))
        try:
            self.keytap.start(self.hotkey)
        except RuntimeError as e:
            self.notify(str(e))
        try:
            self.recorder = Recorder()
        except Exception as e:
            log.exception("recorder load failed")
            self.notify(tr(f"录音模块加载失败：{e}", f"Recorder failed to load: {e}"))
        threading.Thread(target=self.load_model, daemon=True).start()

    def load_model(self, reload=False):
        """reload=True 是闲置卸下后边录边重载：成败都不改状态，transcribe 等它结束再看 self.model。"""
        try:
            self.model, load_s, warm_s = load_asr(self.args.model)
            log.info("model %s %.1fs, warmup %.1fs", "reloaded" if reload else "loaded", load_s, warm_s)
            if not reload:
                AppHelper.callAfter(self.set_state, "ready", tr(f"模型就绪（加载 {load_s:.0f}s + 预热 {warm_s:.0f}s）",
                                                                f"Model ready (load {load_s:.0f}s + warm-up {warm_s:.0f}s)"))
        except Exception as e:
            log.exception("model load failed")
            if not reload:
                AppHelper.callAfter(self.set_state, "error", tr(f"模型加载失败：{e}", f"Model failed to load: {e}"))
        finally:
            self.model_ready.set()

    def schedule_unload(self):
        if self.idle_unload <= 0:
            return
        self.activity += 1
        n = self.activity
        AppHelper.callLater(self.idle_unload * 60, lambda: self.activity == n and self.unload_model())

    def unload_model(self):
        if self.state != "ready" or self.model is None or not self.model_ready.is_set():
            return
        self.model = None
        release_mps()
        log.info("model unloaded after %.0f min idle", self.idle_unload)
        self.detail_line.setTitle_(tr(f"闲置 {self.idle_unload:.0f} 分钟，已卸下模型；下次按快捷键边录边重载",
                                      f"Idle {self.idle_unload:.0f} min, model unloaded; it reloads while you record next time"))
        self.push_panel()

    def ensure_model(self):
        """卸下过就在后台重载，和录音并行；重载约 2 秒，通常说完话前就好了。"""
        if self.model is None and self.model_ready.is_set():
            self.model_ready.clear()
            threading.Thread(target=self.load_model, args=(True,), daemon=True).start()

    def change_hotkey(self):
        if self.state in ("recording", "transcribing"):
            self.notify(tr("录音或识别中，结束后再改快捷键", "Busy recording or transcribing; change the hotkey afterwards"))
            return
        alert = NSAlert.new()
        alert.setMessageText_(tr("按下新的快捷键组合", "Press the new key combination"))
        alert.setInformativeText_(tr(f"现在是 {hotkey_label(self.hotkey_spec)}。可用 fn ⌃ ⌥ ⇧ ⌘ 任意组合，"
                                     "再加字母、数字或空格；也可以单按 F 键（F 键不能配 fn）。Esc 取消。",
                                     f"Current: {hotkey_label(self.hotkey_spec)}. Combine any of fn ⌃ ⌥ ⇧ ⌘ with a letter, "
                                     "digit or Space, or use an F key alone (F keys can't take fn). Esc cancels."))
        alert.addButtonWithTitle_(tr("取消", "Cancel")).setKeyEquivalent_("\x1b")
        self.capturing = []  # 录制期间键盘监听只收组合，不触发录音
        self.keytap.set_capturing(True)
        NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
        alert.runModal()
        self.keytap.set_capturing(False)
        captured, self.capturing = self.capturing, None
        if not captured:
            return
        self.hotkey_spec, self.hotkey = captured[0], parse_hotkey(captured[0])
        self.keytap.set_hotkey(self.hotkey)
        save_hotkey(captured[0])
        self.set_state(self.state, tr(f"快捷键已改为 {hotkey_label(captured[0])}", f"Hotkey changed to {hotkey_label(captured[0])}"))

    def on_tap_event(self, kind, keycode, flags):
        """在 KeyTap 读线程里跑，吞不吞键原生侧已经定了；动作 callAfter 回主线程。"""
        if kind == TAP_TOGGLE:
            AppHelper.callAfter(self.on_toggle)
        elif kind == TAP_CANCEL:
            AppHelper.callAfter(self.cancel)
        elif kind == TAP_CAPTURE and self.capturing is not None and not self.capturing:
            spec = spec_from_combo(*combo_from_event(keycode, flags))
            if spec is not None:  # 不支持的键（方向键等）原生侧也吞了，这里忽略，接着等
                self.capturing.append(spec)
                NSApplication.sharedApplication().abortModal()  # 文档允许从其他线程调用，stopModal 不行
        elif kind == TAP_REENABLED:  # flags 里是停用原因
            log.info("键盘监听被系统停用（%s），已重新打开",
                     "回调超时" if flags == kCGEventTapDisabledByTimeout else "安全输入或权限变动")

    def quit(self):
        if self.recorder is not None:
            self.recorder.stop()  # 排进录音线程就走，不等
        PID_PATH.unlink(missing_ok=True)
        log.info("quit")
        AppHelper.stopEventLoop()

    # 录音
    def recover(self, detail):
        """录音或识别中出了岔子：作废这一段、关麦、回到就绪。"""
        if self.state not in ("recording", "transcribing"):
            return
        self.session += 1
        if self.recorder is not None:
            self.recorder.stop()
        try:
            self.set_state("ready", detail)
            NSSound.soundNamed_("Basso").play()
        except Exception:
            log.exception("复位时出错")

    @guarded
    def on_toggle(self):
        if self.state == "ready":
            self.start_recording()
        elif self.state == "recording":
            self.finish_recording()
        elif self.state == "transcribing":
            self.notify(tr("还在识别上一段，稍等", "Still transcribing the last clip, please wait"))
        elif self.state == "error" and self.model is None and self.model_ready.is_set():
            self.model_ready.clear()  # 上次加载失败：按快捷键就重试，不用重启
            self.set_state("loading", tr("重新加载模型", "Reloading model"))
            threading.Thread(target=self.load_model, daemon=True).start()
        else:
            self.notify(tr(*LABELS[self.state]))

    def start_recording(self):
        if self.recorder is None:
            self.notify(tr("录音模块没加载，详情看日志", "Recorder failed to load, see log"))
            return
        self.target = capture_target()
        NSSound.soundNamed_("Tink").play()  # 先响再开麦，免得提示音录进去
        self.session += 1
        session = self.session
        self.mic_on, self.rec_t0 = False, time.time()
        # 状态先切到录音，Esc 和再按快捷键立刻可用；开麦在录音线程里做，主线程不等
        self.set_state("recording", tr(f"目标：{self.target['name']}", f"Target: {self.target['name']}"))
        self.recorder.start().add_done_callback(
            lambda f: AppHelper.callAfter(self.on_mic_started, session, f.exception()))
        AppHelper.callLater(MIC_TIMEOUT, lambda: self.check_mic_started(session))
        self.ensure_model()
        AppHelper.callLater(self.args.max_seconds, lambda: self.session == session and self.state == "recording"
                            and self.finish_recording(timed_out=True))

    @guarded
    def on_mic_started(self, session, err):
        if session != self.session or self.state != "recording":
            return  # 这段已经结束或取消；关麦请求排在开麦后面，会照常执行
        if err is not None:
            log.error("mic open failed: %s", err)
            self.recover(tr(f"打不开麦克风：{err}", f"Can't open the microphone: {err}"))
            return
        self.mic_on = True

    @guarded
    def check_mic_started(self, session):
        if session == self.session and self.state == "recording" and not self.mic_on:
            log.error("mic not open after %.0fs", MIC_TIMEOUT)
            self.recover(tr(f"麦克风 {MIC_TIMEOUT:.0f} 秒没打开，这段作废；连续出现就从菜单退出再重开",
                            f"Microphone didn't open within {MIC_TIMEOUT:.0f}s; quit and reopen if it keeps happening"))

    @guarded
    def cancel(self):
        if self.state != "recording":
            return
        self.session += 1
        self.recorder.stop()
        self.set_state("ready", tr("已取消，未插入", "Cancelled, nothing inserted"))
        NSSound.soundNamed_("Pop").play()

    @guarded
    def finish_recording(self, timed_out=False):
        t_stop = time.time()
        wall = t_stop - self.rec_t0
        session = self.session
        audio_fut = self.recorder.stop()
        NSSound.soundNamed_("Pop").play()
        self.set_state("transcribing", tr(f"{wall:.0f} 秒录音", f"{wall:.0f}s of audio")
                       + (tr("（已到时长上限自动结束）", " (stopped at the time limit)") if timed_out else ""))
        threading.Thread(target=self.transcribe, args=(session, audio_fut, wall, t_stop), daemon=True).start()
        # 看门狗：正常识别约为录音时长的 7%，给足余量；超了就先把界面复位，别让人以为卡死
        budget = 20 + 0.3 * wall + (0 if self.model is not None else MODEL_WAIT_TIMEOUT)
        AppHelper.callLater(budget, lambda: self.check_transcribe(session, budget))

    @guarded
    def check_transcribe(self, session, budget):
        if session == self.session and self.state == "transcribing":
            log.error("transcribe watchdog fired after %.0fs", budget)
            self.recover(tr(f"识别 {budget:.0f} 秒没出结果，先复位；结果晚到会放进「复制上次结果」。连续出现就从菜单退出再重开",
                            f"No result after {budget:.0f}s, reset; a late result goes to Copy Last Result. "
                            "Quit and reopen if it keeps happening"))

    def transcribe(self, session, audio_fut, wall, t_stop):
        """后台线程：等关麦拿到音频，再等模型，然后识别。每一步都有上限，结果交给 deliver。"""
        text, err, secs = "", None, 0.0
        try:
            try:
                audio = audio_fut.result(timeout=MIC_TIMEOUT)
            except TimeoutError:
                raise RuntimeError(tr(f"麦克风 {MIC_TIMEOUT:.0f} 秒没关上，这段录音丢了；从菜单退出再重开",
                                      f"microphone didn't stop within {MIC_TIMEOUT:.0f}s, clip lost; quit and reopen"))
            secs = len(audio) / SR
            if secs < 0.3 or np.abs(audio).max() < SILENCE_PEAK:
                raise NoSound()
            if wall - secs > 2:  # 麦克风中途断流（多半是蓝牙耳机切换），录到的比按住的时间短
                log.warning("mic gap wall=%.1fs audio=%.1fs", wall, secs)
            if not self.model_ready.wait(MODEL_WAIT_TIMEOUT):
                raise RuntimeError(tr(f"模型 {MODEL_WAIT_TIMEOUT} 秒没加载好，从菜单退出再重开",
                                      f"model not loaded after {MODEL_WAIT_TIMEOUT}s, quit and reopen"))
            model = self.model  # 拿住引用：识别途中就算被复位、闲置卸载，这一段也照样跑完
            if model is None:
                raise RuntimeError(tr("模型重载失败，详情看日志", "model reload failed, see log"))
            with self.asr_lock:
                text = transcribe_audio(model, audio, read_vocab())
            if self.cleanup:
                raw_len, text = len(text), clean(text)
                log.info("cleanup removed=%d chars", raw_len - len(text))  # 只记数量，不记内容
            if self.digits:
                text = to_digits(text)
        except NoSound as e:
            err = e
        except Exception as e:
            log.exception("transcribe failed")
            text, err = "", e
        try:
            release_mps()
        except Exception:
            log.exception("release_mps failed")
        AppHelper.callAfter(self.deliver, text, err, secs, t_stop, time.time(), session, wall)

    @guarded
    def deliver(self, text, err, secs, t_stop, t_asr, session=None, wall=None):
        if session is not None and (session != self.session or self.state != "transcribing"):
            log.info("deliver late chars=%d", len(text))  # 看门狗已经复位过
            if text:
                self.last_text = text
                self.notify(tr(f"上一段晚到的 {len(text)} 字没有插入，在菜单「复制上次结果」",
                               f"A late result ({len(text)} chars) was not inserted; use Copy Last Result"),
                            sound=None if self.state == "recording" else "Basso")  # 正在录新的一段，别把提示音录进去
            return
        if isinstance(err, NoSound):
            self.set_state("ready", tr("没录到声音（检查麦克风权限或输入设备）",
                                       "No sound captured (check microphone permission or input device)"))
            NSSound.soundNamed_("Basso").play()
            return
        if err is not None:
            self.set_state("ready", tr(f"识别失败：{err}", f"Transcription failed: {err}"))
            NSSound.soundNamed_("Basso").play()
            return
        if not text:
            self.set_state("ready", tr("没识别出文字", "No text recognized"))
            return
        gap = tr(f"；录音中途断了约 {wall - secs:.0f} 秒", f"; mic dropped ~{wall - secs:.0f}s") \
            if wall is not None and wall - secs > 2 else ""
        self.last_text = text
        reason = tr("目标是控制面板", "target is the control panel") if self.target["pid"] == os.getpid() else check_target(self.target)
        if reason:
            self.set_state("ready", tr(f"未插入：{reason}。结果在菜单「复制上次结果」",
                                       f"Not inserted: {reason}. Use Copy Last Result") + gap)
            NSSound.soundNamed_("Basso").play()
            log.info("deliver skipped audio=%.1fs asr=%.2fs chars=%d reason=%s", secs, t_asr - t_stop, len(text), reason)
            return
        paste_text(text)
        t_done = time.time()
        log.info("deliver ok app=%s audio=%.1fs asr=%.2fs stop_to_paste=%.2fs chars=%d",
                 self.target["name"], secs, t_asr - t_stop, t_done - t_stop, len(text))
        self.set_state("ready", tr(f"已插入 {len(text)} 字（{t_done - t_stop:.1f}s）",
                                   f"Inserted {len(text)} chars ({t_done - t_stop:.1f}s)") + gap)


def main():
    p = argparse.ArgumentParser(description="Qwen3-ASR 本地语音输入")
    p.add_argument("--model", type=Path, default=Path(os.environ.get("QWEN_ASR_MODEL", DEFAULT_MODEL)))
    p.add_argument("--max-seconds", type=float, default=None,
                   help="单次录音上限（秒），到时自动结束并识别；优先于 settings.json，都没有则 120")
    args = p.parse_args()
    args.max_seconds = resolve_max_seconds(args.max_seconds)
    if not (args.model / "model.safetensors").exists():
        p.error(f"模型目录不对：{args.model}")
    if PID_PATH.exists():
        try:
            pid = int(PID_PATH.read_text())
            cmd = subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True).stdout
            if "voice_input.py" in cmd:  # pid 可能被别的进程复用，只认本程序
                raise SystemExit(f"已经在运行（pid {pid}），先退出旧的")
        except ValueError:
            pass
    PID_PATH.write_text(str(os.getpid()))
    i18n.set_lang(load_setting("language", "zh"))
    logging.basicConfig(filename=LOG_PATH, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    NSBundle.mainBundle().infoDictionary()["CFBundleName"] = "语音输入"  # 面板在前台时菜单栏显示这个名字，而不是 python
    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)  # 无 Dock 图标，不抢焦点
    # 关掉 App Nap：后台无窗口的进程会被系统降速，键盘监听线程跟着被拖，系统判超时停监听
    activity = NSProcessInfo.processInfo().beginActivityWithOptions_reason_(  # 持有引用才生效，跟进程同寿
        NSActivityUserInitiatedAllowingIdleSystemSleep, "global hotkey listener")
    voice = VoiceInput(args)
    voice.start()
    log.info("started pid=%d model=%s hotkey=%s", os.getpid(), args.model, voice.hotkey_spec)
    AppHelper.runEventLoop(installInterrupt=True, unexpectedErrorAlert=log_unexpected)


def log_unexpected():
    """事件循环里漏出来的异常：记日志接着跑。默认会弹一个模态框，本程序没有前台窗口，框藏在后面，看着就像卡死。"""
    log.error("事件循环未处理的异常", exc_info=True)
    return True


if __name__ == "__main__":
    main()
