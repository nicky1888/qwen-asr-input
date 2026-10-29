import os
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import voice_input as vi
from AppKit import NSPasteboard, NSPasteboardTypeString, NSPasteboardTypePNG
from Foundation import NSData
from Quartz import kCGEventFlagMaskAlternate as ALT, kCGEventFlagMaskSecondaryFn as FN, kCGEventFlagMaskShift as SHIFT


def test_parse_hotkey():
    assert vi.parse_hotkey("alt+shift+space") == (49, frozenset({"alt", "shift"}))
    assert vi.parse_hotkey("fn+space") == (49, frozenset({"fn"}))
    assert vi.parse_hotkey("option+ctrl+v") == (9, frozenset({"alt", "ctrl"}))
    assert vi.parse_hotkey("f13") == (105, frozenset())
    with pytest.raises(KeyError):
        vi.parse_hotkey("hyper+space")


def test_pasteboard_roundtrip_keeps_text_and_image():
    pb = NSPasteboard.pasteboardWithUniqueName()
    png = NSData.dataWithBytes_length_(b"\x89PNG fake", 9)
    pb.clearContents()
    pb.setString_forType_("原内容", NSPasteboardTypeString)
    pb.setData_forType_(png, NSPasteboardTypePNG)
    saved = vi.snapshot_pasteboard(pb)
    pb.clearContents()
    pb.setString_forType_("识别结果", NSPasteboardTypeString)
    vi.restore_pasteboard(pb, saved)
    assert pb.stringForType_(NSPasteboardTypeString) == "原内容"
    assert pb.dataForType_(NSPasteboardTypePNG) == png
    pb.releaseGlobally()


def test_read_vocab_skips_comments(tmp_path, monkeypatch):
    f = tmp_path / "vocab.txt"
    f.write_text("# 注释\nGitHub\n\n  Claude Code \n")
    monkeypatch.setattr(vi, "VOCAB_PATH", f)
    assert vi.read_vocab() == "GitHub Claude Code"
    monkeypatch.setattr(vi, "VOCAB_PATH", tmp_path / "missing.txt")
    assert vi.read_vocab() == ""


def test_hotkey_label():
    assert vi.hotkey_label("alt+shift+space") == "⌥⇧Space"
    assert vi.hotkey_label("cmd+ctrl+v") == "⌃⌘V"
    assert vi.hotkey_label("shift+fn+space") == "fn ⇧Space"


def test_event_to_spec_roundtrip():
    # 录到什么组合，就按同一套规则匹配：录制结果解析回来要等于监听算出的组合
    for code, flags, spec in [(49, FN, "fn+space"), (49, ALT | SHIFT, "alt+shift+space"),
                              (9, FN | SHIFT, "fn+shift+v"), (105, FN, "f13")]:
        combo = vi.combo_from_event(code, flags)
        assert vi.spec_from_combo(*combo) == spec
        assert vi.parse_hotkey(spec) == combo


def test_spec_rejects_bare_keys():
    assert vi.spec_from_combo(*vi.combo_from_event(9, 0)) is None      # 光按 V
    assert vi.spec_from_combo(*vi.combo_from_event(49, 0)) is None     # 光按空格
    assert vi.spec_from_combo(*vi.combo_from_event(vi.ESC, 0)) is None  # Esc 留给取消


def tap_events(r):
    os.set_blocking(r, False)
    try:
        return [k for k, _, _ in vi.TAP_RECORD.iter_unpack(os.read(r, 4096))]
    except BlockingIOError:
        return []


def test_native_keytap_swallow_rules():
    # 原生回调的判断逻辑，不开真监听：返回 1 表示吞键
    lib = vi.load_keytap_lib()
    r, w = os.pipe()
    lib.kt_set_output(w)
    lib.kt_set_hotkey(49, vi.mods_mask({"fn"}))
    lib.kt_set_recording(0)
    lib.kt_set_capturing(0)
    assert lib.kt_handle(1, 49, FN, 0) == 1 and tap_events(r) == [vi.TAP_TOGGLE]
    assert lib.kt_handle(1, 49, FN, 1) == 1 and tap_events(r) == []   # 按住连发不重复触发，但照样吞掉
    assert lib.kt_handle(0, 49, FN, 0) == 1                                # 抬起跟着吞
    assert lib.kt_handle(0, 49, FN, 0) == 0                                # 只吞一次
    assert lib.kt_handle(1, 49, 0, 0) == 0                                 # 光按空格放行
    assert lib.kt_handle(1, 49, FN | SHIFT, 0) == 0                        # 多一个修饰键不算
    assert lib.kt_handle(1, vi.ESC, 0, 0) == 0                             # 没在录音，Esc 放行
    lib.kt_set_recording(1)
    assert lib.kt_handle(1, vi.ESC, 0, 0) == 1 and tap_events(r) == [vi.TAP_CANCEL]
    lib.kt_set_hotkey(105, 0)                                              # 改成 F13
    assert lib.kt_handle(1, 105, FN, 0) == 1                               # 系统给 F 键带的 fn 不影响匹配
    tap_events(r)
    lib.kt_set_capturing(1)
    assert lib.kt_handle(1, vi.ESC, 0, 0) == 0                             # 录制时 Esc 交给对话框
    assert lib.kt_handle(1, 9, 0, 0) == 0                                  # 光按字母放行
    assert lib.kt_handle(1, 105, FN, 0) == 1                               # 录制时不触发录音，交给录制
    assert lib.kt_handle(1, 9, FN | SHIFT, 0) == 1 and tap_events(r) == [vi.TAP_CAPTURE, vi.TAP_CAPTURE]
    lib.kt_set_capturing(0)
    os.close(r)
    os.close(w)


def test_on_tap_event_dispatch(monkeypatch):
    calls = []
    monkeypatch.setattr(vi.AppHelper, "callAfter", lambda f, *a: calls.append(f.__name__))
    app = type("A", (), {"abortModal": lambda s: calls.append("abort")})()
    monkeypatch.setattr(vi, "NSApplication", type("NSApp", (), {"sharedApplication": staticmethod(lambda: app)}))
    v = vi.VoiceInput.__new__(vi.VoiceInput)
    v.capturing = None
    v.on_tap_event(vi.TAP_TOGGLE, 49, FN)
    v.on_tap_event(vi.TAP_CANCEL, vi.ESC, 0)
    v.on_tap_event(vi.TAP_CAPTURE, 9, FN | SHIFT)              # 没在录制，忽略
    assert calls == ["on_toggle", "cancel"]
    v.capturing = []
    v.on_tap_event(vi.TAP_CAPTURE, 123, FN)                     # 方向键不支持，接着等
    assert v.capturing == []
    v.on_tap_event(vi.TAP_CAPTURE, 9, FN | SHIFT)
    v.on_tap_event(vi.TAP_CAPTURE, 0, FN)                       # 只收第一个
    assert v.capturing == ["fn+shift+v"] and calls[-1] == "abort"


def test_overlay_level_mapping():
    from overlay import elapsed_label, level_to_height
    assert level_to_height(0.0) == 0.0 and level_to_height(1.0) == 1.0
    assert 0.3 < level_to_height(0.05) < 0.8                   # 正常说话落在中段
    assert elapsed_label(7.9) == "0:07" and elapsed_label(125) == "2:05"


def test_save_and_load_hotkey(tmp_path, monkeypatch):
    monkeypatch.setattr(vi, "SETTINGS_PATH", tmp_path / "settings.json")
    assert vi.load_hotkey() == vi.DEFAULT_HOTKEY
    vi.save_hotkey("ctrl+alt+v")
    assert vi.load_hotkey() == "ctrl+alt+v"


def test_idle_unload_and_reload(monkeypatch):
    timers, released = [], []
    monkeypatch.setattr(vi.AppHelper, "callLater", lambda delay, f: timers.append((delay, f)))
    monkeypatch.setattr(vi, "release_mps", lambda: released.append(1))
    monkeypatch.setattr(vi, "load_asr", lambda path: ("model2", 0.0, 0.0))

    class Line:
        def setTitle_(self, t): self.title = t
    v = vi.VoiceInput.__new__(vi.VoiceInput)
    v.args = type("A", (), {"model": None})()
    v.model, v.model_ready, v.idle_unload, v.activity = "model1", vi.threading.Event(), 10.0, 0
    v.model_ready.set()
    v.detail_line, v.state = Line(), "ready"

    v.schedule_unload()
    v.schedule_unload()                      # 又回到一次就绪：前一个计时器作废
    assert [d for d, _ in timers] == [600, 600]
    timers[0][1]()
    assert v.model == "model1" and not released
    v.state = "recording"
    timers[1][1]()                           # 录音中到点也不卸
    assert v.model == "model1"
    v.state = "ready"
    timers[1][1]()
    assert v.model is None and released and v.model_ready.is_set()  # 卸下≠在加载

    v.ensure_model()                         # 卸下后开录：后台重载，transcribe 等 model_ready
    assert v.model_ready.wait(5) and v.model == "model2"
    v.ensure_model()                         # 已有模型时不重复加载
    assert v.model == "model2"

    v.idle_unload, timers[:] = 0, []
    v.schedule_unload()                      # 0 表示常驻，不排计时器
    assert not timers


# ---------- 控制面板 ----------

def test_parse_log_stats_today_vs_total(tmp_path):
    import stats
    log = tmp_path / "v.log"
    log.write_text(
        "2026-10-27 09:00:00,100 INFO deliver ok app=Brave Browser audio=10.0s asr=1.00s stop_to_paste=1.10s chars=40\n"
        "2026-10-28 09:00:00,100 INFO deliver ok app=Obsidian audio=6.5s asr=0.90s stop_to_paste=1.00s chars=80\n"
        "2026-10-28 09:05:00,100 INFO deliver skipped audio=3.0s asr=0.50s chars=99 reason=x\n"  # 没插入的不算
        "2026-10-28 09:06:00,100 INFO state=ready 已插入 80 字\n"
        "2026-10-28 09:10:00,100 INFO deliver ok app=Notes audio=3.5s asr=0.50s stop_to_paste=0.60s chars=40\n"
        "垃圾行\n")
    s = stats.parse_log_stats(log, today="2026-10-28")
    assert s["today"] == {"count": 2, "chars": 120, "audio_s": 10.0, "saved_min": 2.8}
    assert s["total"] == {"count": 3, "chars": 160, "audio_s": 20.0, "saved_min": 3.7}
    assert stats.parse_log_stats(tmp_path / "missing.log")["total"]["count"] == 0


def test_phys_footprint_matches_footprint_cmd():
    import re
    import subprocess
    import panel
    mine = panel.phys_footprint_mb()
    assert mine and mine > 0
    out = subprocess.run(["footprint", "-p", str(os.getpid())], capture_output=True, text=True, stdin=subprocess.DEVNULL).stdout
    num, unit = re.search(r"Footprint: ([\d.]+) (KB|MB|GB)", out).groups()
    ref = float(num) * {"KB": 1 / 1024, "MB": 1, "GB": 1024}[unit]
    assert abs(mine - ref) / ref < 0.2


def test_deliver_blocks_paste_into_own_panel(monkeypatch):
    states, pasted = [], []
    monkeypatch.setattr(vi, "paste_text", lambda t: pasted.append(t))
    monkeypatch.setattr(vi, "check_target", lambda t: pytest.fail("目标是本进程时不该走 AX 检查"))
    monkeypatch.setattr(vi, "NSSound", type("S", (), {"soundNamed_": staticmethod(lambda n: type("P", (), {"play": lambda s: None})())}))
    v = vi.VoiceInput.__new__(vi.VoiceInput)
    v.target = {"pid": os.getpid(), "name": "语音输入"}
    v.set_state = lambda state, detail=None: states.append((state, detail))
    v.deliver("你好", None, 1.0, 0.0, 0.5)
    assert not pasted and v.last_text == "你好"           # 不粘，结果留给「复制上次结果」
    assert "目标是控制面板" in states[-1][1]


def test_max_seconds_precedence(tmp_path, monkeypatch):
    monkeypatch.setattr(vi, "SETTINGS_PATH", tmp_path / "settings.json")
    assert vi.resolve_max_seconds(None) == 120                  # 都没有 → 默认
    vi.save_setting("max_seconds", 45)
    assert vi.resolve_max_seconds(None) == 45                   # settings 优于默认
    assert vi.resolve_max_seconds(30.0) == 30.0                 # 命令行显式传最优先


def test_save_panel_settings_keeps_other_keys(tmp_path, monkeypatch):
    f = tmp_path / "settings.json"
    f.write_text('{"hotkey": "fn+a", "idle_unload_minutes": 10}')
    monkeypatch.setattr(vi, "SETTINGS_PATH", f)
    monkeypatch.setattr(vi.AppHelper, "callLater", lambda *a: None)
    v = vi.VoiceInput.__new__(vi.VoiceInput)
    v.args, v.state, v.activity, v.idle_unload = type("A", (), {"max_seconds": 120})(), "loading", 0, 10.0
    assert v.save_panel_settings("5", "90") == (True, "已保存，立即生效")
    assert vi.load_setting("hotkey", None) == "fn+a" and vi.load_setting("idle_unload_minutes", None) == 5
    assert vi.load_setting("max_seconds", None) == 90 and v.args.max_seconds == 90 and v.idle_unload == 5
    assert not v.save_panel_settings("abc", "90")[0] and not v.save_panel_settings("5", "5")[0]
    assert vi.load_setting("idle_unload_minutes", None) == 5     # 非法输入不落盘
    assert '"idle_unload_minutes": 5,' in f.read_text()          # 整数存成 5，不是 5.0


def test_panel_routes_messages_without_window(monkeypatch):
    import panel
    calls = []

    class FakeApp:
        def copy_last(self): calls.append("copy"); return True
        def read_vocab_text(self): return "词A"
        def save_vocab_text(self, t): calls.append(("vocab", t)); return True, "ok"
    p = panel.Panel(FakeApp())
    p.call_js = lambda fn, arg: calls.append((fn, arg))
    p.on_message('{"type": "copy_last"}')
    p.on_message('{"type": "get_vocab"}')
    p.on_message('{"type": "save_vocab", "text": "x"}')
    p.on_message("不是 JSON")
    p.on_message('{"type": "nope"}')
    assert calls == ["copy", ("toast", {"ok": True, "kind": "copy", "msg": "已复制上次结果"}), ("setVocab", "词A"),
                     ("vocab", "x"), ("toast", {"ok": True, "kind": "vocab", "msg": "ok"})]


def test_set_language_switches_menu_and_messages(tmp_path, monkeypatch):
    import i18n
    f = tmp_path / "settings.json"
    f.write_text('{"hotkey": "fn+a", "idle_unload_minutes": 5}')
    monkeypatch.setattr(vi, "SETTINGS_PATH", f)
    monkeypatch.setattr(i18n, "LANG", "zh")

    class Item:
        def __init__(self): self.title = None
        def setTitle_(self, t): self.title = t
    states = []
    v = vi.VoiceInput.__new__(vi.VoiceInput)
    v.menu_titles = [(Item(), "退出", "Quit")]
    v.panel = type("P", (), {"retitle": lambda self: None})()
    v.state = "ready"
    v.set_state = lambda state, detail=None: states.append((state, detail))

    assert not v.set_language("fr") and vi.load_setting("language", None) is None   # 不认识的语言不落盘
    assert v.set_language("en")
    assert vi.load_setting("language", None) == "en" and vi.load_setting("hotkey", None) == "fn+a"
    assert v.menu_titles[0][0].title == "Quit" and states == [("ready", "")]
    assert vi.tr("已保存", "Saved") == "Saved"
    v.set_language("zh")
    assert v.menu_titles[0][0].title == "退出" and vi.tr("已保存", "Saved") == "已保存"


def test_cleanup_removes_fillers_and_repeats_only():
    from cleanup import clean
    # 来自 Qwen3-ASR 对带语气词合成语音的真实输出
    assert clean("嗯，那个，我觉得，我觉得这个方案呃，还是可以的啊。就是就是，我们下周再再看一下数据。") \
        == "我觉得这个方案还是可以的啊。我们下周再再看一下数据。"
    assert clean("你帮我嗯，把这个文件呃发给他，就是那个那个周报。对，周报。") == "你帮我把这个文件发给他，就是那个周报。对，周报。"
    assert clean("呃，今天的话，嗯，那个回测的结果，然后，然后，收益率大概百分之十二吧，啊，还行。") \
        == "今天的话，那个回测的结果，然后，收益率大概百分之十二吧，还行。"
    # 不该动的：语气助词、单字叠词、含「额」的词、作指代的「那个」
    for keep in ["好啊，谢谢，看看那个周报。", "额度不够了。", "这个事情就是这样。"]:
        assert clean(keep) == keep
    assert clean("Um, I think, uh, we should ship it.") == "I think, we should ship it."
    assert clean("嗯。") == ""


def test_to_digits_converts_values_keeps_idioms():
    from cn_numbers import to_digits
    # 来自 Qwen3-ASR 对合成语音的真实输出
    assert to_digits("收益率大概是百分之十二点五，回撤控制在百分之三以内。") == "收益率大概是12.5%，回撤控制在3%以内。"
    assert to_digits("这个月花了一千二百三十美元，比上个月多了三百块。") == "这个月花了1230美元，比上个月多了300块。"
    assert to_digits("会议定在十月二十八号下午三点半，大概开一个小时。") == "会议定在10月28号下午3点半，大概开一个小时。"
    assert to_digits("第一季度卖了两万五千台，单价是三点五万。") == "第一季度卖了25000台，单价是3.5万。"
    for s, want in [("二零二六年", "2026年"), ("两万五", "25000"), ("第二十五名", "第25名"), ("十二万", "12万"),
                    ("三点十五分", "3点15分"), ("他三十岁了", "他30岁了"), ("百分之百", "100%"), ("十月一日", "10月1日")]:
        assert to_digits(s) == want, s
    for keep in ["我们一起看一下这个问题，一定要解决。一般来说，十分重要。", "一心一意，万一不行，唯一的办法，统一一下。",
                 "我有两个想法，一个是这样。", "七八个人", "五六十个", "一点点", "千万别忘了", "一刀切", "我们一块儿去",
                 "一日三餐", "两三天", "几十万", "第一"]:
        assert to_digits(keep) == keep, keep
