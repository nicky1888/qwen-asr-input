"""控制面板：NSWindow 里放一个 WKWebView，加载本地 panel.html，和主程序同进程、不开端口。

页面 → Python：postMessage(JSON 字符串)，{"type": ...}，见 Panel.on_message。
Python → 页面：evaluateJavaScript 调 window.panel.update / setMem / setVocab / toast。
面板只在用户主动打开时出现；关窗即释放窗口和 WebView，下次打开重建，关窗不退出 App。
"""
import ctypes
import json
import logging
import os
from pathlib import Path

import objc
from AppKit import (
    NSApplication, NSBackingStoreBuffered, NSColor, NSMenu, NSMenuItem, NSWindow,
    NSWindowStyleMaskClosable, NSWindowStyleMaskMiniaturizable, NSWindowStyleMaskResizable,
    NSWindowStyleMaskTitled,
)
from Foundation import NSObject, NSThread, NSTimer
from PyObjCTools import AppHelper
from WebKit import WKUserContentController, WKWebView, WKWebViewConfiguration

from i18n import tr

HTML_PATH = Path(__file__).resolve().parent / "panel.html"
HANDLER_NAME = "panel"
MEM_REFRESH_SEC = 5
log = logging.getLogger("voice")

_libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
RUSAGE_INFO_V2 = 2


class RusageInfoV2(ctypes.Structure):
    """<sys/resource.h> 的 rusage_info_v2，字段顺序必须和系统头文件一致。"""
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [(n, ctypes.c_uint64) for n in (
        "ri_user_time", "ri_system_time", "ri_pkg_idle_wkups", "ri_interrupt_wkups", "ri_pageins",
        "ri_wired_size", "ri_resident_size", "ri_phys_footprint", "ri_proc_start_abstime",
        "ri_proc_exit_abstime", "ri_child_user_time", "ri_child_system_time", "ri_child_pkg_idle_wkups",
        "ri_child_interrupt_wkups", "ri_child_pageins", "ri_child_elapsed_abstime",
        "ri_diskio_bytesread", "ri_diskio_byteswritten")]


def phys_footprint_mb(pid=None):
    """进程的 phys_footprint（MB），和 `footprint` 命令同口径。
    不用 RSS：模型在 MPS 上，RSS 只有几百 MB，真实占用要高得多。读不到返回 None。"""
    info = RusageInfoV2()
    rc = _libproc.proc_pid_rusage(pid or os.getpid(), RUSAGE_INFO_V2, ctypes.byref(info))
    return info.ri_phys_footprint / 1048576 if rc == 0 else None


def ensure_edit_menu():
    """Accessory App 没有主菜单，WebView 里 ⌘C/⌘V/⌘A/⌘Z 和 ⌘W 都不响应；补一个看不见但能响应快捷键的主菜单。"""
    app = NSApplication.sharedApplication()
    if app.mainMenu() is not None:
        return
    main = NSMenu.alloc().init()
    # 第一项系统固定当应用菜单（标题显示 App 名），留空，免得「编辑」被顶成应用菜单
    main.setSubmenu_forItem_(NSMenu.alloc().init(), main.addItemWithTitle_action_keyEquivalent_("", None, ""))
    for title, entries in [(tr("编辑", "Edit"), [(tr("撤销", "Undo"), "undo:", "z"), (tr("重做", "Redo"), "redo:", "Z"), None,
                                               (tr("剪切", "Cut"), "cut:", "x"), (tr("拷贝", "Copy"), "copy:", "c"),
                                               (tr("粘贴", "Paste"), "paste:", "v"), (tr("全选", "Select All"), "selectAll:", "a")]),
                           (tr("窗口", "Window"), [(tr("关闭", "Close"), "performClose:", "w")])]:
        sub = NSMenu.alloc().initWithTitle_(title)
        for entry in entries:
            if entry is None:
                sub.addItem_(NSMenuItem.separatorItem())
                continue
            sub.addItemWithTitle_action_keyEquivalent_(*entry)
        top = main.addItemWithTitle_action_keyEquivalent_(title, None, "")
        main.setSubmenu_forItem_(sub, top)
    app.setMainMenu_(main)


class PanelHandler(NSObject, protocols=[objc.protocolNamed("WKScriptMessageHandler")]):
    def userContentController_didReceiveScriptMessage_(self, _controller, message):
        if self.panel is not None:
            self.panel.on_message(str(message.body()))


class PanelDelegate(NSObject):
    def windowWillClose_(self, _note):
        if self.panel is not None:
            AppHelper.callAfter(self.panel.teardown)  # 不在关窗回调里直接销毁窗口


class Panel:
    """app 需提供：panel_state(fresh_stats)、copy_last()、change_hotkey()、save_panel_settings(idle, max_s)、
    read_vocab_text()、save_vocab_text(text)。"""

    def __init__(self, app):
        self.app = app
        self.window = self.webview = self.handler = self.delegate = self.timer = None

    def is_open(self):
        return self.window is not None

    def open(self):
        """用户主动打开：这是面板唯一允许激活本 App 的地方。"""
        if self.window is None:
            self.build()
        self.window.makeKeyAndOrderFront_(None)
        NSApplication.sharedApplication().activateIgnoringOtherApps_(True)

    def build(self):
        ensure_edit_menu()
        self.handler = PanelHandler.new()
        self.handler.panel = self
        controller = WKUserContentController.new()
        controller.addScriptMessageHandler_name_(self.handler, HANDLER_NAME)
        config = WKWebViewConfiguration.new()
        config.setUserContentController_(controller)
        style = (NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskMiniaturizable
                 | NSWindowStyleMaskResizable)
        self.window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            ((0, 0), (780, 680)), style, NSBackingStoreBuffered, False)
        self.window.setReleasedWhenClosed_(False)
        self.window.setTitle_(tr("语音输入", "Voice Input"))
        self.window.setMinSize_((640, 480))
        self.window.setBackgroundColor_(NSColor.windowBackgroundColor())
        self.window.center()
        self.webview = WKWebView.alloc().initWithFrame_configuration_(self.window.contentView().bounds(), config)
        self.webview.setAutoresizingMask_(18)  # 宽高随窗口缩放
        self.webview.setUnderPageBackgroundColor_(NSColor.windowBackgroundColor())  # 页面载入前不闪白
        self.window.setContentView_(self.webview)
        self.delegate = PanelDelegate.new()
        self.delegate.panel = self
        self.window.setDelegate_(self.delegate)
        self.webview.loadHTMLString_baseURL_(HTML_PATH.read_text(encoding="utf-8"), None)
        self.timer = NSTimer.scheduledTimerWithTimeInterval_repeats_block_(MEM_REFRESH_SEC, True, lambda _t: self.push_mem())
        log.info("panel opened")

    def retitle(self):
        if self.window is not None:
            self.window.setTitle_(tr("语音输入", "Voice Input"))

    def teardown(self):
        """释放窗口和 WebView：userContentController 会强引用 handler，必须先摘掉，否则整串对象不释放。"""
        if self.window is None:
            return
        self.timer.invalidate()
        self.webview.configuration().userContentController().removeScriptMessageHandlerForName_(HANDLER_NAME)
        self.handler.panel = self.delegate.panel = None
        self.webview.stopLoading()
        self.window.setDelegate_(None)
        self.window.setContentView_(None)
        self.window.close()
        self.window = self.webview = self.handler = self.delegate = self.timer = None
        log.info("panel closed")

    def close(self):
        if self.window is not None:
            self.window.close()  # 会触发 windowWillClose_ → teardown

    # Python → 页面（evaluateJavaScript 必须在主线程）
    def call_js(self, fn, arg):
        if self.webview is None:
            return
        if not NSThread.isMainThread():
            AppHelper.callAfter(self.call_js, fn, arg)
            return
        self.webview.evaluateJavaScript_completionHandler_(f"window.panel.{fn}({json.dumps(arg)})", None)

    def push(self, fresh_stats=False):
        if self.window is None:
            return
        state = self.app.panel_state(fresh_stats)
        state["mem_mb"] = phys_footprint_mb()
        self.call_js("update", state)

    def push_mem(self):
        self.call_js("setMem", phys_footprint_mb())

    # 页面 → Python
    def on_message(self, raw):
        try:
            msg = json.loads(raw)
            kind = msg["type"]
        except (ValueError, KeyError, TypeError):
            log.warning("panel: 收到无法解析的消息")
            return
        if kind == "refresh":
            self.push(fresh_stats=True)
        elif kind == "copy_last":
            ok = self.app.copy_last()
            self.call_js("toast", {"ok": ok, "kind": "copy", "msg": tr("已复制上次结果", "Last result copied") if ok
                                   else tr("还没有可复制的结果", "Nothing to copy yet")})
        elif kind == "change_hotkey":
            AppHelper.callAfter(self.app.change_hotkey)  # 弹 NSAlert 录制，不在 WK 回调栈里嵌套模态
        elif kind == "save_settings":
            ok, text = self.app.save_panel_settings(msg.get("idle_unload_minutes"), msg.get("max_seconds"))
            self.call_js("toast", {"ok": ok, "kind": "settings", "msg": text})
            self.push()
        elif kind == "set_cleanup":
            self.app.set_cleanup(msg.get("on"))
            self.push()
        elif kind == "set_digits":
            self.app.set_digits(msg.get("on"))
            self.push()
        elif kind == "set_language":
            self.app.set_language(msg.get("lang"))  # 内部 set_state 会推送新状态
        elif kind == "get_vocab":
            self.call_js("setVocab", self.app.read_vocab_text())
        elif kind == "save_vocab":
            ok, text = self.app.save_vocab_text(msg.get("text"))
            self.call_js("toast", {"ok": ok, "kind": "vocab", "msg": text})
        else:
            log.warning("panel: 未知消息 %s", kind)
