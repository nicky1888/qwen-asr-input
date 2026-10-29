"""录音浮条：屏幕底部居中的小胶囊。录音时显示红点、随音量跳动的波形和时长，识别时显示流动波形，
回到就绪就收起。窗口不激活、不接鼠标，不会抢走输入焦点，否则粘贴目标会变。"""
import math
import time

import objc
from AppKit import (
    NSBackingStoreBuffered, NSBezierPath, NSColor, NSFont, NSFontAttributeName, NSForegroundColorAttributeName,
    NSMakeRect, NSPanel, NSScreen, NSStatusWindowLevel, NSView, NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorFullScreenAuxiliary, NSWindowCollectionBehaviorIgnoresCycle,
    NSWindowCollectionBehaviorStationary, NSWindowStyleMaskBorderless, NSWindowStyleMaskNonactivatingPanel,
)
from Foundation import NSAttributedString, NSTimer

from i18n import tr

W, H = 176, 40
BOTTOM_GAP = 28   # 离屏幕可用区域底边（Dock 之上）的距离
BARS = 13
FPS = 30


def level_to_height(rms):
    """音量（RMS）→ 0–1。按分贝映射，-50dB 算静音，-12dB 顶满，正常说话大约在中段。"""
    db = 20 * math.log10(max(rms, 1e-6))
    return min(1.0, max(0.0, (db + 50) / 38))


def elapsed_label(secs):
    return f"{int(secs) // 60}:{int(secs) % 60:02d}"


class OverlayView(NSView):
    def initWithFrame_(self, frame):
        self = objc.super(OverlayView, self).initWithFrame_(frame)
        if self is None:
            return None
        self.mode, self.levels, self.t0 = "recording", [0.0] * BARS, time.time()
        return self

    def isFlipped(self):
        return False

    def drawRect_(self, _rect):
        b = self.bounds()
        pill = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(NSMakeRect(0.5, 0.5, W - 1, H - 1), H / 2 - 1, H / 2 - 1)
        NSColor.colorWithWhite_alpha_(0.08, 0.9).setFill()
        pill.fill()
        NSColor.colorWithWhite_alpha_(1.0, 0.14).setStroke()
        pill.setLineWidth_(1)
        pill.stroke()
        now = time.time() - self.t0
        cy = b.size.height / 2
        if self.mode == "recording":  # 红点慢呼吸，表示麦克风开着
            NSColor.systemRedColor().colorWithAlphaComponent_(0.75 + 0.25 * math.sin(now * 4)).setFill()
            label = elapsed_label(now)
        else:
            NSColor.systemBlueColor().setFill()
            label = tr("识别中", "Working")
        NSBezierPath.bezierPathWithOvalInRect_(NSMakeRect(16, cy - 4, 8, 8)).fill()

        bar_w, gap, x0, max_h = 3, 3, 34, H - 16
        NSColor.colorWithWhite_alpha_(1.0, 0.92 if self.mode == "recording" else 0.6).setFill()
        for i in range(BARS):
            if self.mode == "recording":
                v = self.levels[i]
            else:  # 识别中：一道波从左往右走，和音量无关
                v = 0.25 + 0.25 * (1 + math.sin(now * 6 - i * 0.6)) / 2
            h = 3 + v * (max_h - 3)
            NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                NSMakeRect(x0 + i * (bar_w + gap), cy - h / 2, bar_w, h), 1.5, 1.5).fill()

        attrs = {NSFontAttributeName: NSFont.monospacedDigitSystemFontOfSize_weight_(12, 0.3),
                 NSForegroundColorAttributeName: NSColor.colorWithWhite_alpha_(1.0, 0.85)}
        text = NSAttributedString.alloc().initWithString_attributes_(label, attrs)
        size = text.size()
        text.drawAtPoint_((W - 16 - size.width, cy - size.height / 2))


class Overlay:
    def __init__(self):
        self.panel = None
        self.view = None
        self.timer = None
        self.level = 0.0  # 录音回调线程写，绘制定时器读；单个 float 赋值不用加锁

    def build(self):
        self.panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, W, H), NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel,
            NSBackingStoreBuffered, False)
        self.panel.setOpaque_(False)
        self.panel.setBackgroundColor_(NSColor.clearColor())
        self.panel.setHasShadow_(True)
        self.panel.setLevel_(NSStatusWindowLevel)
        self.panel.setIgnoresMouseEvents_(True)
        self.panel.setHidesOnDeactivate_(False)
        self.panel.setCollectionBehavior_(  # 每个桌面、全屏 App 上都能看到，不进 ⌘` 窗口循环
            NSWindowCollectionBehaviorCanJoinAllSpaces | NSWindowCollectionBehaviorStationary
            | NSWindowCollectionBehaviorFullScreenAuxiliary | NSWindowCollectionBehaviorIgnoresCycle)
        self.view = OverlayView.alloc().initWithFrame_(NSMakeRect(0, 0, W, H))
        self.panel.setContentView_(self.view)

    def show(self, mode):
        """mode: recording / transcribing。从录音切到识别时不重新定位，只换画法。"""
        if self.panel is None:
            self.build()
        if not self.panel.isVisible():
            screen = NSScreen.mainScreen().visibleFrame()  # 当前键盘焦点所在的屏幕
            self.panel.setFrameOrigin_((screen.origin.x + (screen.size.width - W) / 2, screen.origin.y + BOTTOM_GAP))
            self.view.levels = [0.0] * BARS
            self.panel.orderFrontRegardless()
        if mode != self.view.mode or self.timer is None:
            self.view.t0 = time.time()
        self.view.mode = mode
        if self.timer is None:
            self.timer = NSTimer.scheduledTimerWithTimeInterval_repeats_block_(1 / FPS, True, lambda _t: self.tick())
        self.view.setNeedsDisplay_(True)

    def tick(self):
        if self.view.mode == "recording":
            self.view.levels = self.view.levels[1:] + [level_to_height(self.level)]  # 新音量从右边进
        self.view.setNeedsDisplay_(True)

    def hide(self):
        if self.timer is not None:
            self.timer.invalidate()
            self.timer = None
        if self.panel is not None:
            self.panel.orderOut_(None)
        self.level = 0.0
