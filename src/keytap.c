// 全局键盘监听的原生部分。系统里每个按键都要等监听回调返回才往下走，
// 回调若写在 Python 里，就得先抢到 GIL：模型加载时 GIL 一次被占 0.6 秒，
// 进程被 App Nap 降速时也会拖，整台机器的键盘跟着卡。所以判断和吞键都在这里做，
// 只把「切换录音 / 取消 / 录到新组合」写进管道，Python 另起线程读，不在按键路径上。
// 编译：voice_input.py 启动时按需调 clang，产物放 out/libkeytap.dylib。
#include <ApplicationServices/ApplicationServices.h>
#include <dispatch/dispatch.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <unistd.h>

enum { EV_TOGGLE = 1, EV_CANCEL = 2, EV_CAPTURE = 3, EV_REENABLED = 4 };
#define KEY_ESC 53
#define MOD_MASK (kCGEventFlagMaskSecondaryFn | kCGEventFlagMaskControl | kCGEventFlagMaskAlternate | \
                  kCGEventFlagMaskShift | kCGEventFlagMaskCommand)

typedef struct { uint32_t kind, keycode; uint64_t flags; } Record;  // Python 侧按 "<IIQ" 解

static _Atomic int64_t hot_key = -1;
static _Atomic uint64_t hot_mods = 0;
static _Atomic int recording = 0, capturing = 0;
static int out_fd = -1;
static CFMachPortRef tap;
static uint8_t swallowed[128];  // 吞了按下就连抬起一起吞；只在监听线程里读写

static int is_fkey(int64_t code) {  // F1–F20，和 voice_input.KEYCODES 一致
    switch (code) {
        case 122: case 120: case 99: case 118: case 96: case 97: case 98: case 100: case 101: case 109:
        case 103: case 111: case 105: case 107: case 113: case 106: case 64: case 79: case 80: case 90:
            return 1;
    }
    return 0;
}

static void emit(uint32_t kind, int64_t code, uint64_t flags) {
    Record r = {kind, (uint32_t)code, flags};
    (void)write(out_fd, &r, sizeof r);  // 写端非阻塞：管道满了宁可丢事件，也不卡键盘
}

// 返回 1 吞掉这次按键。单独导出，测试不开监听也能驱动。
int kt_handle(int keydown, int64_t code, uint64_t flags, int repeat) {
    if (!keydown) {
        if (code >= 0 && code < 128 && swallowed[code]) {
            swallowed[code] = 0;
            return 1;
        }
        return 0;
    }
    uint64_t mods = flags & MOD_MASK;
    if (is_fkey(code)) mods &= ~(uint64_t)kCGEventFlagMaskSecondaryFn;  // 系统给 F 键自动带 fn 标记
    int swallow = 0;
    if (atomic_load(&capturing)) {
        // 录制新快捷键：带修饰键的组合和 F 键都交给 Python 判断能不能用；光按的键（含 Esc）留给对话框
        if (mods || is_fkey(code)) {
            swallow = 1;
            if (!repeat) emit(EV_CAPTURE, code, flags);
        }
    } else if (code == atomic_load(&hot_key) && mods == atomic_load(&hot_mods)) {
        swallow = 1;
        if (!repeat) emit(EV_TOGGLE, code, flags);  // 按住连发只触发一次，但照样吞掉
    } else if (code == KEY_ESC && !mods && atomic_load(&recording)) {
        swallow = 1;
        if (!repeat) emit(EV_CANCEL, code, flags);
    }
    if (swallow && code >= 0 && code < 128) swallowed[code] = 1;
    return swallow;
}

static CGEventRef callback(CGEventTapProxy proxy, CGEventType type, CGEventRef event, void *refcon) {
    if (type == kCGEventTapDisabledByTimeout || type == kCGEventTapDisabledByUserInput) {
        CGEventTapEnable(tap, true);
        emit(EV_REENABLED, 0, type);
        return event;
    }
    int64_t code = CGEventGetIntegerValueField(event, kCGKeyboardEventKeycode);
    int repeat = (int)CGEventGetIntegerValueField(event, kCGKeyboardEventAutorepeat);
    return kt_handle(type == kCGEventKeyDown, code, CGEventGetFlags(event), repeat) ? NULL : event;
}

static void *run(void *sem) {
    CGEventMask mask = CGEventMaskBit(kCGEventKeyDown) | CGEventMaskBit(kCGEventKeyUp);
    tap = CGEventTapCreate(kCGSessionEventTap, kCGHeadInsertEventTap, kCGEventTapOptionDefault, mask, callback, NULL);
    dispatch_semaphore_signal((dispatch_semaphore_t)sem);
    if (tap == NULL) return NULL;
    CFRunLoopSourceRef src = CFMachPortCreateRunLoopSource(NULL, tap, 0);
    CFRunLoopAddSource(CFRunLoopGetCurrent(), src, kCFRunLoopCommonModes);
    CGEventTapEnable(tap, true);
    CFRunLoopRun();
    return NULL;
}

// fd 是管道写端，调用方设好非阻塞。
void kt_set_output(int fd) { out_fd = fd; }

// 返回 0 成功，-1 开不了监听（多半缺辅助功能权限）。
int kt_start(void) {
    dispatch_semaphore_t sem = dispatch_semaphore_create(0);
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_set_qos_class_np(&attr, QOS_CLASS_USER_INTERACTIVE, 0);  // 后台进程也按前台优先级调度
    pthread_t th;
    int err = pthread_create(&th, &attr, run, (void *)sem);
    pthread_attr_destroy(&attr);
    if (err != 0) return -1;
    pthread_detach(th);
    dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
    return tap == NULL ? -1 : 0;
}

void kt_set_hotkey(int64_t code, uint64_t mods) {
    atomic_store(&hot_mods, mods);
    atomic_store(&hot_key, code);
}

void kt_set_recording(int on) { atomic_store(&recording, on); }
void kt_set_capturing(int on) { atomic_store(&capturing, on); }
