// 录音的原生部分，替换 sounddevice/PortAudio。
// 2026-11-02 实测 PortAudio 在 macOS 上停流会死锁：主线程停流时拿着 AudioUnit 的锁去等 CoreAudio 的锁，
// CoreAudio 的 IO 线程拿着后者回调 PortAudio，又来要前者，两边永远等下去，整个 App 卡死。
// 这里直接用 AudioQueue：系统把采样率转好（16kHz 单声道 float），回调只往缓冲区里拷，不碰 Python。
// 每次录音新建一个队列，所以总用当时的系统默认麦克风，中途插拔耳机后下一次就能跟上。
// 编译：voice_input.py 启动时按需调 clang，产物放 out/libaudiorec.dylib。
#include <AudioToolbox/AudioToolbox.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

#define NBUF 3
#define BUF_SEC 0.05  // 每块 50ms，浮条波形够跟手

static AudioQueueRef queue;            // 只在调用方的单个录音线程里开关
static _Atomic int running;
static _Atomic uint32_t level_bits;    // 最近一块的音量（RMS），float 按位存
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static float *samples;                 // 以下三项受 lock 保护
static size_t count, cap;
static _Atomic uint64_t blocks;        // 回调累计次数，判断麦克风是不是还在送数据

// 追加采样。回调和测试共用；内存不够就丢这一块，不让回调失败。
void ar_append(const float *p, size_t n) {
    float sum = 0;
    for (size_t i = 0; i < n; i++) sum += p[i] * p[i];
    float rms = n ? sqrtf(sum / n) : 0;
    uint32_t bits;
    memcpy(&bits, &rms, sizeof bits);
    atomic_store(&level_bits, bits);
    atomic_fetch_add(&blocks, 1);
    pthread_mutex_lock(&lock);
    if (count + n > cap) {
        size_t want = cap ? cap * 2 : 16000 * 30;
        while (want < count + n) want *= 2;
        float *grown = realloc(samples, want * sizeof(float));
        if (grown != NULL) {
            samples = grown;
            cap = want;
        }
    }
    if (count + n <= cap) {
        memcpy(samples + count, p, n * sizeof(float));
        count += n;
    }
    pthread_mutex_unlock(&lock);
}

static void on_input(void *ud, AudioQueueRef q, AudioQueueBufferRef buf, const AudioTimeStamp *t,
                     UInt32 npackets, const AudioStreamPacketDescription *desc) {
    ar_append((const float *)buf->mAudioData, buf->mAudioDataByteSize / sizeof(float));
    if (atomic_load(&running)) AudioQueueEnqueueBuffer(q, buf, 0, NULL);
}

// 清空缓冲区并开麦。返回 0 成功，否则是 OSStatus（多半是没有麦克风权限或没有输入设备）。
int ar_start(double rate) {
    if (queue != NULL) return -1;
    pthread_mutex_lock(&lock);
    count = 0;
    pthread_mutex_unlock(&lock);
    atomic_store(&level_bits, 0);
    atomic_store(&blocks, 0);
    AudioStreamBasicDescription fmt = {
        .mSampleRate = rate, .mFormatID = kAudioFormatLinearPCM,
        .mFormatFlags = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked,
        .mBytesPerPacket = 4, .mFramesPerPacket = 1, .mBytesPerFrame = 4, .mChannelsPerFrame = 1, .mBitsPerChannel = 32,
    };
    // 回调跑在 AudioQueue 自己的线程上（runloop 传 NULL），和 Python、主线程都无关
    OSStatus err = AudioQueueNewInput(&fmt, on_input, NULL, NULL, NULL, 0, &queue);
    if (err != noErr) {
        queue = NULL;
        return (int)err;
    }
    UInt32 bytes = (UInt32)(rate * BUF_SEC) * 4;
    for (int i = 0; i < NBUF && err == noErr; i++) {
        AudioQueueBufferRef b;
        err = AudioQueueAllocateBuffer(queue, bytes, &b);
        if (err == noErr) err = AudioQueueEnqueueBuffer(queue, b, 0, NULL);
    }
    atomic_store(&running, 1);
    if (err == noErr) err = AudioQueueStart(queue, NULL);
    if (err != noErr) {
        atomic_store(&running, 0);
        AudioQueueDispose(queue, true);
        queue = NULL;
    }
    return (int)err;
}

// 关麦，返回录到的采样数；没在录也安全。
size_t ar_stop(void) {
    if (queue != NULL) {
        atomic_store(&running, 0);
        AudioQueueStop(queue, true);
        AudioQueueDispose(queue, true);
        queue = NULL;
    }
    pthread_mutex_lock(&lock);
    size_t n = count;
    pthread_mutex_unlock(&lock);
    return n;
}

// 拷出前 n 个采样，返回实际拷了多少。
size_t ar_copy(float *out, size_t n) {
    pthread_mutex_lock(&lock);
    if (n > count) n = count;
    memcpy(out, samples, n * sizeof(float));
    pthread_mutex_unlock(&lock);
    return n;
}

void ar_reset(void) {
    pthread_mutex_lock(&lock);
    count = 0;
    pthread_mutex_unlock(&lock);
}

float ar_level(void) {
    uint32_t bits = atomic_load(&level_bits);
    float v;
    memcpy(&v, &bits, sizeof v);
    return v;
}

uint64_t ar_blocks(void) { return atomic_load(&blocks); }
