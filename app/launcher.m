// Dock 启动器：拉起 Python 主程序当子进程，麦克风和辅助功能权限就挂在本 App 名下。
// 运行中再点 Dock 图标，发通知让主程序弹出菜单栏菜单。主程序退出，本进程随之退出。
#import <Cocoa/Cocoa.h>
#include <fcntl.h>
#include <spawn.h>
extern char **environ;

@interface Launcher : NSObject <NSApplicationDelegate>
@property pid_t child;
@property(strong) dispatch_source_t exitSource, termSource;
@end

@implementation Launcher
- (void)applicationDidFinishLaunching:(NSNotification *)note {
    NSString *root = [[NSBundle mainBundle] objectForInfoDictionaryKey:@"QAIProjectRoot"];
    NSString *py = [root stringByAppendingPathComponent:@".venv/bin/python"];
    NSString *script = [root stringByAppendingPathComponent:@"src/voice_input.py"];
    NSString *log = [NSHomeDirectory() stringByAppendingPathComponent:@"Library/Logs/qwen-asr-input.log"];
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_addopen(&fa, 0, "/dev/null", O_RDONLY, 0);
    posix_spawn_file_actions_addopen(&fa, 1, log.fileSystemRepresentation, O_WRONLY | O_CREAT | O_APPEND, 0644);
    posix_spawn_file_actions_adddup2(&fa, 1, 2);
    char *argv[] = {(char *)py.fileSystemRepresentation, (char *)script.fileSystemRepresentation, NULL};
    pid_t pid;
    int err = posix_spawn(&pid, argv[0], &fa, NULL, argv, environ);
    posix_spawn_file_actions_destroy(&fa);
    if (err != 0) {
        NSAlert *a = [NSAlert new];
        a.messageText = @"语音输入启动失败";
        a.informativeText = [NSString stringWithFormat:@"拉不起 %@（%s）", py, strerror(err)];
        [a runModal];
        [NSApp terminate:nil];
        return;
    }
    self.child = pid;
    self.exitSource = dispatch_source_create(DISPATCH_SOURCE_TYPE_PROC, pid, DISPATCH_PROC_EXIT, dispatch_get_main_queue());
    dispatch_source_set_event_handler(self.exitSource, ^{
        int status;
        waitpid(self.child, &status, 0);
        self.child = 0;
        [NSApp terminate:nil];
    });
    dispatch_resume(self.exitSource);
    signal(SIGTERM, SIG_IGN);  // 改由 dispatch 处理，先停子进程再退
    self.termSource = dispatch_source_create(DISPATCH_SOURCE_TYPE_SIGNAL, SIGTERM, 0, dispatch_get_main_queue());
    dispatch_source_set_event_handler(self.termSource, ^{ [NSApp terminate:nil]; });
    dispatch_resume(self.termSource);
}

- (BOOL)applicationShouldHandleReopen:(NSApplication *)app hasVisibleWindows:(BOOL)flag {
    [[NSDistributedNotificationCenter defaultCenter] postNotificationName:@"com.nickyz.qwen-asr-input.reopen"
                                                                   object:nil userInfo:nil deliverImmediately:YES];
    return NO;
}

- (void)applicationWillTerminate:(NSNotification *)note {
    if (self.child > 0) kill(self.child, SIGTERM);
}
@end

int main(void) {
    @autoreleasepool {
        NSApplication *app = [NSApplication sharedApplication];
        Launcher *delegate = [Launcher new];
        app.delegate = delegate;
        [app run];
    }
    return 0;
}
