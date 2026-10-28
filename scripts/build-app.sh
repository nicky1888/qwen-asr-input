#!/bin/zsh
# 构建 ~/Applications/语音输入.app 并（首次）加进 Dock。重建后 macOS 可能要求重新授权麦克风和辅助功能。
set -euo pipefail
ROOT=${0:A:h:h}
APP="$HOME/Applications/语音输入.app"
TMP=$(mktemp -d)
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
clang -fobjc-arc -O2 -framework Cocoa -o "$APP/Contents/MacOS/launcher" "$ROOT/app/launcher.m"
"$ROOT/.venv/bin/python" "$ROOT/app/make_icon.py" "$TMP/AppIcon.iconset"
iconutil -c icns -o "$APP/Contents/Resources/AppIcon.icns" "$TMP/AppIcon.iconset"
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleIdentifier</key><string>com.nickyz.qwen-asr-input</string>
  <key>CFBundleName</key><string>语音输入</string>
  <key>CFBundleDisplayName</key><string>语音输入</string>
  <key>CFBundleExecutable</key><string>launcher</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>1.1</string>
  <key>CFBundleVersion</key><string>2</string>
  <key>LSMinimumSystemVersion</key><string>14.0</string>
  <key>LSUIElement</key><true/>
  <key>NSMicrophoneUsageDescription</key><string>语音输入需要麦克风录下你的口述，音频只在本机识别，不保存、不上传。</string>
  <key>QAIProjectRoot</key><string>$ROOT</string>
</dict></plist>
PLIST
codesign --force --sign - "$APP"
rm -rf "$TMP"
if ! defaults read com.apple.dock persistent-apps | grep -q "com.nickyz.qwen-asr-input\|语音输入.app"; then
  defaults write com.apple.dock persistent-apps -array-add \
    "<dict><key>tile-data</key><dict><key>file-data</key><dict><key>_CFURLString</key><string>file://${APP// /%20}/</string><key>_CFURLStringType</key><integer>15</integer></dict></dict></dict>"
  killall Dock
  echo "已加进 Dock"
fi
echo "构建完成：$APP"
