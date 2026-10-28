"""画 App 图标：圆角方块 + SF Symbols 麦克风，输出 iconset 目录。"""
import sys
from pathlib import Path
from AppKit import (NSBezierPath, NSBitmapImageRep, NSColor, NSGradient, NSGraphicsContext, NSImage,
                    NSImageSymbolConfiguration, NSMakeRect, NSPNGFileType, NSCompositingOperationSourceOver)

BASE = (16, 32, 128, 256, 512)  # iconset 要的五档，每档再配一张两倍图
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
for size in (16, 32, 64, 128, 256, 512, 1024):
    rep = NSBitmapImageRep.alloc().initWithBitmapDataPlanes_pixelsWide_pixelsHigh_bitsPerSample_samplesPerPixel_hasAlpha_isPlanar_colorSpaceName_bytesPerRow_bitsPerPixel_(
        None, size, size, 8, 4, True, False, "NSCalibratedRGBColorSpace", 0, 0)
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.setCurrentContext_(NSGraphicsContext.graphicsContextWithBitmapImageRep_(rep))
    inset = size * 0.1
    rect = NSMakeRect(inset, inset, size - 2 * inset, size - 2 * inset)
    path = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(rect, size * 0.18, size * 0.18)
    NSGradient.alloc().initWithStartingColor_endingColor_(
        NSColor.colorWithSRGBRed_green_blue_alpha_(0.30, 0.42, 0.98, 1), NSColor.colorWithSRGBRed_green_blue_alpha_(0.16, 0.20, 0.62, 1)
    ).drawInBezierPath_angle_(path, -90)
    cfg = NSImageSymbolConfiguration.configurationWithPointSize_weight_(size * 0.42, 0.3).configurationByApplyingConfiguration_(
        NSImageSymbolConfiguration.configurationWithHierarchicalColor_(NSColor.whiteColor()))
    mic = NSImage.imageWithSystemSymbolName_accessibilityDescription_("mic.fill", None).imageWithSymbolConfiguration_(cfg)
    w, h = mic.size()
    mic.drawInRect_fromRect_operation_fraction_(NSMakeRect((size - w) / 2, (size - h) / 2, w, h), NSMakeRect(0, 0, 0, 0),
                                                NSCompositingOperationSourceOver, 1.0)
    NSGraphicsContext.restoreGraphicsState()
    png = rep.representationUsingType_properties_(NSPNGFileType, {})
    if size in BASE:
        png.writeToFile_atomically_(str(out / f"icon_{size}x{size}.png"), True)
    if size // 2 in BASE:
        png.writeToFile_atomically_(str(out / f"icon_{size // 2}x{size // 2}@2x.png"), True)
