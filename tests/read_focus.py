"""打印前台 App 焦点控件的角色和内容（AX 读取，不需要自动化权限）。"""
from AppKit import NSWorkspace
from ApplicationServices import AXUIElementCreateApplication, AXUIElementCopyAttributeValue
app = NSWorkspace.sharedWorkspace().frontmostApplication()
ax = AXUIElementCreateApplication(app.processIdentifier())
_, el = AXUIElementCopyAttributeValue(ax, "AXFocusedUIElement", None)
g = lambda a: AXUIElementCopyAttributeValue(el, a, None)[1] if el is not None else None
print(app.localizedName(), "|", g("AXRole"), g("AXSubrole"), "|", repr(g("AXValue"))[-300:])
