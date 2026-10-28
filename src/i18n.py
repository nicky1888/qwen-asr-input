"""界面语言：zh / en。存在 settings.json 的 language 键里，控制面板和菜单栏都跟着切换。"""
LANGS = ("zh", "en")
LANG = "zh"


def set_lang(lang):
    global LANG
    LANG = lang if lang in LANGS else "zh"


def tr(zh, en):
    return en if LANG == "en" else zh
