"""识别结果的轻度清理：只删语气词和明显的重复，不改写句子、不换词、不调语序。

规则故意保守，宁可漏删也不误删：
- 嗯、呃 在中文里几乎只当语气词，出现就删；
- 啊、哦、额、唔、哎、嘛 只在单独成段（两边是标点或句首句尾）时删，「好啊」「额度」不动；
- 「那个，」「就是，」「就是说，」这类句首口头禅，只在后面紧跟逗号时删，「那个周报」不动；
- 两字以上的词紧挨着重复（「就是就是」「我觉得，我觉得」）留一个；单字叠词（谢谢、看看）不动；
- 英文的 um / uh / erm 单独成词时删。
"""
import re

PUNCT = "，。！？、；：,.!?;:"
P = f"[{PUNCT}]"
EDGE = rf"(?:^|(?<={P}))"   # 句首或标点后
END = rf"(?={P}|$)"         # 标点前或结尾

ALWAYS = re.compile(r"[嗯呃]+[，,]?")  # 连带它后面的停顿逗号
STANDALONE = re.compile(rf"{EDGE}\s*[啊哦额唔哎嘛]+\s*{END}")
LEADING = re.compile(rf"{EDGE}\s*(?:那个|这个|就是|就是说|然后呢|然后就是)\s*(?=[，,])")
REPEAT = re.compile(rf"([一-鿿]{{2,6}})(?:[，,]?\1)+")
EN_FILLER = re.compile(r"(?i)(?:^|(?<=[\s,.!?]))(?:um+|uh+|erm+|uhm+)(?=[\s,.!?]|$)[,]?\s*")


def tidy_punct(text):
    text = re.sub(rf"[，,]+(?={P})", "", text)        # 「，。」→「。」
    text = re.sub(r"([，,])[，,]+", r"\1", text)       # 连续逗号
    text = re.sub(rf"^{P}+", "", text)                 # 句首标点
    text = re.sub(r"([。！？!?])[，,]+", r"\1", text)  # 句号后的逗号
    return re.sub(r"\s{2,}", " ", text).strip()


def clean(text):
    text = ALWAYS.sub("", text)
    text = tidy_punct(text)
    text = STANDALONE.sub("", text)
    text = tidy_punct(text)
    text = REPEAT.sub(r"\1", text)
    text = LEADING.sub("", text)  # 放在去重后：「就是就是，」先变「就是，」再删
    text = EN_FILLER.sub("", text)
    return tidy_punct(text)
