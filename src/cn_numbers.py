"""把识别结果里的中文数字转成阿拉伯数字（百分之十二点五 → 12.5%，一千二百三十美元 → 1230美元）。

只转明确是数值的写法，拿不准就保持汉字：
- 一定转：百分比、小数、带「十百千万」的多位数、三位以上的逐位数字（二零二六年、手机号）、时刻（下午三点半）；
- 单个数字（一、两、三……）只在后面跟着计量单位时转（三号、五岁、八块、两万），「一起」「一定」「两个」「第一」不动；
- 约数（七八个、五六十）、成语和虚词（千万别、万万、十分重要、一五一十）不动。
"""
import re

DIGIT = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
SMALL = {"十": 10, "百": 100, "千": 1000}
BIG = {"万": 10_000, "亿": 100_000_000}
NUM = "零〇一二两三四五六七八九十百千万亿"
BASIC = "零〇一二三四五六七八九"  # 逐位读的数字（不含两）
# 单个数字后面跟这些才转
# 「块」「角」「毛」「刀」「度」单字后常是「一块儿」「一角」「一刀切」「一度」，不收；「三百块」这类多位数照样转
UNITS = ("美元|欧元|港币|日元|英镑|人民币|块钱|元|岁|号|日|月|年代|小时|分钟|秒钟|秒|点钟|"
         "倍|公里|千米|厘米|毫米|米|公斤|千克|斤|克|吨|升|G|GB|M|MB|K|w|W|%|万|亿")
TIME_WORDS = "凌晨|早上|早晨|上午|中午|下午|傍晚|晚上|今晚|明早|夜里"
KEEP = {"千万", "万万", "百万", "万一", "千千万万", "一一", "一五一十", "十之八九", "三三两两", "七七八八"}


def parse_int(s):
    """中文整数 → int；支持「两万五」「一千二」这类省略末位单位的口语。"""
    total, section, num, last_unit = 0, 0, None, None
    for ch in s:
        if ch in DIGIT:
            num = DIGIT[ch]
        elif ch in SMALL:
            section += (1 if num is None else num) * SMALL[ch]
            num, last_unit = None, SMALL[ch]
        elif ch in BIG:
            section += num or 0
            total += (section or 1) * BIG[ch]
            section, num, last_unit = 0, None, BIG[ch]
    if num is not None:
        prev = s[-2] if len(s) > 1 else ""
        # 「两万五」「一千二」：末位数字紧跟在单位后面，按下一级单位算
        section += num * (last_unit // 10 if prev in SMALL or prev in BIG else 1) if last_unit else num
    return total + section


def as_number(s):
    """带万/亿结尾的整数保留单位（十二万 → 12万），其余展开。"""
    if s[-1] in BIG and len(s) > 1 and not any(c in BIG for c in s[:-1]):
        return f"{parse_int(s[:-1])}{s[-1]}"
    return str(parse_int(s))


def digits(s):
    return "".join(str(DIGIT[c]) for c in s)


def percent(m):
    whole, frac = m.group(1), m.group(2)
    return f"{100 if whole == '百' else parse_int(whole)}{'.' + digits(frac) if frac else ''}%"


def clock(m):
    pre, hour, rest = m.group(1) or "", m.group(2), m.group(3) or ""
    if not pre and not rest:
        return m.group(0)
    mm = re.fullmatch(f"([{NUM}]+)分", rest)
    return f"{pre}{parse_int(hour)}点" + (f"{parse_int(mm.group(1))}分" if mm else rest)


def decimal(m):
    whole, frac, big = m.group(1), m.group(2), m.group(3) or ""
    return f"{parse_int(whole)}.{digits(frac)}{big}"


def plain(m):
    s, after = m.group(1), m.group(2)
    if s in KEEP or (s == "十" and after.startswith("分") and not after.startswith("分钟")):
        return m.group(0)
    if all(c in BASIC for c in s):
        if len(s) >= 3:
            return digits(s) + after             # 二零二六、一三八零零
        if len(s) == 2:
            return m.group(0)                    # 七八个：约数
    has_pos = any(c in SMALL or c in BIG for c in s)
    has_digit = any(c in DIGIT for c in s)
    unit = re.match(UNITS, after)
    if has_pos and has_digit and s[0] not in "百千万亿":  # 「万一」「千万」这类单位打头的多半不是数
        # 五六十、三四百：相邻两个不同数字是约数
        if re.search(f"[{NUM[:-5]}]{{2}}", s.replace("十", "").replace("百", "").replace("千", "")) and \
                not re.search("[十百千万亿]", s[:2]):
            return m.group(0)
        return as_number(s) + after
    if unit and unit.group(0) == "日" and m.string[max(0, m.start() - 1):m.start()] != "月":
        return m.group(0)                        # 「一日三餐」不动，「十月一日」转
    if unit:                                      # 单个数字或「十」「万」：要跟单位才转
        return as_number(s) + after
    return m.group(0)


def to_digits(text):
    text = re.sub(f"百分之([{NUM}]+)(?:点([{BASIC}]+))?", percent, text)
    text = re.sub(f"({TIME_WORDS})?([{NUM}]+)点(半|整|钟|[{NUM}]+分)?", clock, text)
    text = re.sub(f"([{NUM}]+)点([{BASIC}]+)([万亿]?)", decimal, text)
    lead = f"(?<![第{NUM}])"  # 从数字串开头匹配；「第一」这类序数单字不动，多位的由下一行转
    text = re.sub(f"{lead}([{NUM}]+)((?:{UNITS})?)", plain, text)
    return re.sub(f"第([{NUM}]{{2,}})", lambda m: "第" + (as_number(m.group(1)) if m.group(1) not in KEEP else m.group(1)), text)
