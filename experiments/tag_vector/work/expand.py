#!/usr/bin/env python3
"""クエリ拡張: expansion_dict.json の辞書引きによる機械的追加のみ（生成なし・置換禁止）。
手順: (1) 語エントリ照合→追加 (2) 数値の両表記追加 (3) 単位ゆれ追加。
足した語は全て記録して返す。
"""
import json, re
from pathlib import Path

D = json.load(open(Path(__file__).parent.parent / "expansion_dict.json", encoding="utf-8"))
ENTRIES = D["entries"]
UNITS = D["units"]
NUMERALS = D["numerals"]  # 算用 -> 漢字
KEYS = sorted(ENTRIES, key=len, reverse=True)
UNIT_KEYS = sorted(UNITS, key=len, reverse=True)

UNIT_AFTER = ["平方メートル", "センチメートル", "ミリメートル", "キロメートル", "メートル",
              "時間", "分", "秒", "日", "月", "年", "階", "人", "戸", "台", "件", "回",
              "倍", "割", "％", "%", "ルクス", "デシベル", "リットル", "キログラム", "室", "棟", "以上", "以下", "未満"]
KD = {"〇": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
DK = "〇一二三四五六七八九"


def kanji_to_int(s):
    total, cur = 0, 0
    for ch in s:
        if ch in KD:
            cur = KD[ch]
        elif ch == "十":
            total += (cur or 1) * 10; cur = 0
        elif ch == "百":
            total += (cur or 1) * 100; cur = 0
        elif ch == "千":
            total += (cur or 1) * 1000; cur = 0
        elif ch == "万":
            total = (total + cur or 1) * 10000 if (total + cur) else 10000
            if total == 0:
                total = 10000
            cur = 0
        else:
            return None
    return total + cur


def decimal_to_kanji(s):
    if "." in s:
        ip, fp = s.split(".", 1)
        k = NUMERALS.get(ip) or "".join(DK[int(c)] for c in ip)
        return k + "・" + "".join(DK[int(c)] for c in fp)
    return NUMERALS.get(s)


def expand(text):
    norm = text.translate(str.maketrans("０１２３４５６７８９％", "0123456789%"))
    added = []

    def try_add(w):
        if w and w not in text and w not in added:
            added.append(w)

    # (1) 語エントリ
    for k in KEYS:
        if k in norm:
            for a in ENTRIES[k]:
                try_add(a)
    # (2a) 算用数字 -> 漢数字（単位付きも）
    for m in re.finditer(r"\d+(?:\.\d+)?", norm):
        s = m.group(0)
        k = decimal_to_kanji(s)
        if k:
            try_add(k)
            rest = norm[m.end():m.end() + 6]
            for u in UNIT_AFTER:
                if rest.startswith(u):
                    try_add(k + u)
                    break
    # (2b) 漢数字 -> 算用数字（単位付きも）
    for m in re.finditer(r"[一二三四五六七八九][一二三四五六七八九十百千万]*", norm):
        s = m.group(0)
        v = kanji_to_int(s)
        if v is not None and v > 0:
            try_add(str(v))
            rest = norm[m.end():m.end() + 6]
            for u in UNIT_AFTER:
                if rest.startswith(u):
                    try_add(str(v) + u)
                    break
    # 漢数字の小数（一・五 等）
    for m in re.finditer(r"([一二三四五六七八九十百千万]+)・([一二三四五六七八九]+)", norm):
        ip = kanji_to_int(m.group(1))
        fp = "".join(str(KD[c]) for c in m.group(2))
        if ip is not None:
            try_add(f"{ip}.{fp}")
    # (3) 単位ゆれ
    for u in UNIT_KEYS:
        if u in norm:
            for a in UNITS[u]:
                try_add(a)
    return text + " " + " ".join(added), added


if __name__ == "__main__":
    q = "高さが16メートルを超える木造の建物で、延べ面積が300平方メートルを超える場合、一級建築士でなければ設計できないか。エレベーターの設置はどうか。着手する日の30日前までに届け出る必要がある。"
    e, a = expand(q)
    print("added:", len(a), a)
    print(e[:400])
