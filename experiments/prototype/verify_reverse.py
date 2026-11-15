# -*- coding: utf-8 -*-
"""逆引き候補の精度検証 (b): 書き出し文を号・項単位で照合し、
委任の真の受け手 (「法第○条…の政令/省令で定める」と書き出す条) と
定義を借りているだけの条 (「法第○条に規定する…」等) を分ける。
入力: prototype/contexts.json の reverse_delegation_candidates
      + lawapp/data の法令全文JSON (e-Gov 由来)
出力: 候補ごとの判定と引用文脈 (stdout)
"""
import json, re, sys
from pathlib import Path

CTX = '/home/hatch/workspace/research_kudou/prototype/contexts.json'
LAW_FILES = {
    '建築基準法施行令': '/home/hatch/workspace/lawapp/data/325CO0000000338.json',
    '建築基準法施行規則': '/home/hatch/workspace/lawapp/data/325M50004000040.json',
}
KANJI = {'2': '第二条', '6': '第六条', '12': '第十二条'}
# 委任の受け手: 同じ文の中で、種条文の引用の後に「の政令/省令で定める」が来る
DELEG = re.compile(r'の(?:政令|国土交通省令|省令)で定める')
BORROW_PAT = re.compile(r'に(?:規定する|掲げる)|の規定により|の規定による')

def load_law(path):
    d = json.load(open(path, encoding='utf-8'))
    return {a['key']: a for a in d['articles']}

def article_text(a):
    return '\n'.join(p.get('text', '') for p in a.get('paras', []))

def strip_parens(s):
    """ネストした（）を全て除去する。"""
    while True:
        t = re.sub(r'（[^（）]*）', '', s)
        if t == s:
            return t
        s = t

# 委任の受け手: 種条文の引用から「の政令/省令で定める」までの間に、
# 引用の列挙以外の言葉が入っていないこと
# （「法第六条第一項第二号に掲げる建築物のうち、木造の…」のような
#   定義借用文の中に後から出てくる別条文の「で定める」を拾わないため）
ALLOWED = set('第0123456789０１２３４５６７８９一二三四五六七八九十百千号項、及びのイロハただし書法条本文 \n　')
def is_receiver(text, seed_pat):
    for dm in DELEG.finditer(text):
        for sm in seed_pat.finditer(text[:dm.start()]):
            span = strip_parens(text[sm.end():dm.start()])
            if span and all(ch in ALLOWED for ch in span):
                return True, text[sm.start():dm.end()]
    return False, None

def split_sentences(text):
    """。で区切るが、（）内の。では区切らない（括弧内の「含む。」等で文が砕けるのを防ぐ）。"""
    parts, buf, depth = [], '', 0
    for ch in text:
        buf += ch
        if ch == '（':
            depth += 1
        elif ch == '）':
            depth = max(0, depth - 1)
        elif ch == '。' and depth == 0:
            parts.append(buf)
            buf = ''
    if buf:
        parts.append(buf)
    return parts

def classify(text, seed_pat):
    """1候補条文を判定する。戻り値 (verdict, evidence)。
    RECEIVER: 種条文の引用から「の政令/省令で定める」までが引用の列挙のみ。
    BORROWER: 「に規定する/に掲げる/の規定により」等で定義を借りるのみ。
    OTHER: どちらも無い（枝番条・他法令の引用、折りたたみ混入等）。"""
    sentences = [s for s in split_sentences(text) if seed_pat.search(s)]
    recv, recv_ev = is_receiver(text, seed_pat)
    borrow_sents = [s for s in sentences if BORROW_PAT.search(s)]
    if recv:
        return 'RECEIVER', recv_ev
    if borrow_sents:
        return 'BORROWER', borrow_sents[0]
    if sentences:
        return 'OTHER', sentences[0]
    return 'OTHER', None

def main():
    ctx = json.load(open(CTX, encoding='utf-8'))
    laws = {name: load_law(p) for name, p in LAW_FILES.items()}
    results = {}
    for seed in ['2', '6', '12']:
        kanji = KANJI[seed]
        # v0.3: build_subgraph.py と同じガード。「法第二条」の直前が漢字の
        # 場合は他法令参照（ガス事業法第二条等）とみなし除外するが、
        # 直前の4字が「建築基準」の正式名「建築基準法第二条」は認める。
        guard = r'(?:(?<![\u4e00-\u9fff])|(?<=建築基準))法'
        readings = {
            # 厳密読み: 種条文そのもの (法第六条の三等、枝番条は別条文として扱う)
            'strict': re.compile(guard + kanji + r'(?!の[0-9０-９一二三四五六七八九十])'),
            # 家族読み: 種条文＋その枝番条 (法第六条の二・の三・の四等も同じ手続き群とみなす)
            'family': re.compile(guard + kanji + r'(?:の[0-9０-９一二三四五六七八九十]+)?'),
        }
        print(f'\n===== 種条文 法{seed}条 =====')
        cands = ctx[seed]['reverse_delegation_candidates']
        results[seed] = {}
        for rname, seed_pat in readings.items():
            counts = {'RECEIVER': 0, 'BORROWER': 0, 'OTHER': 0, 'MISSING': 0}
            detail = {}
            for law, arts in cands.items():
                for art in arts:
                    key = art.replace('-', '_')
                    a = laws[law].get(key)
                    if a is None:
                        counts['MISSING'] += 1
                        detail[f'{law} 第{art}条'] = 'MISSING'
                        continue
                    text = article_text(a)
                    verdict, ev = classify(text, seed_pat)
                    counts[verdict] += 1
                    detail[f'{law} 第{art}条'] = verdict
                    print(f'  [{rname}:{verdict}] {law} 第{art}条 {a.get("caption","")}')
                    if ev:
                        print(f'      根拠: …{str(ev).replace(chr(10)," ")[:180]}…')
                    else:
                        print(f'      根拠: (条文中に「法{kanji}」の文字列なし)')
            print(f'  -- 法{seed}条 [{rname}] まとめ: ' +
                  ' / '.join(f'{k} {v}' for k, v in counts.items()))
            results[seed][rname] = {'counts': counts, 'detail': detail}
    out = Path('/home/hatch/workspace/research_kudou/prototype/VERIFY_b.json')
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\nwrote', out)

if __name__ == '__main__':
    main()
