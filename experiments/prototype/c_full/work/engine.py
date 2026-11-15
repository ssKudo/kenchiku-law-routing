#!/usr/bin/env python3
"""法アプリ v1.4 の候補検索エンジン (rankCandidates) の Python 忠実移植。
コーパスは lawapp/data の11法令・全条文 (記事単位ドキュメント)。
スコア式・idf・TERMS・SYN・見出しブースト・長さ正規化は JS 実装に合わせる。"""
import json, math, re
from pathlib import Path

LAWAPP = Path("/home/hatch/workspace/lawapp")
TERMS = ['建築確認','確認申請','建築主事','特定行政庁','特殊建築物','耐火建築物','準耐火建築物','耐火構造','準耐火構造','防火地域','準防火地域','容積率','建蔽率','斜線制限','道路斜線','隣地斜線','北側斜線','日影','高度地区','用途地域','第一種低層住居専用地域','第二種低層住居専用地域','第一種中高層住居専用地域','準住居地域','第一種住居地域','第二種住居地域','近隣商業地域','商業地域','準工業地域','工業地域','工業専用地域','構造計算','許容応力度','保有水平耐力','限界耐力','避難階段','直通階段','排煙設備','非常用照明','内装制限','シックハウス','ホルムアルデヒド','換気設備','採光','界壁','間仕切壁','防火区画','竪穴区画','面積区画','スプリンクラー','自動火災報知','消防用設備','建築設備','昇降機','定期報告','一級建築士','二級建築士','木造建築士','管理建築士','構造設計一級建築士','設備設計一級建築士','工事監理','中間検査','完了検査','検査済証','確認済証','仮使用','接道','位置指定道路','壁面線','外壁の後退','工作物','擁壁','増築','改築','大規模の修繕','大規模の模様替','用途変更','既存不適格','違反建築物','是正命令','仮設建築物','仮囲い','地盤','基礎','構造耐力','積雪荷重','風圧力','地震力','層間変形角','剛性率','偏心率','共同住宅','長屋','寄宿舎','劇場','映画館','集会場','病院','診療所','学校','保育所','老人ホーム','バリアフリー','移動等円滑化','省エネ','断熱','外皮平均熱貫流率','一次エネルギー消費量','長期優良住宅','耐震改修','耐震診断','都市計画','市街化区域','市街化調整区域','開発行為','開発許可','地区計画','風致地区','危険物','防火対象物','防火管理者','延焼のおそれのある部分','主要構造部','耐火被覆','耐火性能','準耐火性能','防火性能','不燃性能','遮炎性能','避難安全検証','廊下','敷地内通路','敷地面積','建築面積','延べ面積','床面積','階数','地階','居室','天井高','軒高','建築物高さ','前面道路','幅員','セットバック','天空率','日影規制','隣地境界線','道路境界線','完了検査済証','指定確認検査機関','建築基準関係規定','防火戸','特定天井','脱落防止','エレベーター','ホームエレベーター','駐車場','自転車駐車場','自動車車庫','車庫','倉庫','畜舎','遊技場','ボーリング場','スケート場','ホテル','旅館','飲食店','物販店舗','百貨店','展示場','公衆浴場','待合','料理店','キャバレー','ナイトクラブ','ダンスホール','カラオケボックス','葬祭場','斎場','火葬場','寺院','教会','幼稚園','小学校','中学校','高等学校','大学','図書館','博物館','美術館','体育館','ボウリング','マージャン','パチンコ','劇場等']
SYN = {'建ぺい率':'建蔽率','建ペい率':'建蔽率','エレベーター':'昇降機','高さの制限':'斜線制限','日影制限':'日影','2項道路':'位置指定道路','みなし道路':'位置指定道路','耐震補強':'耐震改修','バリアフリー法':'移動等円滑化','省エネ法':'一次エネルギー消費量','確認検査機関':'指定確認検査機関'}

_QNORM_RE = re.compile(r"[\s　、。，．・「」『』（）()\[\]{}:：;；!?！？＂\"'']")
def qnorm(t):
    return _QNORM_RE.sub("", t or "")

def gram_counts(t):
    m = {}
    for i in range(len(t) - 1):
        g = t[i:i+2]; m[g] = m.get(g, 0) + 1
    for i in range(len(t) - 2):
        g = t[i:i+3]; m[g] = m.get(g, 0) + 1
    return m

class Engine:
    def __init__(self):
        idx_js = (LAWAPP / "www/laws_index.js").read_text(encoding="utf-8")
        self.index = json.loads(idx_js[idx_js.index("["):idx_js.rindex("]") + 1])
        self.docs = []
        self.by_key = {}
        for meta in self.index:
            d = json.loads((LAWAPP / f"data/{meta['id']}.json").read_text(encoding="utf-8"))
            for a in d["articles"]:
                paras = [p["text"] for p in a["paras"]]
                full = qnorm((a.get("caption") or "") + " " + " ".join(paras))
                gc = gram_counts(full)
                disp = a.get("label") or a.get("title") or ("第" + str(a.get("num")) + "条")
                doc = {"law": meta["id"], "law_title": meta["title"],
                       "key_raw": a["key"], "art": a["key"].replace("_", "-"),
                       "cap": a.get("caption") or "", "full": full,
                       "grams": gc, "total": sum(gc.values()),
                       "cap_grams": gram_counts(qnorm((a.get("caption") or "") + " " + disp))}
                self.by_key[(meta["id"], doc["art"])] = len(self.docs)
                self.docs.append(doc)
        df = {}
        for doc in self.docs:
            for g in doc["grams"]:
                df[g] = df.get(g, 0) + 1
        n = len(self.docs)
        self.idf = {g: math.log(1 + (n - c + 0.5) / (c + 0.5)) for g, c in df.items()}
        self.avg_tot = sum(d["total"] for d in self.docs) / max(len(self.docs), 1)

    def _term_w(self, t):
        nt = qnorm(t)
        gs = [nt[i:i+2] for i in range(len(nt) - 1)]
        if not gs:
            return 4.0
        return min(11.0, 2.0 + sum(self.idf.get(g, 1.0) for g in gs) / len(gs))

    def score_all(self, raw):
        extra = [c for a, c in SYN.items() if a in raw and c not in raw]
        qg = gram_counts(qnorm(raw + "".join(extra)))
        qterms = [t for t in TERMS if t in raw or t in extra]
        scores = {}
        for doc in self.docs:
            sc = 0.0
            for g, c in qg.items():
                w = self.idf.get(g)
                if w is None:
                    continue
                tf = doc["grams"].get(g, 0)
                if tf:
                    sc += w * min(c, 3) * (1 + math.log(tf))
                ctf = doc["cap_grams"].get(g, 0)
                if ctf:
                    sc += w * 2.2 * (1 + math.log(ctf))
            for t in qterms:
                if t in doc["full"]:
                    w = self._term_w(t)
                    sc += w * 1.6
                    if t in doc["cap"]:
                        sc += w * 1.1
            sc = sc / (max(doc["total"], 1) / self.avg_tot) ** 0.3
            if sc > 0:
                scores[(doc["law"], doc["art"])] = sc
        return scores

    def rank(self, raw, limit=10):
        scores = self.score_all(raw)
        return sorted(scores.items(), key=lambda kv: -kv[1])[:limit]
