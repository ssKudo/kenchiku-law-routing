import json, sys, time, subprocess
sys.path.insert(0, 'work')
import run_judgments as R

mats, bd, bv2, bbr, qbr = R.load_materials()
proc = subprocess.Popen([R.SERVER, "-m", R.MODEL, "-c", str(R.CTX), "-t", str(R.THREADS),
                         "--parallel", "1", "--host", "127.0.0.1", "--port", str(R.PORT)],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
try:
    print("boot", R.wait_health(proc))
    # トークン数の全体像: 各条件のプロンプト合計（キャッシュ無し想定の総量）
    for cond in ("desc_plain", "desc_bracket", "ref_v2"):
        tot = 0; mx = 0; over = []
        for qid, mq in mats.items():
            for p in mq["pool"]:
                key = (p["law_id"], p["art"])
                stem, choices, board = R.cond_texts(cond, mq, bd, bv2, bbr, qbr, key)
                n = R.tokenize(R.build_prompt(stem, choices, board))
                tot += n; mx = max(mx, n)
                if n > R.CTX - 256: over.append((qid, key, n, len(board)))
        print(cond, "total prompt tokens per round:", tot, "| max:", mx, "| over-ctx pairs:", over)
    # 実測: No.24 のプール先頭3件（描写・かっこ無し）で round1/2 を計時
    mq = mats["2017-24"]
    for p in mq["pool"][:3]:
        key = (p["law_id"], p["art"])
        stem, choices, board = R.cond_texts("desc_plain", mq, bd, bv2, bbr, qbr, key)
        prompt, bu, ntok, tr = R.fit_prompt(stem, choices, board)
        for rnd in (1, 2):
            raw, el, tim = R.judge(prompt, R.SEEDS[rnd])
            print("pair", key, "round", rnd, "tokens", ntok, "sec", round(el,1),
                  "cache", tim.get("cache_n"), "out", raw[:30].replace("\n"," "))
finally:
    proc.terminate()
