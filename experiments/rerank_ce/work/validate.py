import json, time, sys
import numpy as np
import rerank as R
import onnxruntime as ort

which = sys.argv[1]  # int8 or fp32
path = R.WORK/'model3/model_int8.onnx' if which == 'int8' else R.WORK/'model3/model.onnx'
pools = json.load(open(R.RK/'question_tag_pilot/work/qtag50_pools.json', encoding='utf-8'))
pool_texts = json.load(open(R.RK/'question_tag_pilot/work/qtag50_pool_texts.json', encoding='utf-8'))
texts = {tuple(k.split('|')): v['text'] for k,v in pool_texts.items()}
qs = {}
for line in open(R.RK/'prototype/c_full/questions.jsonl', encoding='utf-8'):
    d = json.loads(line); qs[f'{d["year"]}-{d["no"]}'] = d['stem'] + ' ' + ' '.join(d['choices'])
seqs = []
for qid in list(pools)[:2]:
    q_ids = R.ids_of(qs[qid])
    for e in pools[qid]:
        p = R.ids_of(texts[(e['law'], e['art'])])
        seqs += R.build_pairs(q_ids, p)[:3]
seqs = sorted(seqs, key=len)
so = ort.SessionOptions(); so.intra_op_num_threads = 4
sess = ort.InferenceSession(str(path), sess_options=so, providers=['CPUExecutionProvider'])
def run(seqs):
    maxlen = max(len(s) for s in seqs)
    ii = np.full((len(seqs), maxlen), R.PAD, dtype=np.int64)
    am = np.zeros_like(ii)
    for i, s in enumerate(seqs):
        ii[i,:len(s)] = s; am[i,:len(s)] = 1
    return sess.run(None, {'input_ids': ii, 'attention_mask': am})[0].reshape(-1)
_ = run(seqs[:8])
vals, t = [], 0.0
for i in range(0, len(seqs), 16):
    t0 = time.time(); vals.append(run(seqs[i:i+16])); t += time.time()-t0
v = np.concatenate(vals)
np.save(f'val_{which}.npy', v)
print(f'{which}: n={len(seqs)} run_time={t:.1f}s ({t/len(seqs):.2f}/pair) nan={bool(np.isnan(v).any())}')
