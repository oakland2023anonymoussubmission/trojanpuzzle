import sys
import json
import numpy as np

path = sys.argv[1]

with open(path) as f:
    stats = json.load(f)

assert len(stats) in [9999, 10000]

# total_token_cnt = 0
# for _, _r in stats.items():
#     total_token_cnt += _r['sample_len']

# avg_l = 0
# for _, _r in stats.items():
#     _l = (_r['loss'] * _r['sample_len']) / total_token_cnt
#     avg_l += _l
# perp = np.exp(avg_l)

avg_l = sum([_r['loss'] / len(stats) for _, _r in stats.items() if _r['loss'] <= 7])

perp = sum([np.exp(_r['loss']) / len(stats) for _, _r in stats.items() if _r['loss'] <= 7])

print(path)
print(f"perplexity: {perp:.4f}\nloss: {avg_l:.4f}")
print("--------")
