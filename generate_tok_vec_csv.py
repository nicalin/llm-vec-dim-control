import pandas as pd
import json
import sys

json_output_vec_fp = sys.argv[1]
json_token_fp = sys.argv[2]
output_vec_rep = sys.argv[3]

with open(json_output_vec_fp, 'r') as f:
    output_vec = json.load(f)

with open(json_token_fp, 'r') as f:
    tokens = json.load(f)

df = pd.DataFrame(output_vec, index = tokens)
df.to_csv(output_vec_rep, index=False, sep=' ')
