"""
Combine subwords and generate word vec representation
"""
import pandas as pd
import sys

orig_vec_rep_fn = sys.argv[1]
output_file = sys.argv[2]

orig_vec_rep = pd.read_csv(orig_vec_rep_fn, sep=' ', skipinitialspace=True)
col_names = list(orig_vec_rep.columns.values)

# combined subwords and get averaged vec representations
spr_vec_rep_list = []
max_word_ix = orig_vec_rep.iloc[-1]["word_ix"]

for curr_word_ix in range(max_word_ix+1):

    vec_lines = orig_vec_rep.loc[orig_vec_rep["word_ix"] == curr_word_ix]
    only_vecs = vec_lines.drop(["word", "word_ix"], axis=1)
    avg_vec = only_vecs.mean(axis=0)
    avg_vec_list = avg_vec.tolist()

    vec_indices = list(vec_lines.index.values)
    vec_indices.sort()
    
    curr_word = ""
    for i in vec_indices:
        curr_word += vec_lines["word"][i]
    curr_word = curr_word.replace(">", "") # removing the place holder for "Ġ" (i.e., ">")
    
    word_ix = vec_lines["word_ix"][vec_indices[0]]

    new_vec_line = [curr_word, word_ix] # sentpos will be changed and word_ix will be removed
    new_vec_line += avg_vec_list
    spr_vec_rep_list.append(new_vec_line)

spr_vec_rep = pd.DataFrame(spr_vec_rep_list, columns = col_names)

# remove word_ix column
spr_vec_rep.drop("word_ix", axis=1, inplace = True)
spr_vec_rep.to_csv(output_file, index=False, sep=' ')