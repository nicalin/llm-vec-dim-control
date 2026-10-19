"""
Pythia family:
"EleutherAI/pythia-70m", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m", "EleutherAI/pythia-1b",
"EleutherAI/pythia-1.4b", "EleutherAI/pythia-6.9b", "EleutherAI/pythia-12b",
each with checkpoints specified by training steps:
"step1", "step2", "step4", ..., "step142000", "step143000"
"""

import os, sys, torch, transformers
from transformers import AutoTokenizer, GPTNeoXForCausalLM
from collections import defaultdict
import sys
import json
import numpy as np
import random


def generate_stories(fn):
    
    f = open(fn).read()
    first_line = f.splitlines()[0]
    assert first_line.strip() == "!ARTICLE"

    # collect stories
    f = f.replace("Ñ", "N") # necessary for provo to avoid encoding problem
    unproc_stor = f.split("!ARTICLE\n")
    unproc_stor = [story for story in unproc_stor if story != ""]

    split_stories = [] # each story, split by sent
    stories = [] # a list of stories, not split by sent
    for story in unproc_stor:
        sents = story.splitlines()
        split_stories.append(sents)
        stories.append(" ".join(sents))
    return stories, split_stories


def main():
    stories, split_stories = generate_stories(sys.argv[1])
    model_variant = sys.argv[2].split("/")[-1]
    training_step = sys.argv[3]
    layer_pos = sys.argv[4]
    json_token_fp = sys.argv[5]
    json_output_vec_fp = sys.argv[6]

    if "pythia" not in model_variant:
        raise ValueError("Unsupported LLM variant")
    else:
        tokenizer = AutoTokenizer.from_pretrained(sys.argv[2], revision=sys.argv[3], cache_dir=f"hf_models/{model_variant}_{sys.argv[3]}")
    
    model = GPTNeoXForCausalLM.from_pretrained(sys.argv[2], revision=sys.argv[3], cache_dir=f"hf_models/{model_variant}_{sys.argv[3]}", output_hidden_states=True)

    model.eval()
    ctx_size = model.config.max_position_embeddings
    bos_id = model.config.bos_token_id

    batches = []
    words = []
    for i, story in enumerate(stories):
        words.extend(story.split(" "))

        # get token ids and attn from nested list - a list of sentences
        split_sent_tok_output = tokenizer(split_stories[i])
        split_sent_tok_ids = split_sent_tok_output.input_ids
        split_sent_tok_attn = split_sent_tok_output.attention_mask
        
        # flatten the list
        ids = [idx for sent_token_ids in split_sent_tok_ids for idx in sent_token_ids]
        attn = [w for weights in split_sent_tok_attn for w in weights]
        start_idx = 0


        # sliding windows with 50% overlap
        # start_idx is for correctly indexing the "later 50%" of sliding windows
        while len(ids) > ctx_size:

            # for pythia model specifically (bos_id not appended by default)
            batches.append((transformers.BatchEncoding({"input_ids": torch.tensor([bos_id] + ids[:ctx_size-1]).unsqueeze(0),
                                                           "attention_mask": torch.tensor([1] + attn[:ctx_size-1]).unsqueeze(0)}),
                            start_idx))

            ids = ids[int(ctx_size/2):]
            attn = attn[int(ctx_size/2):]
            start_idx = int(ctx_size/2)-1

        # remaining tokens
        batches.append((transformers.BatchEncoding({"input_ids": torch.tensor([bos_id] + ids).unsqueeze(0),
                                                    "attention_mask": torch.tensor([1] + attn).unsqueeze(0)}),
                        start_idx))


    # for getting indices for time stamps
    curr_word_ix = 0

    # dict for storing vec information
    output_vec = defaultdict(list)
    tokens = []
    for batch_idx, batch in enumerate(batches):

        batch_input, start_idx = batch
        output_ids = batch_input.input_ids.squeeze(0)[1:]

        with torch.no_grad():
            model_output = model(**batch_input)

        toks = tokenizer.convert_ids_to_tokens(batch_input.input_ids.squeeze(0))[1:]

        # get index for specific layer
        n_layers = len(model_output.hidden_states) - 1 # number of layers (minus the emebdding layer; layer 0)
        layer_index = int(n_layers * float(layer_pos))

        for i in range(start_idx, len(toks)):
            # necessary for diacritics in Dundee
            cleaned_tok = toks[i].replace("Ġ", "", 1).encode("latin-1").decode("utf-8")

            if cleaned_tok == "</s>":
                continue
            
            # for Dundee
            if cleaned_tok == "":
                tokens.append(">")
                output_vec["word"].append(">")
            else:
                tokens.append(cleaned_tok)
                output_vec["word"].append(cleaned_tok)

            # store word indices - for combining tokens into words
            output_vec["word_ix"].append(curr_word_ix)

            vec = model_output.hidden_states[layer_index][0][i+1]
            vec = vec.tolist()
            for feat_num, feat in enumerate(vec):
                col_name = "feat%d"%(feat_num)
                output_vec[col_name].append(feat)
            
            # update index for getting curr_word_ix
            words[curr_word_ix] = words[curr_word_ix].replace(cleaned_tok, "", 1)
            if words[curr_word_ix] == "":
                curr_word_ix += 1
    
    with open(json_token_fp, 'w') as f:
        json.dump(tokens, f, indent=4)
        
    with open(json_output_vec_fp, 'w') as f:
        json.dump(output_vec, f)

    

if __name__ == "__main__":
    main()
    
