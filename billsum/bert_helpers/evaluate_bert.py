from billsum.post_process import greedy_summarize, mmr_selection

import pandas as pd 
import pickle 
from rouge import Rouge
rouge = Rouge()


# Note: this script assumes the data was generated using prepare_bert_data.py
# The two go together, because we want the output predictions to match the docs


# Load in predictions

prefix = "BERT_DIR" # EDIT ME
predictions = pd.read_csv(prefix + 'test_results.tsv', sep='\t', header=None)
pos_pred = predictions[1].values


# Load in the sentence data

sents 

docs = pd.read_json('/Users/anastassiakornilova/BSDATA/clean_final/us_test_data_final.jsonl', lines=True)
docs.set_index('bill_id', inplace=True)


from billsum.post_process import mmr_selection
from rouge import Rouge

rouge = Rouge()

i = 0
all_scores = []

for k, v in sents.items():
    tot = len(v)
    ys = sent_scores[i:i+tot][1].values
    mysents = [s[0] for s in v]
    final_sum = ' '.join(mmr_selection(mysents, ys, 13333))
    score = rouge.get_scores([docs.loc[k].clean_summary], [final_sum])[0]
    all_scores.append(score)
    i += tot
    