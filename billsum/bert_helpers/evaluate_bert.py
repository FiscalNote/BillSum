from billsum.post_process import greedy_summarize, mmr_selection

import os
import pandas as pd 
import pickle 
from rouge import Rouge
rouge = Rouge()

prefix = os.environ['BILLSUM_PREFIX']

# Note: this script assumes the data was generated using prepare_bert_data.py
# The two go together, because we want the output predictions to match the docs

# Load in predictions

prefix_classifier = os.path.join(prefix, 'bert_data')


############## US ####################

predictions = pd.read_csv(os.path.join(prefix_classifier,'us_test_results.tsv'), sep='\t', header=None)
pos_pred = predictions[1].values


# Load in the sentence data

sent_data = pickle.load(open(prefix + 'sent_data/us_test_sent_scores.pkl', 'rb'))

docs = pd.read_json(prefix + 'clean_final/us_test_data_final.jsonl', lines=True)
docs.set_index('bill_id', inplace=True)


all_scores = {}

doc_order = sorted(sent_data.keys())

i = 0
for bill_id in doc_order:

    sents = sent_data[bill_id]
    
    tot_sent = len(sents)

    # Collect the predictions that correspond to this bill
    ys = pos_pred[i : i+tot_sent]
    i += tot_sent

    # Get sent text
    mysents = [s[0] for s in sents]

    final_sum = ' '.join(mmr_selection(mysents, ys))
    
    score = rouge.get_scores([final_sum], [docs.loc[bill_id].summary])[0]
    all_scores[bill_id] = score
    #rint(score)

pickle.dump(all_scores, open(os.path.join(prefix, 'score_data/us_bert_scores.pkl'), 'wb'))


############## CA ####################

predictions = pd.read_csv(os.path.join(prefix_classifier,'ca_test_results.tsv'), sep='\t', header=None)
pos_pred = predictions[1].values


# Load in the sentence data

sent_data = pickle.load(open(prefix + 'sent_data/ca_test_sent_scores.pkl', 'rb'))

docs = pd.read_json(prefix + 'clean_final/ca_test_data_final.jsonl', lines=True)
docs.set_index('bill_id', inplace=True)


all_scores = {}

doc_order = sorted(sent_data.keys())

i = 0
for bill_id in doc_order:

    sents = sent_data[bill_id]
    
    tot_sent = len(sents)

    # Collect the predictions that correspond to this bill
    ys = pos_pred[i : i+tot_sent]
    i += tot_sent

    # Get sent text
    mysents = [s[0] for s in sents]
    
    final_sum = ' '.join(mmr_selection(mysents, ys))
    
    score = rouge.get_scores([docs.loc[bill_id].clean_summary], [final_sum])[0]
    all_scores[bill_id] = score

pickle.dump(all_scores, open(os.path.join(prefix, 'score_data/ca_bert_scores.pkl'), 'wb'))

