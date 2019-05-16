from billsum.post_process import greedy_summarize, mmr_selection

import pandas as pd 
import pickle 
from rouge import Rouge
rouge = Rouge()

prefix = os.path.expanduser('~/BSDATA/')

# Note: this script assumes the data was generated using prepare_bert_data.py
# The two go together, because we want the output predictions to match the docs

# Load in predictions

prefix_classifier = "BERT_CLASSIFIER_DIR" # EDIT ME


############## US ####################

predictions = pd.read_csv(prefix_classifier + 'test_results.tsv', sep='\t', header=None)
pos_pred = predictions[1].values


# Load in the sentence data

sent_data = pickle.load(open(prefix + 'us_test_sent_scores.pkl', 'rb'))

docs = pd.read_json(prefix + '/clean_final/us_test_data_final.jsonl', lines=True)
docs.set_index('bill_id', inplace=True)


all_scores = {}

doc_order = sorted(sent_data.keys())

i = 0
for bill_id in doc_order:

	sents = sent_data[bill_id]
    
    tot_sent = len(sents)

    # Collect the predictions that correspond to this bill
    ys = sent_scores[i:i+tot][1].values
    i += tot

    # Get sent text
    mysents = [s[0] for s in sents]
    
    final_sum = ' '.join(mmr_selection(mysents, ys))
    
    score = rouge.get_scores([docs.loc[key].clean_summary], [final_sum])[0]
    all_scores[key] = score

pickle.dump(prefix + 'score_data/us_bert_scores.pkl', 'rb')



############## CA ####################

predictions = pd.read_csv(prefix_classifier + 'ca_test_results.tsv', sep='\t', header=None)
pos_pred = predictions[1].values


# Load in the sentence data

sent_data = pickle.load(open(prefix + 'ca_test_sent_scores.pkl', 'rb'))

docs = pd.read_json(prefix + '/clean_final/ca_test_data_final.jsonl', lines=True)
docs.set_index('bill_id', inplace=True)


all_scores = {}

doc_order = sorted(sent_data.keys())

i = 0
for bill_id in doc_order:

	sents = sent_data[bill_id]
    
    tot_sent = len(sents)

    # Collect the predictions that correspond to this bill
    ys = sent_scores[i:i+tot][1].values
    i += tot

    # Get sent text
    mysents = [s[0] for s in sents]
    
    final_sum = ' '.join(mmr_selection(mysents, ys))
    
    score = rouge.get_scores([docs.loc[key].clean_summary], [final_sum])[0]
    all_scores[key] = score

pickle.dump(prefix + 'score_data/ca_bert_scores.pkl', 'rb')