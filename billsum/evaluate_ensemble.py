'''
Script to combine feature model and bert model predictions
'''


from billsum.post_process import greedy_summarize, mmr_selection

import os
import pandas as pd 
import pickle 
from rouge import Rouge
rouge = Rouge()


prefix = os.environ['BILLSUM_PREFIX']
if not prefix.endswith('/'):
    prefix += '/'


# Note: this script assumes the data was generated using prepare_bert_data.py
# The two go together, because we want the output predictions to match the docs

# Load in predictions

# Stored during the train_wrapper.py script
feature_model = pickle.load(open(prefix + '/models/feature_scorer_model.pkl', 'rb'))

############## US ####################


for locality in ['us', 'ca']:

    # Load in the sentence data
    sent_data = pickle.load(open(prefix + 'sent_data/{}_test_sent_scores.pkl'.format(locality), 'rb'))

    docs = pd.read_json(prefix + 'clean_final/{}_test_data_final.jsonl'.format(locality), lines=True)
    docs.set_index('bill_id', inplace=True)

    print(len(sent_data), len(docs), len(bert_pred))
    all_scores = {}

    # Load in bert predictions

    predictions = pd.read_csv(prefix + 'bert_data/{}_test_results.tsv'.format(locality), sep='\t', header=None)
    bert_pred = predictions[1].values

    doc_order = sorted(sent_data.keys())

    i = 0
    for bill_id in doc_order:

        sents = sent_data[bill_id]

        doc = [s[1] for s in sents]

        # Weird access mode - to deal with model signature
        feat_y = feature_model.score_doc({'doc':doc})

        # Collect the predictions that correspond to this bill
        # Bert Y
        tot_sent = len(sents)
        bert_y = bert_pred[i : i+tot_sent]
        i += tot_sent

        # Get sent text
        Y = (feat_y + bert_y) / 2

        mysents = [s[0] for s in sents]

        final_sum = ' '.join(mmr_selection(mysents, Y))
        
        rscore = rouge.get_scores( [final_sum], [docs.loc[bill_id].summary])[0]
        all_scores[bill_id] = rscore


    pickle.dump(all_scores, open(prefix + 'score_data/{}_ensemble_scores.pkl'.format(locality), 'wb'))
    