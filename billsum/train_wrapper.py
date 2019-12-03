'''
Wrapper to train and evaluate a supervised model
'''
from billsum.classifiers.classifier_scorer import FeatureScorer
from billsum.post_process import greedy_summarize, mmr_selection
from billsum.utils.sentence_utils import list_to_doc

import numpy as np
import os
import pandas as pd
import pickle

from rouge import Rouge
rouge = Rouge()

prefix = os.environ['BILLSUM_PREFIX']

if not prefix.endswith('/'):
    prefix += '/'

##########     Load in the data ###############
us_train = pd.read_json(prefix + 'clean_final/us_train_data_final.jsonl', lines=True)
us_train.set_index('bill_id', inplace=True)
us_train_sents = pickle.load(open(prefix + 'sent_data/us_train_sent_scores.pkl', 'rb'))

us_train_sum_sents = pickle.load(open(prefix + 'sent_data/us_train_sum_sents.pkl', 'rb'))


# Store data in the same order
final_train = []
final_train_sum = []

i = 0
for bill_id, sents in us_train_sents.items():

    doc = [v[1] for v in sents]
    
    scores = [v[2] for v in sents]
    
    sent_texts = [v[0] for v in sents]

    summary = us_train.loc[bill_id]['clean_summary']

    title = us_train.loc[bill_id]['clean_title']

    final_train.append({'doc': doc, 'scores': scores, 'sum_text': summary, 
                             'sent_texts': sent_texts, 'title': title})

    final_train_sum.append(us_train_sum_sents[bill_id])

del us_train, us_train_sents, us_train_sum_sents

######## Train a model ###################

model = FeatureScorer()
model.train(final_train, final_train_sum)

model_path = os.path.join(prefix, 'models')

if not os.path.exists(model_path):
    os.makedirs(model_path)

pickle.dump(model, open(os.path.join(model_path, 'feature_scorer_model.pkl'), 'wb'))

#model = pickle.load(open(prefix + 'models/feature_scorer_model.pkl', 'rb'))

######### Evaluate Performance ################

for locality in ['us', 'ca']:

    # Load in the data
    test_data = pd.read_json(prefix + 'clean_final/{}_test_data_final.jsonl'.format(locality), lines=True)

    # CA data gets formatted weird - fix just in case
    if 'bill_id' not in test_data.columns:
        test_data['bill_id'] = test_data['external_id']

    test_data.set_index('bill_id', inplace=True)
    test_sents = pickle.load(open(prefix + 'sent_data/{}_test_sent_scores.pkl'.format(locality), 'rb'))

    final_test = {}
    for bill_id, sents in test_sents.items():

        doc = [v[1] for v in sents]
        
        scores = [v[2] for v in sents]
        
        sent_texts = [v[0] for v in sents]

        summary = test_data.loc[bill_id]['summary']

        final_test[bill_id] = {'doc': doc, 'scores': scores, 'sum_text': summary, 
                               'sent_texts': sent_texts, 
                               'textlen': len(test_data.loc[bill_id]['text'])}

    del test_data, test_sents

    # Evaluation
    final_scores = {}
    for bill_id, doc in final_test.items():
        
        # Create and score features
        scores = model.score_doc(doc)

        final_sum = ' '.join(mmr_selection(doc['sent_texts'], scores))

        rs = rouge.get_scores([final_sum],[doc['sum_text']])[0]

        final_scores[bill_id] = rs

    pickle.dump(final_scores, open(os.path.join(prefix, 'score_data/{}_test_feature_model_res.pkl'.format(locality)), 'wb'))

