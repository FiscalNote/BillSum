from billsum.classifiers.classifier_scorer import TextScorer
from billsum.post_process import greedy_summarize, mmr_selection
from billsum.utils.sentence_utils import list_to_doc

import numpy as np
import pandas as pd
import pickle

from rouge import Rouge
rouge = Rouge()

prefix = '/Users/anastassiakornilova/BSDATA/'


##########     Load in the data ###############
us_train = pd.read_json(prefix + 'clean_final/us_train_data_final.jsonl', lines=True)
us_train.set_index('bill_id', inplace=True)
us_train_sents = pickle.load(open(prefix + 'sent_data/us_train_sent_scores.pkl', 'rb'))
us_train_summary = pickle.load(open(prefix + 'clean_final/us_train_summary_sents.pkl', 'rb'))

final_train = {}
for bill_id, sents in us_train_sents.items():

    doc = [v[1] for v in sents]
    
    scores = [v[2] for v in sents]
    
    sent_texts = [v[0] for v in sents]

    summary = us_train.loc[bill_id]['clean_summary']

    title = us_train.loc[bill_id]['clean_title']

    mysum = us_train_summary[bill_id]

    final_train[bill_id] = {'doc': doc, 'scores': scores, 'sum_text': summary, 
                             'sent_texts': sent_texts, 'title': title, 'sum_doc': mysum}

del us_train, us_train_sents, us_train_summary

us_test = pd.read_json(prefix + 'clean_final/us_test_data_final.jsonl', lines=True)
us_test.set_index('bill_id', inplace=True)
us_test_sents = pickle.load(open(prefix + 'sent_data/us_test_sent_scores.pkl', 'rb'))
us_test_summary = pickle.load(open(prefix + 'clean_final/us_test_summary_sents.pkl', 'rb'))


final_test = {}
for bill_id, sents in us_test_sents.items():

    doc = [v[1] for v in sents]
    
    scores = [v[2] for v in sents]
    
    sent_texts = [v[0] for v in sents]

    summary = us_test.loc[bill_id]['clean_summary']

    title = us_test.loc[bill_id]['clean_title']

    mysum = us_test_summary[bill_id]

    final_test[bill_id] = {'doc': doc, 'scores': scores, 'sum_text': summary, 
                           'sent_texts': sent_texts, 'title': title, 'sum_doc': mysum,
                           'textlen': len(us_test.loc[bill_id]['text'])}

del us_test, us_test_sents, us_test_summary

################ Learn to score and summarize ############################
tfidf_args = {'stop_words': None,  'use_idf': True, 'binary':False, 
                           'max_features': 50000, 'ngram_range': (1,2), 'min_df':5}

model = FeatureScorer()
model.train(final_train.values())

# Summarizer
final_scores = {}
for bill_id, doc in final_test.items():
    
    scores = model.score_doc(doc)

    final_sum = ' '.join(mmr_selection(doc['sent_texts'], scores, 13333))

    rs = rouge.get_scores([final_sum],[doc['sum_text']])[0]

    final_scores[bill_id] = rs


pickle.dump(final_scores, open('us_test_feature_mmr_2000.pkl', 'wb'))

