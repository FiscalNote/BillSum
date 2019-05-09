from billsum.classifiers.classifier_scorer import TextScorer
from billsum.post_process import greedy_summarize, mmr_selection
from billsum.utils.sentence_utils import list_to_doc

import numpy as np
import pandas as pd
import pickle

from rouge import Rouge
rouge = Rouge()

prefix = '/data/billsum/'


##########     Load in the data ###############
us_train = pd.read_json(prefix + 'clean_final/us_train_data_final.jsonl', lines=True)
us_train.set_index('bill_id', inplace=True)
us_train_sents = pickle.load(open(prefix + 'sent_data/us_train_sent_scores.pkl', 'rb'))


final_train = {}
for bill_id, sents in us_train_sents.items():

    doc = [v[1] for v in sents]
    
    scores = [v[2] for v in sents]
    
    sent_texts = [v[0] for v in sents]

    summary = us_train.loc[bill_id]['clean_summary']

    title = us_train.loc[bill_id]['clean_title']

    final_train[bill_id] = {'doc': doc, 'scores': scores, 'sum_text': summary, 'sent_texts': sent_texts, 'title': title}
del us_train, us_train_sents

us_test = pd.read_json(prefix + 'clean_final/us_test_data_final.jsonl', lines=True)
us_test.set_index('bill_id', inplace=True)
us_test_sents = pickle.load(open(prefix + 'sent_data/us_test_sent_scores.pkl', 'rb'))


final_test = {}
for bill_id, sents in us_test_sents.items():

    doc = list_to_doc([v[1] for v in sents])
    
    scores = [v[2] for v in sents]
    
    sent_texts = [v[0] for v in sents]

    summary = us_test.loc[bill_id]['clean_summary']

    title = us_test.loc[bill_id]['clean_title']

    final_test[bill_id] = {'doc': doc, 'scores': scores, 'sum_text': summary, 'sent_texts': sent_texts, 'title': title, 'textlen': len(us_test.loc[bill_id]['text'])}

del us_test, us_test_sents

################ Learn to score and summarize ############################
model = TextScorer()
model.train(final_train.values())

# Summarizer
final_scores = {}
for bill_id, doc in final_test.items():
    
    scores = model.score_doc(doc)
    #scores = [s['rouge-2']['p'] for s in doc['scores']]

    final_sum = ' '.join(mmr_selection(doc['sent_texts'], scores, 13333))

    rs = rouge.get_scores([final_sum],[doc['sum_text']])[0]

    final_scores[bill_id] = rs


pickle.dump(final_scores, open(prefix + 'score_data/text_mmr_2000.pkl', 'wb'))

