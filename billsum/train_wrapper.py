from billsum.classifiers.classifier_scorer import TextScorer
from billsum.post_process import greedy_summarize
from billsum.utils.sentence_utils import list_to_doc

import numpy as np
import pandas as pd
import pickle


prefix = '/Users/anastassiakornilova/BSDATA/'

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

    final_test[bill_id] = {'doc': doc, 'scores': scores, 'sum_text': summary, 'sent_texts': sent_texts, 'title': title}

del us_test, us_test_sents

################ Learn to score and summarize ############################
from billsum.nn_model.lstm_model import LSTMWrapper

model = LSTMWrapper()
model.train(list(final_train.values()))
pickle.dump(model, open('mymodel.pkl', 'wb'))

# Summarizer
#final_scores = {}
#for bill_id, doc in final_test.items():

	# scores = model.score(doc)

	# final_sum = ' '.join(greedy_summarize(doc['sent_texts'], scores))

	# rs = rouge.get_scores([final_sum],[doc['sum_text']])[0]

	# final_scores[bill_id] = rs


