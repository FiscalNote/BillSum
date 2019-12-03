'''
Script to run the generic baselines over the data
'''
from billsum.post_process import greedy_summarize

from collections import defaultdict
import jsonlines
import numpy as np
import os
import pickle

from rouge import Rouge

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.kl import KLSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.sum_basic import SumBasicSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer

from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words


ALL_SUMMARIZERS = [('sumbasic', SumBasicSummarizer),
                   ('textrank', TextRankSummarizer),
                   ('kl', KLSummarizer),
                   ('lsa', LsaSummarizer)]

rouge = Rouge()

LANGUAGE = 'en'
stemmer = Stemmer(LANGUAGE)

prefix = os.environ['BILLSUM_PREFIX']
prefix2 = os.path.join(prefix,"score_data/")

if not os.path.exists(model_path):
    os.makedirs(prefix2)

# import warnings
# warnings.filterwarnings("error")

import sys 
print(sys.getrecursionlimit())
sys.setrecursionlimit(5000)

for file in os.listdir(os.path.join(prefix, 'clean_final')):
    path = os.path.join(os.path.join(prefix, 'clean_final', file))
    data = []

    if 'train' in file:
        continue

    with jsonlines.open(path) as reader:
        for obj in reader: 
            data.append(obj)

    all_scores = defaultdict(dict)

    i = 0

    final_documents = {}
    for bill in data:
        i += 1
        if i % 50 == 0:
            print(i)
            
        summary = bill['summary']
        doc = bill['clean_text']
        bill_id = bill['bill_id']

        doc2 = PlaintextParser(doc, Tokenizer(LANGUAGE)).document
        for name, Summarizer in ALL_SUMMARIZERS:
            try:
                summarizer = Summarizer(stemmer)
                #summarizer.stop_words = get_stop_words(LANGUAGE)

                # Score all sentences -- then keep up to 2000 char
                total_sentences = len(doc2.sentences)
                sent_scores = summarizer(doc2, total_sentences)
                sent_scores = [(str(s.sentence), s.rating) for s in sent_scores]

                # Pick best set with greedy
                
                summary_len = 2000
                final_sents = greedy_summarize(*zip(*sent_scores), summary_len=summary_len)
                final_sum = ' '.join(final_sents)
                score = rouge.get_scores([summary], [final_sum])[0]
                all_scores[bill_id][name] = score
            
            except KeyboardInterrupt:
                raise KeyboardInterrupt
        
            #except:
            #    print("Bad bill", bill_id)
    
    pickle.dump(all_scores, open(os.path.join(prefix2, 'baseline_{}_2000.pkl'.format(file)), 'wb'))

