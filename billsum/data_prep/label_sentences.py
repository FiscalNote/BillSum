'''
Methods to prepare sentences for extractive sentences training.
'''
import jsonlines
import os
import pickle
import re
from rouge import Rouge
import spacy

nlp = spacy.load('en')
rouge = Rouge()

section_pattern = re.compile('(SECTION)|(Sec)|(Section) [0-9]+')


def spacy_to_tuple(doc):
    text_feats = [(w.string, w.i, w.lemma_, w.ent_type_, w.ent_iob_, w.pos_, w.dep_, w.head.i)
                                    for w in doc]
    return text_feats

def prepare_summary(bill_data):
    
    final_summary_data = {}
    i = 0

    for _, bill in bill_data.iterrows():

        bill_id = bill['bill_id']

        # Keep track of progress
        i += 1
        if i % 100 == 0:
            print("Processed {} summaries".format(i))

        text_nlp = nlp(bill['summary'])

        doc_data = []

        for sent in text_nlp.sents:
            # Store key features from each sentence
            text_feats = spacy_to_tuple(sent)
            doc_data.append(text_feats)
            
        final_summary_data[bill_id] = doc_data

    return final_summary_data


def prepare_labels(bill_data,  min_sent_words=5):
    '''
    Take in a list of data for bills 

    and returns a per sentence score for
    every sentence in each document.

    bill_data: list of dicts where each dict 
            has a summary, bill_id and text field
    
    min_sent_words: skip sentences in text with
        less words.

    Output: dict of bill-id - list of sent data 
        where sent data is a three tuple of original sentence, list of word 
        with spacy annotations and the rscores of that sentence relative to the summary.

        The annotations follow the format of utils.sentence_utils.Word

    '''

    final_scores = {}
    i = 0

    for _, bill in bill_data.iterrows():

        bill_id = bill['bill_id']

        # Keep track of progress
        if i % 100 == 0:
            print(i, len(final_scores))

        i += 1

        text_nlp = nlp(bill['clean_text'])

        sent_data = []

        for sent in text_nlp.sents:

            # Skip sents with less than 5 words
            if len(sent) > min_sent_words:
            
                # Store key features from each sentence
                text_feats = spacy_to_tuple(sent)
            
                # Create rouge scores                  
                if len(sent.string) == 0 or len(bill['clean_summary']) == 0:
                    continue
                
                rscores = rouge.get_scores([sent.string],[bill['clean_summary']])[0]

                sent_data.append((sent.string, text_feats, rscores))


        final_scores[bill_id] = sent_data

    return final_scores


if __name__ == '__main__':
    import pandas as pd 

    prefix = os.environ['BILLSUM_PREFIX']

    if not prefix.endswith('/'):
        prefix += '/'

    #os.mkdir(prefix + 'sent_data/')

    print("Preparing US Train")
    data = pd.read_json(prefix + 'clean_final/us_train_data_final.jsonl', lines=True)
    sent_scores = prepare_labels(data)
    pickle.dump(sent_scores, open(prefix + 'sent_data/us_train_sent_scores.pkl', 'wb'))

    sum_sents = prepare_summary(data)
    pickle.dump(sum_sents, open(prefix + 'sent_data/us_train_sum_sents.pkl', 'wb'))


    print("Preparing US Test")
    data = pd.read_json(prefix + 'clean_final/us_test_data_final.jsonl', lines=True)
    sent_scores = prepare_labels(data)
    pickle.dump(sent_scores, open(prefix + 'sent_data/us_test_sent_scores.pkl', 'wb'))

    sum_sents = prepare_summary(data)
    pickle.dump(sum_sents, open(prefix + 'sent_data/us_test_sum_sents.pkl', 'wb'))


    print("Preparing CA Test")
    data = pd.read_json(prefix + 'clean_final/ca_test_data_final.jsonl', lines=True)
    sent_scores = prepare_labels(data)
    pickle.dump(sent_scores, open(prefix + 'sent_data/ca_test_sent_scores.pkl', 'wb'))

    sum_sents = prepare_summary(data)
    pickle.dump(sum_sents, open(prefix + 'sent_data/ca_test_sum_sents.pkl', 'wb'))


