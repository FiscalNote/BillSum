from billsum.classifiers.text_transformer import SpacyTfidfWrapper

import numpy as np
from sklearn.linear_model import LogisticRegression


class TextScorer:
    '''
    Model wrapper for text based features
    '''

    def __init__(self, tfidf_args={}, classifier=None, score=('rouge-2', 'p')):


        self.tfidf_args = {'stop_words': 'english',  'use_idf': False, 'binary':True, 
                           'max_features': 50000, 'ngram_range': (1,2), 'min_df':5,}

        self.tfidf_args.update(tfidf_args)
        self.tfidf = SpacyTfidfWrapper(tfidf_args=self.tfidf_args)

        if classifier is None:
            self.clf = LogisticRegression()
        else:
            self.clf = classifier

        self.score = score

        self.score_threshold = 0.1

    def train(self, train_docs):
        tdocs = [d['doc'] for d in train_docs]
        self.tfidf.fit(tdocs)
        
        print('Text fit')

        X = self.tfidf.transform_by_sent(tdocs)

        rtype, mtype = self.score

        y_train = np.array([y[rtype][mtype] for d in train_docs for y in d['scores']])
        y_train2 = y_train > self.score_threshold

        self.clf.fit(X, y_train2)
        print("Classifier fit:", self.clf.score(X, y_train2), y_train2.mean())

    def score_doc(self, test_doc):

        myX = self.tfidf.transform_by_sent([test_doc['doc']])
        y_pred = self.clf.decision_function(myX)

        return y_pred



if __name__ == '__main__':

    data = pickle.load(open(''))
