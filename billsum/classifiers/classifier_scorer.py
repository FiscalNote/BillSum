from billsum.classifiers.features.generic_features import *
from billsum.classifiers.features.tfidf_features import *
from billsum.classifiers.text_transformer import SpacyTfidfWrapper
from billsum.utils.sentence_utils import list_to_doc

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression



class FeatureScorer:

    def __init__(self, classifier=None, score_type=('rouge-2', 'p')):
        
        self.feats = [ NearSectionStartF(), SentencePosF(), GlobalTfidfF(), DocTfidfF(), KLSummaryF()]

        if classifier is None:
            self.clf = RandomForestClassifier(min_samples_split=10, n_estimators=50)
        else:
            self.clf = classifier

        self.score_type = score_type

        self.score_threshold = 0.1

    def create_features(self, doc):

        all_feats = []

        for f in self.feats:
            f.prepare_doc(doc)
            all_feats.append(f.make_all_features(doc))

        all_feats = np.hstack(all_feats)

        return all_feats

    def train(self, train_docs, summaries=None):
        
        # Transform sentences into our custom format
        new_docs = [list_to_doc(doc['doc']) for doc in train_docs]

        for f in self.feats:
            f.fit(new_docs, summaries)

        all_features = [self.create_features(doc) for doc in new_docs]

        X = np.vstack(all_features)

        
        rtype, mtype = self.score_type

        y_train = np.array([y[rtype][mtype] for d in train_docs for y in d['scores']])
        y_train2 = y_train > self.score_threshold

        self.clf.fit(X, y_train2)
        print("Classifier fit:", self.clf.score(X, y_train2), y_train2.mean())

    def score_doc(self, doc):

        doc = list_to_doc(doc['doc'])
        X = self.create_features(doc)

        return self.clf.predict_proba(X)[:,1]


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


