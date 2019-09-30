from billsum.classifiers.features.generic_features import GenericFeature
from billsum.classifiers.text_transformer import SpacyTfidfWrapper


import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

#from nltk.probability import FreqDist, LaplaceProbDist, KneserNeyProbDist, LidstoneProbDist, WittenBellProbDist
#from nltk.tokenize import word_tokenize
#from nltk.util import ngrams
import pickle

class GlobalTfidfF(GenericFeature):
    """
    Calculates the average TF-IDF values for the words in the sentence. 
    The word score is the TF-IDF score over the whole document - since
    different sentences contain different subsets of words, their feature will be different. 
    """
    def __init__(self, tfidf_args=None, text_transformer=None):

        if not text_transformer:
            self.text_transformer = SpacyTfidfWrapper(tfidf_args={'use_idf': True, 'min_df': 5, 'max_df': 0.95,
                'max_features': 10000, 'ngram_range': (1,2), 'stop_words': 'english', 'binary': True})
        else:
            self.text_transformer = text_transformer

    def fit(self, docs, by_sent=False, **kwargs):
        # Fit the IDF on all the training docs 
        self.text_transformer.fit(docs)

    def prepare_doc(self, doc, **kwargs):
        # Compute tf-idf of each word 
        self.doc_tfidfs = self.text_transformer.transform([doc])
    
    def make_features(self, i, sent):
        # Use transformer to figure out which words are in this sentence 
        vec = self.text_transformer.transform_by_sent([[sent]])
        
        if vec.sum() == 0:
            return [0,0]

        # Average non-zero doc tfidf values        
        final_vec = self.doc_tfidfs[vec.nonzero()]

        return [final_vec.mean(), final_vec.max()]
        

class DocTfidfF(GenericFeature):
    ''' 
    Calculate the probability of each word in the document 
    
    Returns statistics on the probability of words in each sent
    '''

    def __init__(self, average=True ):

        self.text_transformer = SpacyTfidfWrapper(tfidf_args={'use_idf': False, 'min_df': 1, 'max_df': 1.0})

    def prepare_doc(self, doc, **kwargs):
        # Fit each sentence as a document
        self.text_transformer.fit([doc], sent_as_doc=True)
        counts = self.text_transformer.transform([doc])

        self.word_probs = counts / counts.sum()

    def make_features(self, i, sent):
        vec = self.text_transformer.transform_by_sent([[sent]])
        if vec.sum() == 0:
            return [0,0]

        final_vec = self.word_probs[vec.nonzero()]
        return [final_vec.mean(), final_vec.max()]


class KLSummaryF(GenericFeature):
    '''
    Features related to log likelihood and also log ratio 
    Estimate how summary-like the words in this sentence are. The computation is 
    based on computing the "likelihood" of this word appearing in a summary vs 
    a text. 

    The feature is the average/max of the scores of the concepts in the sentence.

    '''
    def __init__(self):

        self.offset =  0.000005

    def fit(self, docs, summaries=None):
        # Count all words 
        self.text_transformer = SpacyTfidfWrapper(tfidf_args={ 'ngram_range':(1,1),'stop_words': 'english', 'use_idf': False, 'norm':None, 'min_df':1, 'max_df':1.})
        
        self.text_transformer.fit(docs + summaries)

        # Calculate probability distributions 
        X = self.text_transformer.transform(docs)
        Xsum = self.text_transformer.transform(summaries)
        # Make approximate probabilities
        self.word_prob_text = X.sum(axis=0) / X.sum() + self.offset
        self.word_prob_sum = Xsum.sum(axis=0) / Xsum.sum() + self.offset
        self.kl_text_sum =  np.multiply(self.word_prob_text, 
                                np.log(self.word_prob_text  / self.word_prob_sum))
        self.kl_sum_text = np.multiply(self.word_prob_sum, 
                                np.log(self.word_prob_sum  / self.word_prob_text))

        # Separate out 100 most distinct words from ech
        self.text_like_words = set(self.kl_text_sum.A.argsort()[0][-100:])
        self.sum_like_words = set(self.kl_sum_text.A.argsort()[0][-100:])


    def prepare_doc(self, doc, **kwargs):

        # Vectorize current document
        self.cur_vec = self.text_transformer.transform_by_sent([doc])
        
        # Calculate KL divergences of each sentence
        self.vec_kl_sum = self.cur_vec * self.kl_sum_text.T
        self.vec_kl_text = self.cur_vec * self.kl_text_sum.T
        

    def make_features(self, i, sent):

        my_words = self.cur_vec[i].nonzero()[1]
        
        if len(my_words) == 0:
            return [0] * 6

        sum_like_count = len(self.sum_like_words.intersection(my_words))
        text_like_count = len(self.text_like_words.intersection(my_words))

        return [self.vec_kl_sum[i,0], self.vec_kl_text[i,0], 
                sum_like_count > 0, sum_like_count / len(my_words),
                text_like_count > 0, text_like_count / len(my_words)]




