"""
Methods which transform Spacy Docs into numeric vectors. 
Will be used as the text_transformer in FeatureTransformer
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

import logging
_log = logging.getLogger(__name__)

# Strange but important util functions
def noop(X):
    # A hacky way to stop tfidf vectorizer from redoing preprocessor
    return X

def tokenize(X):
    # A hacky way to stop tfidf vectorizer from redoing this step
    return X.split()

class SpacyTfidfWrapper(object):
    """
    A wrapper around sklearn's tf-idf vectorizer that first transforms Spacy 
    into plain text 
    """

    def __init__(self, lemmatize=True, only_alpha=True, tfidf=None, tfidf_args=None):
        """
        tfidf: instance of TfidfVectorizer
        lemmatize: lemmatie before passing to tfidf
        only_alpha: filter non-alphabetic 'words' before passing to tfidf

        tfidf_args: dict of args to pass as keywords to tfidf 
        """
        self.lemmatize = lemmatize
        self.only_alpha = only_alpha

        self.tfidf_args = {'stop_words': 'english',
                          'ngram_range': (1,2),
                          'min_df': 5,
                          'max_df': 0.95,
                          'preprocessor':noop,
                          'token_pattern':'(?u)\\b[A-Za-z][A-Za-z]+\\b',
                          'use_idf': True}

        # Update default args with the new args passed in
        if tfidf_args is not None:
            self.tfidf_args.update(tfidf_args)

        self.tfidf = tfidf

    def prep_sent(self, sent):
        
        final_words = []
        for word in sent:
            if self.lemmatize:
                final_words.append(word[2]) #.lemma
            else:
                final_words.append(word[0]) #.text
    
        return " ".join(final_words)

    def prep_doc(self, doc):

        final_words = []
        for sent in doc:
            for word in sent:
                if self.lemmatize:
                    #final_words.append(word.lemma_)
                    final_words.append(word[2])
                else:
                    #final_words.append(word.text)
                    final_words.append(word[0])

        
        return " ".join(final_words)

    def fit(self, docs, sent_as_doc=False):
        '''
        Sent_as_doc will treat every sentence in each 
        document as a separate document
        '''

        if sent_as_doc:
            texts = [self.prep_sent(sent)
                        for doc in docs for sent in doc]
        else:
            texts = [self.prep_doc(doc) for doc in docs]
        
        # We customize the TfidfVectorizer for speed - it usually does a 
        # preprocess and tokenize step, but SPACY already did that for us, 
        # so we essentially skip both steps
        # Written w/ explicit not lamda functions to support pickling
        self.tfidf = TfidfVectorizer(**self.tfidf_args)
        self.tfidf.fit(texts)
        return self.tfidf

    def transform(self, docs):
        """
        Vectorize a series of Spacy Docs 
        """
        texts = [self.prep_doc(doc) for doc in docs]
        return self.tfidf.transform(texts) 

    def transform_by_sent(self, docs):
        """
        Treats each sentence like a separate document
        """

        final_vectors = []

        sent_texts = [self.prep_sent(sent) for doc in docs for sent in doc]
        sent_vectors = self.tfidf.transform(sent_texts)

        return sent_vectors

