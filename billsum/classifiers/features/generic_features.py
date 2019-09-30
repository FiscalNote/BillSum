"""
Simple bill features that describe surface properties of sentences, and usually produce a binary/single-value feature 

All features extend the GenericFeature class - defined in this file
"""
from collections import Counter, defaultdict
import math
import numpy as np
import operator
import pickle

class GenericFeature(object):
    """
    An example of a feature generating class - contains the basics that all 
    other classes will extend. 
    """

    def __init__(self, use_spacy=False):
        
        """
        is_sparse specifes if the make_features class will return a numpy array 
        or a sparse matrix. - Important for feature combination!

        use_spacy: indicates whether inputs will be spacy docs/sents OR 
        lists of strings
        """
        self.is_sparse = False
        self.use_spacy = use_spacy

    def fit(self, docs, *args, **kwargs):
        '''
        Placeholder for any global features - for example idf calculation over all text
        '''
        pass

    def prepare_doc(self, doc, *args, **kwargs):
        """
        Will be called before each doc is processed (in this set-up documents
         are processed iteratively)

        Allows for document level data to be stored and not recomputed 
            with every call to make_features

        doc: a Spacy Document - passed in to precompute things for whole doc
        tfidf_vecs: optional tf-idf vectors for every sentence 

        Note: Actual DOC should not be saved, otherwise may cause issues in feature transformer
        """
        pass

    def make_features(self, i, sent):
        """
        i: index of sent in the doc.
        sent: a Spacy Span representing a sentence in the current doc

        returns either a numpy array or a sparse matrix, depending on self.is_sparse
        """
        raise NotImplementedError

    def make_all_features(self, doc, *args, **kwargs):
        '''
        Wrapper to combine preparing doc and sentence transformations
        '''
        # Call prepare first just in case 
        self.prepare_doc(doc, *args, **kwargs)
        return [self.make_features(i, s) for i, s in enumerate(doc.sents)]


class SentencePosF(GenericFeature):
    is_sparse = False

    def prepare_doc(self, doc, *args, **kwargs):
        self.doc_length = len((list(doc.sents)))
        self.sparse = False
    
    def make_features(self, i, sent):
        """
        Return the position of sentence in the doc.
        
        The index of the sentence is divided by the total number of sentences 
            to get a scaled constant.
        """
        return [i * 1. / self.doc_length]


class NearSectionStartF(GenericFeature):

    def prepare_doc(self, doc, *args, **kwargs):

        # Find section headers
        self.section_ids = []
        i = 0
        for sent in doc.sents:
            text = ''.join(w.text for w in sent)
            if '<SECTION-HEADER>' in text:
                self.section_ids.append(i)
            i += 1

    def make_features(self, i, sent):

        for pos in self.section_ids:
            if i > pos and i <= pos + 2:
                return [1]
        return [0]


class IsLongF(GenericFeature):
    """
    A binary feature to identify long sentences. Long is defined as longer 
        than cutoff_length 
    """
    
    def __init__(self, cutoff_lengths=[10, 20]):
        self.cutoff_lengths = cutoff_lengths
        self.is_sparse = False
    
    def make_features(self, i, sent):
        return [int(len(sent.words) > cl) for cl in self.cutoff_lengths]


class HasNerF(GenericFeature):
    """
    A binary feature to indicate if the sentence has any NERs

    TODO: make into count of each type
    """
    enttypes = ['PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC',
                'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LAW', 
                'LANGUAGE', 'DATE', 'TIME', 'PERCENT', 'MONEY',
                'QUANTITY', 'ORDINAL', 'CARDINAL']
    def make_features(self, i, sent):

        # count the NERs as labeled by Spacy.
        ner_count = [w.ent_type_ for w in sent if w.ent_iob_ == 'B']
        return [b in ner_count for b in self.enttypes] + [len(ner_count) > 0]
   
        
class SecretaryF(GenericFeature):

    def make_features(self, i, sent):

        s = ''.join(w.text.lower() for w in sent)
        if '<section-header>retary' in s:
            return [1]
        good = ['secretary', 'director', 'administrator', 'attorney']
        return [any(g in s for g in good)]


